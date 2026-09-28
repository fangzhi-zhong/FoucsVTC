#!/usr/bin/env python3
"""OpenAI-compatible gateway that runs the GRPO-trained VTC zoom loop.

The backend remains an ordinary vLLM OpenAI server.  This gateway injects the
same tool schema and prompts used by GRPO, executes ``zoom_region`` locally,
and returns the terminal answer as a normal one-shot chat completion.  It lets
the existing benchmark clients use the learned agent without each benchmark
implementing a subtly different tool protocol.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import importlib.util
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from aiohttp import ClientSession, ClientTimeout, web
from PIL import Image

try:
    from .config import load_config
except ImportError:  # direct script execution
    from config import load_config


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
DPI_RE = re.compile(r"(?<=/)dpi_\d+(?=/)")


def _append_text(content: Any, suffix: str) -> Any:
    if isinstance(content, str):
        return content if content.endswith(suffix) else content + suffix
    if not isinstance(content, list):
        raise ValueError(f"unsupported user content: {type(content)!r}")
    result = copy.deepcopy(content)
    if result and result[-1].get("type") == "text":
        if not result[-1].get("text", "").endswith(suffix):
            result[-1]["text"] = result[-1].get("text", "") + suffix
    else:
        result.append({"type": "text", "text": suffix})
    return result


def _prepare_messages(messages: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(messages)
    expected = config["system_prompt"]
    system_indices = [i for i, message in enumerate(messages) if message.get("role") == "system"]
    if system_indices:
        index = system_indices[0]
        current = str(messages[index].get("content", ""))
        if current.rstrip(". ") == expected.rstrip(". "):
            messages[index]["content"] = expected
        elif not current.startswith(expected + "\n\n"):
            messages[index]["content"] = f"{expected}\n\n{current}"
    else:
        messages.insert(0, {"role": "system", "content": expected})

    user_indices = [i for i, message in enumerate(messages) if message.get("role") == "user"]
    if not user_indices:
        raise ValueError("request has no user message")
    index = user_indices[-1]
    messages[index]["content"] = _append_text(messages[index].get("content", ""), config["turn_prompt"])
    return messages


def _image_sources(messages: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                value = part.get("image_url")
                url = value.get("url") if isinstance(value, dict) else value
                if isinstance(url, str):
                    sources.append(url)
    return sources


def _prefer_high_res(source: str, dpi: int | None, required: bool = False) -> str:
    if dpi is None or not source.startswith("file://"):
        return source
    parsed = urlparse(source)
    path = Path(unquote(parsed.path))
    mapped, substitutions = DPI_RE.subn(f"dpi_{dpi}", str(path), count=1)
    if not substitutions:
        if required:
            raise FileNotFoundError(f"cannot map page path to DPI{dpi}: {path}")
        return source
    candidate = Path(mapped)
    if candidate.is_file():
        return candidate.as_uri()
    if required:
        raise FileNotFoundError(f"required DPI{dpi} zoom page is missing: {candidate}")
    return source


async def _load_image(source: str, session: ClientSession) -> Image.Image:
    if source.startswith("data:"):
        _, payload = source.split(",", 1)
        raw = base64.b64decode(payload)
    elif source.startswith("file://"):
        raw = Path(unquote(urlparse(source).path)).read_bytes()
    elif source.startswith(("http://", "https://")):
        async with session.get(source) as response:
            response.raise_for_status()
            raw = await response.read()
    else:
        raw = Path(source).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB").copy()


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    # Training passes the PIL crop directly to the processor. PNG is the
    # lossless wire equivalent; JPEG would alter the DPI144 evidence pixels.
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _trace_source(source: str) -> str:
    """Keep a useful source identifier without copying inline image bytes."""
    if source.startswith("data:"):
        header = source.split(",", 1)[0]
        return header + ",<omitted>"
    return source


def _assistant_raw(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    raw = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
    if "<tool_call>" not in raw:
        calls = message.get("tool_calls") or []
        for call in calls:
            function = call.get("function", call)
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            payload = {"name": function.get("name"), "arguments": arguments}
            raw += "\n<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
    return raw


def _final_answer(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    matches = ANSWER_RE.findall(content)
    if matches:
        return matches[-1].strip()
    if "</think>" in content:
        return content.rsplit("</think>", 1)[1].strip()
    return content.strip()


def _usage_add(total: dict[str, int], response: dict[str, Any]) -> None:
    usage = response.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += int(usage.get(key) or 0)


class Gateway:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        module_path = Path(config["grpo_root"]) / "verl/workers/agent/envs/mm_process_engine/vtc_zoom.py"
        spec = importlib.util.spec_from_file_location("vtc_eval_zoom", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load training zoom module: {module_path}")
        zoom = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zoom)
        self.ZoomToolError = zoom.ZoomToolError
        self.pad_crop = zoom._pad_to_max_aspect_ratio
        self.extract_tool_call = zoom.extract_tool_call
        self.validate_payload = zoom._validate_payload
        self.normalized_bbox_to_pixels = zoom.normalized_bbox_to_pixels
        self.tools = json.loads(Path(config["tool_schema_path"]).read_text(encoding="utf-8"))
        self.backend = config["backend_base_url"].rstrip("/")
        self.session: ClientSession | None = None

    async def start(self, _: web.Application) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=3600))

    async def stop(self, _: web.Application) -> None:
        if self.session is not None:
            await self.session.close()

    async def health(self, _: web.Request) -> web.Response:
        assert self.session is not None
        try:
            async with self.session.get(self.backend.removesuffix("/v1") + "/health") as response:
                if response.status != 200:
                    return web.json_response({"status": "backend_unhealthy"}, status=503)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"status": "backend_unreachable", "error": repr(exc)}, status=503)
        return web.json_response({"status": "ok", "benchmark": self.config.get("benchmark")})

    async def models(self, _: web.Request) -> web.Response:
        assert self.session is not None
        async with self.session.get(self.backend + "/models") as response:
            return web.Response(body=await response.read(), status=response.status, content_type="application/json")

    async def _backend_call(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        assert self.session is not None
        async with self.session.post(self.backend + "/chat/completions", json=body) as response:
            payload = await response.json(content_type=None)
            return response.status, payload

    async def _token_count(self, messages: list[dict[str, Any]], *, answer_only: bool = False) -> int:
        """Count the exact rendered multimodal prompt used by the backend."""
        assert self.session is not None
        body = {
            "model": self.config["served_model_name"],
            "messages": messages,
            "tools": [] if answer_only else self.tools,
            "chat_template_kwargs": {"enable_thinking": not answer_only},
            # Keep /tokenize's multimodal processor-cache entries separate
            # from generation. vLLM's frontend cache otherwise treats a
            # tokenize hit as if its image features had also reached the
            # engine-side receiver cache. This is the checkpoint's explicit
            # default, so the rendered token count is unchanged while the
            # multimodal cache hash is isolated.
            "mm_processor_kwargs": {"do_rescale": True},
        }
        url = self.backend.removesuffix("/v1") + "/tokenize"
        async with self.session.post(url, json=body) as response:
            payload = await response.json(content_type=None)
            if response.status != 200:
                raise RuntimeError(f"tokenize returned HTTP {response.status}: {payload}")
        return int(payload["count"])

    async def _force_final_answer(
        self, body, messages, *, reason, initial_prompt_tokens, usage,
        trace, model_outputs, limits,
    ) -> dict[str, Any]:
        """One constrained answer-only turn, using explicitly reserved budget."""
        final_messages = copy.deepcopy(messages)
        final_messages.append({"role": "user", "content": (
            "Tool-call limit reached. " if reason == "tool_call_limit" else
            "The tool-exploration budget is exhausted. "
        ) + "No more tool calls or reasoning. Use the evidence already gathered to "
            "answer the original question now. Output only <answer>...</answer>."})
        prompt_tokens = await self._token_count(final_messages, answer_only=True)
        growth = prompt_tokens - initial_prompt_tokens
        remaining = min(limits["max_response_tokens"] - growth,
                        limits["max_model_len"] - prompt_tokens)
        if remaining <= 0:
            raise RuntimeError("No context/token budget remains for the reserved final answer")
        final_body = copy.deepcopy(body)
        final_body.pop("tools", None)
        final_body.pop("tool_choice", None)
        final_body.pop("max_completion_tokens", None)
        final_body["messages"] = final_messages
        final_body["chat_template_kwargs"] = {"enable_thinking": False}
        # VTCBench expects a short text answer. Constrained decoding forbids
        # tool XML, unlike tool_choice=none alone (which only disables parsing).
        final_body["structured_outputs"] = {"regex": r"<answer>[^<]*</answer>"}
        final_body["max_tokens"] = min(limits["final_response_max_tokens"], remaining)
        status, completion = await self._backend_call(final_body)
        if status != 200:
            raise RuntimeError(f"final-answer backend returned HTTP {status}: {completion}")
        _usage_add(usage, completion)
        choice = completion["choices"][0]
        message = choice["message"]
        raw = _assistant_raw(message)
        generated = int((completion.get("usage") or {}).get("completion_tokens") or 0)
        model_outputs.append({
            "turn": len(model_outputs) + 1, "forced_final": True,
            "forced_final_reason": reason, "message": copy.deepcopy(message),
            "raw_generation": raw, "finish_reason": choice.get("finish_reason"),
            "max_tokens": final_body["max_tokens"],
            "usage": copy.deepcopy(completion.get("usage") or {}),
            "prompt_tokens": prompt_tokens, "prompt_growth_tokens": growth,
            "trajectory_tokens_after_generation": growth + generated,
        })
        success = choice.get("finish_reason") != "length" and bool(ANSWER_RE.fullmatch(raw.strip()))
        message["content"] = _final_answer(message)
        for key in ("reasoning_content", "reasoning", "tool_calls"):
            message.pop(key, None)
        completion.update({
            "usage": usage, "vtc_tool_trace": trace,
            "vtc_model_outputs": model_outputs, "vtc_raw_generation": raw,
            "vtc_initial_prompt_tokens": initial_prompt_tokens,
            "vtc_cumulative_usage": copy.deepcopy(usage),
            "vtc_final_completion_tokens": generated,
            "vtc_trajectory_tokens": growth + generated,
            "vtc_stop_reason": reason + ("_final_answer" if success else "_forced_generation"),
            "vtc_limits": limits,
        })
        choice["finish_reason"] = "stop" if success else "length"
        return completion

    async def completions(self, request: web.Request) -> web.Response:
        try:
            incoming = await request.json()
            if incoming.get("stream"):
                raise ValueError("streaming is not supported by the VTC tool gateway")
            original_messages = incoming.get("messages") or []
            page_sources = _image_sources(original_messages)
            messages = _prepare_messages(original_messages, self.config)
            result = await self._run_loop(incoming, messages, page_sources)
            return web.json_response(result)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return web.json_response({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": {"message": repr(exc), "type": "gateway_error"}}, status=500)

    async def _run_loop(
        self,
        incoming: dict[str, Any],
        messages: list[dict[str, Any]],
        page_sources: list[str],
    ) -> dict[str, Any]:
        body = {key: copy.deepcopy(value) for key, value in incoming.items() if key not in {"messages", "stream"}}
        body["model"] = self.config["served_model_name"]
        body["tools"] = self.tools
        # Preserve the raw Qwen XML emitted during training. The gateway owns
        # tool parsing, so vLLM must inject tools without parsing the output.
        body["tool_choice"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": True}
        body.update(self.config["sampling"])

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        trace: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []
        completion: dict[str, Any] | None = None
        max_turns = int(self.config["max_turns"])
        max_calls = int(self.config["max_tool_calls"])
        trajectory_budget = int(self.config["max_response_tokens"])
        force_final = bool(self.config.get("force_answer_on_tool_limit", False))
        final_tokens = int(self.config.get("final_response_max_tokens", 4096))
        final_reserve = final_tokens + 512 if force_final else 0
        limits = {
            "single_response_max_tokens": int(self.config["single_response_max_tokens"]),
            "max_response_tokens": trajectory_budget,
            "max_tool_calls": max_calls,
            "max_turns": max_turns,
            "max_model_len": int(self.config["backend"]["max_model_len"]),
            "force_answer_on_tool_limit": force_final,
            "final_response_max_tokens": final_tokens,
            "final_answer_reserve_tokens": final_reserve,
        }
        initial_prompt_tokens = await self._token_count(messages)
        current_prompt_tokens = initial_prompt_tokens
        trajectory_tokens = 0
        stop_reason = "max_turns"

        for turn in range(max_turns):
            prompt_growth = current_prompt_tokens - initial_prompt_tokens
            remaining = trajectory_budget - prompt_growth
            if force_final and (
                len(trace) >= max_calls or turn == max_turns - 1
                or remaining < final_reserve + int(self.config["single_response_max_tokens"])
            ):
                return await self._force_final_answer(
                    body, messages,
                    reason="tool_call_limit" if len(trace) >= max_calls else "trajectory_limit",
                    initial_prompt_tokens=initial_prompt_tokens, usage=usage,
                    trace=trace, model_outputs=model_outputs, limits=limits,
                )
            if remaining <= 0:
                stop_reason = "trajectory_limit_before_generation"
                break
            body["messages"] = messages
            body["max_tokens"] = min(int(self.config["single_response_max_tokens"]), remaining)
            status, completion = await self._backend_call(body)
            if status != 200:
                raise RuntimeError(f"backend returned HTTP {status}: {completion}")
            _usage_add(usage, completion)
            choice = completion["choices"][0]
            message = choice["message"]
            generated_tokens = int((completion.get("usage") or {}).get("completion_tokens") or 0)
            trajectory_tokens = prompt_growth + generated_tokens
            raw = _assistant_raw(message)
            turn_output = {
                "turn": turn + 1,
                # Preserve the backend payload before the gateway removes
                # reasoning/tool fields or extracts the scoreable answer.
                "message": copy.deepcopy(message),
                "raw_generation": raw,
                "finish_reason": choice.get("finish_reason"),
                "max_tokens": body["max_tokens"],
                "usage": copy.deepcopy(completion.get("usage") or {}),
                "prompt_tokens": current_prompt_tokens,
                "prompt_growth_tokens": prompt_growth,
                "trajectory_tokens_after_generation": trajectory_tokens,
            }
            model_outputs.append(turn_output)

            # GRPO marks a length-truncated action done before parsing or
            # executing a tool call.
            if choice.get("finish_reason") == "length":
                stop_reason = "backend_length"
                break
            if trajectory_tokens >= trajectory_budget:
                stop_reason = "trajectory_limit_after_generation"
                break

            if "<tool_call>" not in raw:
                message["content"] = _final_answer(message)
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
                message.pop("tool_calls", None)
                completion["usage"] = usage
                choice["finish_reason"] = "stop"
                completion["vtc_tool_trace"] = trace
                completion["vtc_model_outputs"] = model_outputs
                completion["vtc_raw_generation"] = raw
                completion["vtc_initial_prompt_tokens"] = initial_prompt_tokens
                completion["vtc_cumulative_usage"] = copy.deepcopy(usage)
                completion["vtc_final_completion_tokens"] = int(
                    (model_outputs[-1].get("usage") or {}).get("completion_tokens") or 0
                )
                completion["vtc_trajectory_tokens"] = trajectory_tokens
                completion["vtc_stop_reason"] = "final_answer"
                completion["vtc_limits"] = limits
                return completion

            messages.append({"role": "assistant", "content": raw})
            if len(trace) >= max_calls:
                stop_reason = "tool_call_limit"
                # The model has already seen the last tool observation. Give
                # it one explicit answer-only turn instead of returning the
                # unfinished tool XML as a zero-scoring terminal response.
                forced_messages = messages + [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "<tool_response>\n"
                                    "Tool-call limit reached. Do not call any more tools. "
                                    "Using the evidence already gathered, provide the "
                                    "best final answer now in the required <answer>...</answer> "
                                    "format.\n</tool_response>"
                                ),
                            }
                        ],
                    }
                ]
                forced_prompt_tokens = await self._token_count(forced_messages)
                forced_trajectory_tokens = forced_prompt_tokens - initial_prompt_tokens
                forced_remaining = trajectory_budget - forced_trajectory_tokens
                if forced_remaining > 0:
                    body["messages"] = forced_messages
                    body["max_tokens"] = min(
                        int(self.config["single_response_max_tokens"]),
                        forced_remaining,
                    )
                    forced_status, forced_completion = await self._backend_call(body)
                    if forced_status != 200:
                        raise RuntimeError(
                            f"backend returned HTTP {forced_status}: {forced_completion}"
                        )
                    _usage_add(usage, forced_completion)
                    forced_choice = forced_completion["choices"][0]
                    forced_message = forced_choice["message"]
                    forced_raw = _assistant_raw(forced_message)
                    forced_generated_tokens = int(
                        (forced_completion.get("usage") or {}).get("completion_tokens") or 0
                    )
                    trajectory_tokens = forced_trajectory_tokens + forced_generated_tokens
                    model_outputs.append(
                        {
                            "turn": turn + 2,
                            "forced_final": True,
                            "message": copy.deepcopy(forced_message),
                            "raw_generation": forced_raw,
                            "finish_reason": forced_choice.get("finish_reason"),
                            "max_tokens": body["max_tokens"],
                            "usage": copy.deepcopy(forced_completion.get("usage") or {}),
                            "prompt_tokens": forced_prompt_tokens,
                            "prompt_growth_tokens": forced_trajectory_tokens,
                            "trajectory_tokens_after_generation": trajectory_tokens,
                        }
                    )
                    completion = forced_completion
                    if (
                        forced_choice.get("finish_reason") != "length"
                        and "<tool_call>" not in forced_raw
                    ):
                        forced_message["content"] = _final_answer(forced_message)
                        forced_message.pop("reasoning_content", None)
                        forced_message.pop("reasoning", None)
                        forced_message.pop("tool_calls", None)
                        completion["usage"] = usage
                        forced_choice["finish_reason"] = "stop"
                        completion["vtc_tool_trace"] = trace
                        completion["vtc_model_outputs"] = model_outputs
                        completion["vtc_raw_generation"] = forced_raw
                        completion["vtc_initial_prompt_tokens"] = initial_prompt_tokens
                        completion["vtc_cumulative_usage"] = copy.deepcopy(usage)
                        completion["vtc_final_completion_tokens"] = forced_generated_tokens
                        completion["vtc_trajectory_tokens"] = trajectory_tokens
                        completion["vtc_stop_reason"] = "tool_call_limit_final_answer"
                        completion["vtc_limits"] = limits
                        return completion
                    stop_reason = "tool_call_limit_forced_generation"
                break
            observation, event = await self._execute(raw, page_sources)
            event["turn"] = turn + 1
            trace.append(event)
            # Keep the execution result beside the exact assistant output that
            # requested it. Image bytes are intentionally not duplicated here;
            # page/bbox/source/crop metadata identifies the returned crop.
            turn_output["tool_event"] = event
            candidate_messages = messages + [{"role": "user", "content": observation}]
            candidate_prompt_tokens = await self._token_count(candidate_messages)
            candidate_trajectory_tokens = candidate_prompt_tokens - initial_prompt_tokens
            if candidate_trajectory_tokens >= trajectory_budget - final_reserve:
                event["observation_appended"] = False
                event["candidate_trajectory_tokens"] = candidate_trajectory_tokens
                if force_final:
                    return await self._force_final_answer(
                        body, messages,
                        reason="tool_call_limit" if len(trace) >= max_calls else "trajectory_limit",
                        initial_prompt_tokens=initial_prompt_tokens, usage=usage,
                        trace=trace, model_outputs=model_outputs, limits=limits,
                    )
                stop_reason = "trajectory_limit_after_observation"
                break
            event["observation_appended"] = True
            messages = candidate_messages
            current_prompt_tokens = candidate_prompt_tokens
            trajectory_tokens = candidate_trajectory_tokens

        if completion is None:
            raise RuntimeError("tool loop ended before the first backend completion")
        message = completion["choices"][0]["message"]
        message["content"] = _final_answer(message)
        message.pop("reasoning_content", None)
        message.pop("reasoning", None)
        message.pop("tool_calls", None)
        completion["usage"] = usage
        completion["choices"][0]["finish_reason"] = "length"
        completion["vtc_tool_trace"] = trace
        completion["vtc_model_outputs"] = model_outputs
        completion["vtc_raw_generation"] = (
            model_outputs[-1]["raw_generation"] if model_outputs else ""
        )
        completion["vtc_initial_prompt_tokens"] = initial_prompt_tokens
        completion["vtc_cumulative_usage"] = copy.deepcopy(usage)
        completion["vtc_final_completion_tokens"] = int(
            (model_outputs[-1].get("usage") or {}).get("completion_tokens") or 0
        ) if model_outputs else 0
        completion["vtc_trajectory_tokens"] = trajectory_tokens
        completion["vtc_stop_reason"] = stop_reason
        completion["vtc_limits"] = limits
        return completion

    async def _execute(self, raw: str, page_sources: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert self.session is not None
        try:
            payload = self.extract_tool_call(raw)
            if payload is None:
                raise self.ZoomToolError("no complete tool call found")
            page, bbox = self.validate_payload(payload, len(page_sources))

            source = _prefer_high_res(
                page_sources[page - 1],
                self.config.get("prefer_high_res_dpi")
                if self.config.get("high_res_policy") == "paired_dpi144" else None,
                bool(self.config.get("require_high_res")),
            )
            image = await _load_image(source, self.session)
            pixels = self.normalized_bbox_to_pixels(
                bbox, image.width, image.height, float(self.config["crop_padding_ratio"])
            )
            crop = self.pad_crop(image.crop(pixels).copy())
            text = (
                f"Result from zoom_region on page {page} (crop {crop.width}x{crop.height}). "
                "Inspect the crop and answer the original question; retry zoom_region if it is still unreadable."
            )
            observation = [
                {"type": "text", "text": "<tool_response>\n"},
                {"type": "image_url", "image_url": {"url": _image_data_url(crop)}},
                {"type": "text", "text": f"\n{text}\n</tool_response>{self.config['turn_prompt']}"},
            ]
            return observation, {
                "status": "success",
                "page": page,
                "bbox_2d": bbox,
                "pixel_bbox": list(pixels),
                "crop_size": [crop.width, crop.height],
                "high_res_source": source != page_sources[page - 1],
                "source": _trace_source(source),
                "observation_text": text,
            }
        except (self.ZoomToolError, TypeError, ValueError, OSError) as exc:
            text = f"Zoom failed: {exc}. Try a different page or box."
            observation = [
                {
                    "type": "text",
                    "text": f"<tool_response>\n{text}\n</tool_response>{self.config['turn_prompt']}",
                }
            ]
            return observation, {
                "status": "failed",
                "error": str(exc),
                "observation_text": text,
            }


def create_app(config: dict[str, Any]) -> web.Application:
    gateway = Gateway(config)
    app = web.Application(client_max_size=512 * 1024**2)
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.stop)
    app.router.add_get("/health", gateway.health)
    app.router.add_get("/v1/models", gateway.models)
    app.router.add_post("/v1/chat/completions", gateway.completions)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        f"Starting VTC tool gateway for {config.get('benchmark')} on "
        f"{config['gateway_host']}:{config['gateway_port']} -> {config['backend_base_url']}",
        flush=True,
    )
    web.run_app(create_app(config), host=config["gateway_host"], port=int(config["gateway_port"]))


if __name__ == "__main__":
    main()
