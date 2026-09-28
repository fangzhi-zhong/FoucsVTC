"""Measure character error rate through an OpenAI-compatible vision endpoint."""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import Levenshtein
import httpx
from openai import AsyncOpenAI

PROMPT = ("Transcribe all the text in this image exactly as it appears, character for "
          "character. Preserve spelling, capitalization, digits and punctuation exactly, "
          "including any strings that look like nonsense words. Do not correct, translate, "
          "summarize or add anything. Output only the transcribed text.")
_WS = re.compile(r"\s+")
# Glyph 和 VTC SFT 都是无条件先想再答，转写内容只在 </think> 之后。留一个未闭合的
# <think> 也要丢掉：那说明预算全被推理吃完了，后面没有转写。
_THINK = re.compile(r"<think>.*?</think>|<think>.*$", re.S | re.I)


def norm(s):
    return _WS.sub(" ", s or "").strip()


def cer(pred, gold):
    g = norm(gold)
    return Levenshtein.distance(norm(pred), g) / max(len(g), 1)


def final_text(content, reasoning="", strip_think=False):
    """Return only the transcription, handling native and merged Qwen3.5 outputs."""
    text = content or ""
    if strip_think:
        text = _THINK.sub("", text)
    # vLLM's reasoning parser puts hidden reasoning in ``reasoning_content``;
    # never concatenate it with the answer when the parser already separated it.
    return text.strip()


async def one(client, sem, row, model, strip_think, enable_thinking, retries=3):
    async with sem:
        budget = int(len(row["gold"]) / 2.5) + 256 + (3072 if strip_think else 0)
        for k in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=budget,
                    extra_body={"chat_template_kwargs": {
                        "enable_thinking": bool(enable_thinking)}},
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": Path(row["image"]).resolve().as_uri()}},
                        {"type": "text", "text": PROMPT}]}])
                msg = r.choices[0].message
                pred = final_text(getattr(msg, "content", ""),
                                  getattr(msg, "reasoning_content", ""), strip_think)
                return {**{k2: row[k2] for k2 in ("cond", "font", "pt", "idx")},
                        "pred": pred, "cer": cer(pred, row["gold"]),
                        "trunc": r.choices[0].finish_reason == "length",
                        "finish_reason": r.choices[0].finish_reason,
                        "model": model, "ok": True}
            except Exception as e:
                if k == retries - 1:
                    return {**{k2: row[k2] for k2 in ("cond", "font", "pt", "idx")},
                            "pred": "", "cer": None, "ok": False,
                            "error": str(e)[:500], "model": model}
                await asyncio.sleep(2 * (k + 1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("FOCUSVTC_ACUITY_ROOT", "outputs/font_acuity"))
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--out", default="outputs/font_acuity/transcribe.jsonl")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen3.5-9B"))
    ap.add_argument("--fonts", nargs="*", default=None, help="只跑这些字体，默认全跑")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8021/v1"))
    ap.add_argument("--strip-think", action="store_true",
                    help="兼容未启用 reasoning parser 的合并输出；启用 parser 时无需使用")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="启用 Qwen3.5 原生 thinking；默认关闭以直接测视觉转写")
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()

    if a.concurrency < 1:
        ap.error("--concurrency must be positive")
    with (Path(a.root) / a.manifest).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if a.fonts:
        rows = [r for r in rows if r["font"] in a.fonts]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("ok", True) and r.get("model") == a.model:
                done.add((r["cond"], r["font"], r["pt"], r["idx"]))
    todo = [r for r in rows if (r["cond"], r["font"], r["pt"], r["idx"]) not in done]
    print(f"{len(rows)} 张，已完成 {len(done)}，待跑 {len(todo)}", file=sys.stderr)

    base_url = a.base_url.rstrip("/")
    if base_url == "http://127.0.0.1:8021/v1" and a.port != 8021:
        base_url = f"http://127.0.0.1:{a.port}/v1"
    # Local vLLM must bypass cluster HTTP proxies; otherwise localhost requests
    # can be answered by the proxy with a misleading 502.
    http_client = httpx.AsyncClient(timeout=600, trust_env=False)
    client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get("FOCUSVTC_API_KEY") or "EMPTY", timeout=600,
                         http_client=http_client)
    sem = asyncio.Semaphore(a.concurrency)
    n = 0
    with open(out, "a") as f:
        for fut in asyncio.as_completed(
                [one(client, sem, r, a.model, a.strip_think, a.enable_thinking) for r in todo]):
            r = await fut
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr)
    print("done", file=sys.stderr)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
