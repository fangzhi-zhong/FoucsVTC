#!/usr/bin/env python3
"""Run one Qwen3-VL-8B tool-call loop without verl training.

This deliberately uses vLLM's public offline ``LLM.chat`` API. It validates
the stage-1 model/template/tool/observation loop while the old verl fork is
being ported to the vLLM version required by Qwen3-VL.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from PIL import Image


GRPO_ROOT = Path(__file__).resolve().parents[3]
ZOOM_MODULE_PATH = GRPO_ROOT / "verl/workers/agent/envs/mm_process_engine/vtc_zoom.py"
DEFAULT_MODEL = Path("/vepfs-mlp2/c20250405/400042/models/Qwen3-VL-8B-Instruct")
DEFAULT_IMAGE = Path(
    "/vepfs-mlp2/c20250405/400042/data/VTC_SFT/images/ChatQA-Training-Data/tatqa/"
    "ChatQA-Training-Data_tatqa_10955/ChatQA-Training-Data_tatqa_10955_page_001.png"
)
DEFAULT_QUESTION = "What are the components making up the total Compensation of Key Management Personnel in the table?"
DEFAULT_TRITON_CACHE = Path("/vepfs-mlp2/c20250405/400042/.tmp/triton/qwen3_vl_tool_smoke")

SYSTEM_PROMPT = """You are validating a document zoom tool pipeline.
You must call zoom_region exactly once before answering the question.
Choose the smallest region that contains the evidence needed for the answer.
The page number is 1-based and bbox_2d uses [x1, y1, x2, y2] coordinates normalized to 0-1000.
After the tool result is returned, answer the original question concisely and do not call another tool."""


def _load_zoom_module():
    spec = importlib.util.spec_from_file_location("vtc_zoom_stage1", ZOOM_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load zoom module from {ZOOM_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--first-max-tokens", type=int, default=512)
    parser.add_argument("--final-max-tokens", type=int, default=256)
    parser.add_argument("--use-cudagraph", action="store_true")
    parser.add_argument("--triton-cache-dir", type=Path, default=DEFAULT_TRITON_CACHE)
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Render the native Qwen3-VL tool conversation without loading model weights.",
    )
    return parser.parse_args()


def build_initial_messages(image, question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"image_pil": image},
                {"type": "text", "text": question},
            ],
        },
    ]


def render_template_only(model_path, image, question, zoom):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    simulated_action = (
        '<tool_call>\n{"name":"zoom_region","arguments":'
        '{"page":1,"bbox_2d":[20,690,940,770]}}\n</tool_call>'
    )
    messages = build_initial_messages(image, question)
    messages[1]["content"][0] = {"type": "image"}
    messages.extend(
        [
            {"role": "assistant", "content": simulated_action},
            {
                "role": "tool",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Result from zoom_region on page 1."},
                ],
            },
        ]
    )
    rendered = processor.apply_chat_template(
        messages, tools=[zoom.TOOL_SCHEMA], add_generation_prompt=True, tokenize=False
    )
    print(rendered)


def main():
    args = parse_args()
    zoom = _load_zoom_module()
    page = Image.open(args.image).convert("RGB")

    if args.template_only:
        render_template_only(args.model, page, args.question, zoom)
        return

    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR", args.triton_cache_dir))
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 2},
        enforce_eager=not args.use_cudagraph,
        trust_remote_code=False,
    )
    messages = build_initial_messages(page, args.question)
    first_params = SamplingParams(temperature=0.0, max_tokens=args.first_max_tokens, skip_special_tokens=False)
    first_output = llm.chat(
        messages,
        tools=[zoom.TOOL_SCHEMA],
        sampling_params=first_params,
        use_tqdm=False,
    )[0]
    action = first_output.outputs[0].text
    crop, tool_info = zoom.execute_zoom(action, [page])

    messages.extend(
        [
            {"role": "assistant", "content": action},
            {
                "role": "tool",
                "content": [
                    {"image_pil": crop},
                    {
                        "type": "text",
                        "text": "Result from zoom_region. Use this crop to answer the original question.",
                    },
                ],
            },
        ]
    )
    final_params = SamplingParams(temperature=0.0, max_tokens=args.final_max_tokens, skip_special_tokens=False)
    final_output = llm.chat(
        messages,
        tools=[zoom.TOOL_SCHEMA],
        sampling_params=final_params,
        use_tqdm=False,
    )[0]
    result = {
        "status": "success",
        "model": str(args.model),
        "image": str(args.image),
        "question": args.question,
        "first_action": action,
        "tool_info": tool_info,
        "final_answer": final_output.outputs[0].text,
        "first_prompt_tokens": len(first_output.prompt_token_ids),
        "first_output_tokens": len(first_output.outputs[0].token_ids),
        "final_prompt_tokens": len(final_output.prompt_token_ids),
        "final_output_tokens": len(final_output.outputs[0].token_ids),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

