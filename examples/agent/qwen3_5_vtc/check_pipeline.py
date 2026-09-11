#!/usr/bin/env python3
"""Offline stage-0 check for the Qwen3.5-VL VTC GRPO pipeline.

This loads one Parquet row and the processor, but never loads model weights or
starts Ray/vLLM.  It catches the three common wiring errors before spending a
GPU allocation: tool-schema rendering, high-DPI crop execution, and
sequence-aligned Qwen3.5 multimodal inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.agent.envs.mm_process_engine.visual_toolbox_qwen3_vtc import Qwen3VLVTCZoomTool
    from verl.workers.agent.parallel_env import _preprocess_multi_modal_inputs

    schema = args.schema or args.root / "examples/agent/qwen3_vl_vtc_tool/zoom_region_tools.json"
    tokenizer = hf_tokenizer(str(args.model))
    processor = hf_processor(str(args.model))
    if processor is None:
        raise RuntimeError("AutoProcessor could not be loaded")

    config = OmegaConf.create(
        {
            "cache_dir": "/tmp/vtc-grpo-cache",
            "prompt_key": "prompt",
            "image_key": "images",
            "high_res_image_key": "high_res_images",
            "video_key": "videos",
            "tools_key": "tools",
            "tools_schema_path": str(schema),
            "tools_enabled_key": "enable_tools",
            "max_prompt_length": 16384,
            "return_raw_chat": True,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "filter_overlong_prompts_workers": 1,
        }
    )
    dataset = RLHFDataset(str(args.parquet), tokenizer, config, processor)
    item = dataset[args.index]
    mm = item["multi_modal_inputs"]
    if mm["mm_token_type_ids"].shape[-1] != item["input_ids"].shape[-1]:
        raise AssertionError("mm_token_type_ids is not sequence-aligned")
    print(
        f"row={item['extra_info']['id']} prompt={tuple(item['input_ids'].shape)} "
        f"mrope={tuple(item['position_ids'].shape)} images={len(item['images']) if 'images' in item else len(item['origin_multi_modal_data']['image'])} "
        f"grids={tuple(mm['image_grid_thw'].shape)}"
    )

    evidence = item["extra_info"].get("evidence_locations", [])
    if not evidence:
        raise RuntimeError("selected row has no evidence location")
    target = evidence[0]
    page = int(target["page"])
    bbox = target["bbox"]
    action = (
        "<tool_call><function=zoom_region>"
        f"<parameter=page>{page}</parameter>"
        f"<parameter=bbox_2d>{bbox}</parameter>"
        "</function></tool_call>"
    )
    tool = Qwen3VLVTCZoomTool()
    tool.reset(
        multi_modal_data=item["multi_modal_data"],
        origin_multi_modal_data=item["origin_multi_modal_data"],
        extra_info=item["extra_info"],
    )
    observation, reward, done, info = tool.execute(action)
    if done or "multi_modal_data" not in observation:
        raise AssertionError(f"successful crop unexpectedly terminated: {info}")
    _, obs_ids, obs_mm = _preprocess_multi_modal_inputs(observation["prompt"], processor, **observation)
    print(
        f"tool page={info['page']} iou={info.get('best_iou', 0):.3f} shaping_reward={reward:.3f} "
        f"obs_tokens={obs_ids.numel()} obs_mm={tuple(obs_mm['mm_token_type_ids'].shape)}"
    )
    print("offline pipeline check: OK (no model weights loaded)")


if __name__ == "__main__":
    main()
