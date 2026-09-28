#!/usr/bin/env python3
"""Run one benchmark against a running zoom gateway, then score its outputs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from .config import load_config
except ImportError:
    from config import load_config

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
BENCHMARKS = ("ruler_v1", "ruler_v2", "longbench", "mrcr", "vtcbench", "mmmu", "ocrbench")


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--config", type=Path, help="override the matching JSON configuration")
    parser.add_argument("--limit", type=int, default=0, help="samples per task/subset; 0 means all")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--client-python", default=sys.executable)
    parser.add_argument("--lmms-python", default=sys.executable, help="Python with external lmms-eval or VTCBench installed")
    args = parser.parse_args()
    if args.limit < 0 or args.concurrency < 1:
        parser.error("--limit must be nonnegative and --concurrency must be positive")
    config = load_config(args.config or HERE / "configs" / f"{args.benchmark}.json")
    if config.get("benchmark") != args.benchmark:
        parser.error("--config must select the requested benchmark")
    host = config["gateway_host"]
    base_url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{config['gateway_port']}/v1"
    result = Path(config["results_dir"])
    result.mkdir(parents=True, exist_ok=True)
    model = config["served_model_name"]
    common = ["--base-url", base_url, "--model-name", model, "--results-dir", str(result), "--concurrency", str(args.concurrency), "--limit", str(args.limit)]
    turn_tokens = str(config["single_response_max_tokens"])

    if args.benchmark in ("ruler_v1", "ruler_v2", "longbench"):
        root = EVAL / ("LongBench" if args.benchmark == "longbench" else f"RULER/{args.benchmark}_dpi")
        command = [args.client_python, str(root / "run_eval.py"), *common, "--runs", *config["runs"], "--vtc-root", config["data_root"], "--page-markers", "--training-newline", "--system-prompt", config["system_prompt"]]
        if args.benchmark != "longbench":
            command += ["--max-tokens", turn_tokens]
        if args.benchmark == "ruler_v1":
            command += ["--answer-prefix-mode", "none"]
        run(command)
        run([args.client_python, str(root / "score.py"), "--results-dir", str(result), "--label", model])
    elif args.benchmark == "mrcr":
        root = EVAL / "MRCR"
        # Reserve room for generated turns and returned crop images.
        context_limit = int(config["backend"]["max_model_len"]) - int(config["max_response_tokens"]) + int(config["single_response_max_tokens"])
        command = [args.client_python, str(root / "run_eval.py"), *common, "--data-root", config["data_root"], "--max-tokens", turn_tokens, "--context-limit", str(context_limit), "--max-images", str(config["backend"]["max_images"]), "--vision-profile", "qwen", "--training-newline", "--system-prompt", config["system_prompt"]]
        run(command)
        run([args.client_python, str(root / "score.py"), "--results-dir", str(result), "--label", model])
    elif args.benchmark == "vtcbench":
        bench = Path(config["vtcbench_root"])
        if not (bench / "examples/run_wild.py").is_file():
            raise SystemExit("Set VTCBENCH_ROOT to an external VTCBench checkout with examples/run_wild.py; see docs/evaluation.md")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(bench / "src") + os.pathsep + env.get("PYTHONPATH", "")
        model_config = result / "vtcbench_model.json"
        model_config.write_text(json.dumps({
            "model": model, "api_key": "EMPTY", "api_url": base_url,
            "api_provider": "vllm", "system_prompt": config["system_prompt"],
            "max_tokens": config["single_response_max_tokens"], "temperature": 0.0,
            "top_p": 1.0, "timeout": 3600, "max_retries": 2,
            "tokenizer_type": "huggingface", "tokenizer_model": config["model"],
        }, indent=2) + "\n")
        for tier in config["tiers"]:
            tier_result = result / tier
            tier_result.mkdir(parents=True, exist_ok=True)
            for split in ("Retrieval", "Reasoning", "Memory"):
                command = [args.lmms_python, str(bench / "examples/run_wild.py"), "--model", str(model_config), "--data.path", str(Path(config["data_root"]) / tier), "--data.split", split, "--run.result_dir", str(tier_result), "--run.num_workers", str(args.concurrency), "--run.verbose", "false", "--run.api_cache_dir", str(tier_result / "api_cache"), "--run.image_dir", str(tier_result / "images")]
                if args.limit:
                    command += ["--run.num_tasks", str(args.limit)]
                run(command, cwd=bench, env=env)
            run([args.lmms_python, str(bench / "examples/collect.py"), str(tier_result)], cwd=tier_result, env=env)
    else:
        # Upstream task implementations keep benchmark downloads and licenses external.
        model_args = f"model={model},base_url={base_url},api_key=EMPTY,timeout=3600,max_size_in_mb=100,num_concurrent={args.concurrency}"
        command = [args.lmms_python, "-m", "lmms_eval", "--model", "openai", "--model_args", model_args, "--tasks", config["task"], "--batch_size", "1", "--gen_kwargs", f"max_new_tokens={turn_tokens},temperature=0.0,until=[]", "--log_samples", "--output_path", str(result / "output")]
        if args.benchmark == "ocrbench":
            command += ["--include_path", str(EVAL / "OCRBench/tasks")]
        if args.limit:
            command += ["--limit", str(args.limit)]
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "EMPTY"
        run(command, env=env)


if __name__ == "__main__":
    main()
