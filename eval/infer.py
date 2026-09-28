#!/usr/bin/env python3
"""Ask one question over document pages through an OpenAI-compatible server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True, help="ordered page images; preserve dpi_72/dpi_144 paths for zoom")
    parser.add_argument("--question", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18350/v1")
    parser.add_argument("--model", default="FocusVTC")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="optional complete API response including tool trajectory")
    args = parser.parse_args()
    content = []
    for number, path in enumerate(args.images, 1):
        path = path.expanduser().resolve(strict=True)
        content.extend([
            {"type": "text", "text": f"Page {number}"},
            {"type": "image_url", "image_url": {"url": path.as_uri()}},
            {"type": "text", "text": "\n"},
        ])
    content.append({"type": "text", "text": args.question})
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=3600)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(response.choices[0].message.content or "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(response.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
