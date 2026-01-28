#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List

from datasets import load_dataset


def parse_messages(val: Any) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        return json.loads(val)
    raise ValueError(f"Unsupported message field type: {type(val)}")


def is_tools_nonempty(tools: Any) -> bool:
    if tools is None:
        return False
    if isinstance(tools, str):
        tools = tools.strip()
        if not tools:
            return False
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return True
    if isinstance(tools, list):
        return len(tools) > 0
    if isinstance(tools, dict):
        return len(tools) > 0
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert HF dataset with prompt/completion/tools to speculators conversations JSONL."
    )
    ap.add_argument("--dataset", required=True, help="HF dataset path or local path")
    ap.add_argument("--config", default=None, help="HF dataset config name (for multi-config datasets)")
    ap.add_argument("--split", default="train", help="HF split (default: train)")
    ap.add_argument("--output", required=True, help="Output JSONL file")
    ap.add_argument(
        "--allow-tools",
        action="store_true",
        help="Keep rows with non-empty tools; otherwise filter them out",
    )
    args = ap.parse_args()

    ds = load_dataset(args.dataset, name=args.config, split=args.split)

    kept = 0
    skipped = 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        for row in ds:
            tools = row.get("tools")
            if not args.allow_tools and is_tools_nonempty(tools):
                skipped += 1
                continue

            prompt = parse_messages(row.get("prompt", []))
            completion = parse_messages(row.get("completion", []))
            conversations = prompt + completion

            out = {"conversations": conversations}
            if args.allow_tools and tools is not None:
                out["tools"] = tools

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()
