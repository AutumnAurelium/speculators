#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import time
from typing import List

from datasets import config as datasets_config
from datasets import load_dataset
import httpx


def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


def load_hf_prompts(
    dataset: str,
    config: str | None,
    split: str,
    field: str,
    limit: int | None,
    cache_dir: str | None,
) -> List[str]:
    cache_dir = cache_dir or datasets_config.HF_DATASETS_CACHE
    ds = load_dataset(dataset, config, split=split, cache_dir=cache_dir)
    if field not in ds.column_names:
        raise ValueError(
            f"Field '{field}' not found in dataset columns: {ds.column_names}"
        )
    if limit is not None:
        limit = min(limit, len(ds))
        ds = ds.select(range(limit))
    prompts = []
    for row in ds:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            prompts.append(text)
    if not prompts:
        raise ValueError(
            f"No prompts found in dataset {dataset} (split={split}, field={field})"
        )
    return prompts


def build_payload(model: str, prompt: str, max_tokens: int, temperature: float):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


async def worker(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
):
    async with semaphore:
        t0 = time.perf_counter()
        resp = await client.post(
            url,
            json=build_payload(model, prompt, max_tokens, temperature),
            timeout=None,
        )
        t1 = time.perf_counter()
        if resp.status_code >= 400:
            raise RuntimeError(
                f"HTTP {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "latency_s": t1 - t0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }


async def run(args):
    if args.prompts_file:
        prompts = load_prompts(args.prompts_file)
    elif args.prompt:
        prompts = [args.prompt]
    else:
        try:
            prompts = load_hf_prompts(
                dataset=args.dataset,
                config=args.dataset_config,
                split=args.dataset_split,
                field=args.dataset_field,
                limit=args.dataset_limit,
                cache_dir=args.hf_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load prompts from HuggingFace dataset. "
                "Use --prompt or --prompts-file to bypass dataset loading."
            ) from exc
    url = args.base_url.rstrip("/") + "/chat/completions"

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = []
        t_start = time.perf_counter()
        for i in range(args.requests):
            prompt = random.choice(prompts)
            tasks.append(
                worker(
                    client,
                    semaphore,
                    url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                )
            )
        results = await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    total_time = t_end - t_start
    latencies = [r["latency_s"] for r in results]
    prompt_tokens = [r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None]
    completion_tokens = [r["completion_tokens"] for r in results if r["completion_tokens"] is not None]

    def pct(values, p):
        if not values:
            return None
        values = sorted(values)
        k = int((len(values) - 1) * p)
        return values[k]

    print(json.dumps({
        "requests": args.requests,
        "concurrency": args.concurrency,
        "total_time_s": total_time,
        "rps": args.requests / total_time if total_time > 0 else None,
        "latency_p50_s": pct(latencies, 0.50),
        "latency_p90_s": pct(latencies, 0.90),
        "latency_p99_s": pct(latencies, 0.99),
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "prompt_tps": (sum(prompt_tokens) / total_time) if prompt_tokens else None,
        "completion_tps": (sum(completion_tokens) / total_time) if completion_tokens else None,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Benchmark vLLM OpenAI-compatible server with parallel requests")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="Base URL for vLLM OpenAI API")
    ap.add_argument("--model", required=True, help="Model name (as served by vLLM)")
    ap.add_argument("--requests", type=int, default=320, help="Total number of requests")
    ap.add_argument("--concurrency", type=int, default=32, help="Parallel in-flight requests")
    ap.add_argument(
        "--prompt",
        default=None,
        help="Optional single prompt (overrides dataset default if provided)",
    )
    ap.add_argument("--prompts-file", default=None, help="Optional file with one prompt per line")
    ap.add_argument(
        "--dataset",
        default="gsm8k",
        help="HuggingFace dataset name (default: gsm8k)",
    )
    ap.add_argument(
        "--dataset-config",
        default="main",
        help="HuggingFace dataset config name (default: main)",
    )
    ap.add_argument(
        "--dataset-split",
        default="test",
        help="HuggingFace dataset split (default: test)",
    )
    ap.add_argument(
        "--dataset-field",
        default="question",
        help="Field to use as prompt text (default: question)",
    )
    ap.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional limit on number of dataset prompts loaded",
    )
    ap.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Directory for HuggingFace datasets cache. "
            "If not specified, uses HF_DATASETS_CACHE env var or default location."
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--api-key", default=None, help="API key if required")
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
