import argparse
import random
from pathlib import Path

import torch


def _load_sample(path: Path):
    data = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    if "hidden_states" in data and isinstance(data["hidden_states"], list):
        hidden_states = torch.cat(data["hidden_states"][:-1], dim=-1)
        verifier_last_hidden_states = data["hidden_states"][-1]
    elif "hidden_state" in data:
        hidden_states = data["hidden_state"]
        verifier_last_hidden_states = data.get("target")
    else:
        raise KeyError(
            f"{path.name}: expected hidden_states list or hidden_state tensor"
        )

    return {
        "hidden_states": hidden_states,
        "verifier_last_hidden_states": verifier_last_hidden_states,
        "input_ids": data["input_ids"],
        "loss_mask": data["loss_mask"],
    }


def _tensor_stats(t: torch.Tensor):
    finite = torch.isfinite(t)
    nonfinite = (~finite).sum().item()
    if nonfinite == t.numel():
        max_abs = float("inf")
    else:
        max_abs = t.abs().max().item()
    return nonfinite, max_abs, str(t.dtype)


def _check_loss_mask(loss_mask: torch.Tensor):
    unique_vals = torch.unique(loss_mask)
    allowed = {0, 1}
    bad_vals = [int(v) for v in unique_vals.tolist() if int(v) not in allowed]
    return bad_vals


def validate_file(path: Path, max_abs_warn: float | None):
    sample = _load_sample(path)
    issues = []

    input_ids = sample["input_ids"]
    loss_mask = sample["loss_mask"]
    hs = sample["hidden_states"]
    vhs = sample["verifier_last_hidden_states"]

    seq_len = input_ids.shape[0]
    if loss_mask.shape[0] != seq_len:
        issues.append(
            f"loss_mask length {loss_mask.shape[0]} != input_ids length {seq_len}"
        )
    if hs.shape[0] != seq_len:
        issues.append(f"hidden_states length {hs.shape[0]} != input_ids length {seq_len}")
    if vhs is not None and vhs.shape[0] != seq_len:
        issues.append(
            f"verifier_last_hidden_states length {vhs.shape[0]} != input_ids length {seq_len}"
        )

    bad_mask_vals = _check_loss_mask(loss_mask)
    if bad_mask_vals:
        issues.append(f"loss_mask has values outside {{0,1}}: {bad_mask_vals}")

    hs_nf, hs_max, hs_dtype = _tensor_stats(hs)
    vhs_nf, vhs_max, vhs_dtype = (0, 0.0, "None")
    if vhs is not None:
        vhs_nf, vhs_max, vhs_dtype = _tensor_stats(vhs)

    if hs_nf > 0:
        issues.append(f"hidden_states has {hs_nf} non-finite values")
    if vhs is not None and vhs_nf > 0:
        issues.append(f"verifier_last_hidden_states has {vhs_nf} non-finite values")

    if max_abs_warn is not None:
        if hs_max > max_abs_warn:
            issues.append(f"hidden_states max abs {hs_max:.4g} > {max_abs_warn:.4g}")
        if vhs is not None and vhs_max > max_abs_warn:
            issues.append(
                f"verifier_last_hidden_states max abs {vhs_max:.4g} > {max_abs_warn:.4g}"
            )

    stats = {
        "seq_len": seq_len,
        "loss_mask_sum": int(loss_mask.sum().item()),
        "hidden_states": {"dtype": hs_dtype, "max_abs": hs_max, "nonfinite": hs_nf},
        "verifier_last_hidden_states": {
            "dtype": vhs_dtype,
            "max_abs": vhs_max,
            "nonfinite": vhs_nf,
        },
    }
    return issues, stats


def main():
    parser = argparse.ArgumentParser(description="Validate speculators training data")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-abs-warn", type=float, default=1e3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--delete-bad",
        action="store_true",
        help="Delete files that fail validation checks",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    files = list(data_path.rglob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found under {data_path}")

    random.seed(args.seed)
    random.shuffle(files)
    if args.max_files > 0:
        files = files[: args.max_files]

    total = 0
    files_with_issues = 0
    deleted_files = 0
    worst = {"path": None, "max_abs": 0.0}

    for path in sorted(files):
        total += 1
        issues, stats = validate_file(path, args.max_abs_warn)
        hs_max = stats["hidden_states"]["max_abs"]
        vhs_max = stats["verifier_last_hidden_states"]["max_abs"]
        max_abs = max(hs_max, vhs_max)
        if max_abs > worst["max_abs"]:
            worst = {"path": path, "max_abs": max_abs}

        if issues or args.verbose:
            print(f"\n{path}")
            print(f"  seq_len={stats['seq_len']} loss_mask_sum={stats['loss_mask_sum']}")
            print(
                "  hidden_states:",
                stats["hidden_states"]["dtype"],
                "max_abs=",
                stats["hidden_states"]["max_abs"],
                "nonfinite=",
                stats["hidden_states"]["nonfinite"],
            )
            if stats["verifier_last_hidden_states"]["dtype"] != "None":
                print(
                    "  verifier_last_hidden_states:",
                    stats["verifier_last_hidden_states"]["dtype"],
                    "max_abs=",
                    stats["verifier_last_hidden_states"]["max_abs"],
                    "nonfinite=",
                    stats["verifier_last_hidden_states"]["nonfinite"],
                )
            for issue in issues:
                print("  ISSUE:", issue)

        if issues:
            files_with_issues += 1
            if args.delete_bad:
                try:
                    path.unlink()
                    deleted_files += 1
                except OSError as exc:
                    print(f"  DELETE FAILED: {exc}")

    print("\nSummary")
    print(f"  scanned files: {total}")
    print(f"  files with issues: {files_with_issues}")
    if args.delete_bad:
        print(f"  deleted files: {deleted_files}")
    if worst["path"] is not None:
        print(f"  max abs seen: {worst['max_abs']:.4g} in {worst['path']}")


if __name__ == "__main__":
    main()
