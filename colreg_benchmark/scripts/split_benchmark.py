from __future__ import annotations

import argparse
import json
from pathlib import Path

from colreg_benchmark.benchmark_split import (
    read_benchmark_jsonl,
    write_benchmark_jsonl,
    stratified_split,
    summarize_splits,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    args = ap.parse_args()

    samples = read_benchmark_jsonl(args.infile)
    splits = stratified_split(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    write_benchmark_jsonl(splits["train"], outdir / "train.jsonl")
    write_benchmark_jsonl(splits["val"], outdir / "val.jsonl")
    write_benchmark_jsonl(splits["test"], outdir / "test.jsonl")

    summary = summarize_splits(splits)
    with (outdir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"summary: {outdir / 'split_summary.json'}")


if __name__ == "__main__":
    main()