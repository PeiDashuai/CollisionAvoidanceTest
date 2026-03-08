from __future__ import annotations

import argparse
from pathlib import Path

from colreg_benchmark.benchmark_builder import (
    build_benchmark_from_dataset,
    write_benchmark_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    samples = build_benchmark_from_dataset(args.dataset_root)
    write_benchmark_jsonl(samples, args.out)

    print(f"Built benchmark samples: {len(samples)}")
    print(f"Output: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()