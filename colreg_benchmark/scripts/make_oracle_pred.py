from __future__ import annotations

import argparse
from pathlib import Path

from colreg_benchmark.oracle_builder import (
    read_benchmark_jsonl,
    make_oracle_predictions,
    write_prediction_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True, help="GT benchmark jsonl")
    ap.add_argument("--out", type=str, required=True, help="Oracle prediction jsonl")
    args = ap.parse_args()

    samples = read_benchmark_jsonl(args.infile)
    preds = make_oracle_predictions(samples)
    write_prediction_jsonl(preds, args.out)

    print(f"Oracle predictions written: {len(preds)}")
    print(f"Output: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()