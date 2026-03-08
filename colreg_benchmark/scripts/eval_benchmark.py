from __future__ import annotations

import argparse
import json
from pathlib import Path

from colreg_benchmark.benchmark_eval import evaluate_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True)
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    result = evaluate_benchmark(args.gt, args.pred)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()