from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

from .benchmark_schema import BenchmarkSample


def read_benchmark_jsonl(path: str | Path) -> List[BenchmarkSample]:
    p = Path(path).resolve()
    rows: List[BenchmarkSample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(BenchmarkSample.model_validate_json(line))
    return rows


def write_benchmark_jsonl(samples: Iterable[BenchmarkSample], out_path: str | Path) -> None:
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")


def _group_key(sample: BenchmarkSample) -> Tuple[str, str]:
    return (sample.pattern, sample.difficulty_level)


def stratified_split(
    samples: List[BenchmarkSample],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[BenchmarkSample]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = random.Random(seed)

    groups: Dict[Tuple[str, str], List[BenchmarkSample]] = defaultdict(list)
    for s in samples:
        groups[_group_key(s)].append(s)

    train: List[BenchmarkSample] = []
    val: List[BenchmarkSample] = []
    test: List[BenchmarkSample] = []

    for _, bucket in groups.items():
        bucket = bucket[:]
        rng.shuffle(bucket)

        n = len(bucket)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = int(round(n * val_ratio))
            n_test = n - n_train - n_val

            if n_test < 1:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                elif n_val > 0:
                    n_val -= 1

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    # write split label into sample objects
    out_train = [s.model_copy(update={"split": "train"}) for s in train]
    out_val = [s.model_copy(update={"split": "val"}) for s in val]
    out_test = [s.model_copy(update={"split": "test"}) for s in test]

    return {"train": out_train, "val": out_val, "test": out_test}


def summarize_splits(split_dict: Dict[str, List[BenchmarkSample]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}

    for split_name, samples in split_dict.items():
        summary: Dict[str, int] = {
            "num_samples": len(samples),
        }

        pattern_counts: Dict[str, int] = defaultdict(int)
        difficulty_counts: Dict[str, int] = defaultdict(int)

        for s in samples:
            pattern_counts[s.pattern] += 1
            difficulty_counts[s.difficulty_level] += 1

        for k, v in sorted(pattern_counts.items()):
            summary[f"pattern::{k}"] = v
        for k, v in sorted(difficulty_counts.items()):
            summary[f"difficulty::{k}"] = v

        out[split_name] = summary

    return out