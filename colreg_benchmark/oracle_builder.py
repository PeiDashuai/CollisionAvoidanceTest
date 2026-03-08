from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def _empty_sources(x: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Oracle prediction only needs label keys for current evaluator.
    Keep values as empty lists to mimic model output shape.
    """
    return {str(k): [] for k in x.keys()}


def make_oracle_predictions(samples: Iterable[BenchmarkSample]) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []

    for s in samples:
        pred = {
            "sample_id": s.sample_id,
            "triggered_rules": list(s.labels.triggered_rules),
            "maneuver_allowed": _empty_sources(s.labels.maneuver_allowed),
            "maneuver_forbidden": _empty_sources(s.labels.maneuver_forbidden),
            "lights_required": _empty_sources(s.labels.lights_required),
            "lights_forbidden": _empty_sources(s.labels.lights_forbidden),
            "sounds_required": _empty_sources(s.labels.sounds_required),
            "sounds_forbidden": _empty_sources(s.labels.sounds_forbidden),
        }
        preds.append(pred)

    return preds


def write_prediction_jsonl(preds: Iterable[Dict[str, Any]], out_path: str | Path) -> None:
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")