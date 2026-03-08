from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple, Set

from .benchmark_schema import BenchmarkSample


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path).resolve()
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_map_from_gt(path: str | Path) -> Dict[str, BenchmarkSample]:
    rows = _read_jsonl(path)
    return {str(r["sample_id"]): BenchmarkSample.model_validate(r) for r in rows}


def _sample_map_from_pred(path: str | Path) -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(path)
    return {str(r["sample_id"]): r for r in rows}


def _set_from_rule_list(x: Any) -> Set[str]:
    if x is None:
        return set()
    if isinstance(x, list):
        return {str(t) for t in x}
    return set()


def _set_from_source_map(x: Any) -> Set[str]:
    """
    Flatten {"TURN_PORT": [...], "REDUCE_SPEED": [...]} into {"TURN_PORT","REDUCE_SPEED"}.
    """
    if not isinstance(x, dict):
        return set()
    return {str(k) for k in x.keys()}


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _compare_sets(gt: Set[str], pred: Set[str]) -> Tuple[int, int, int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return tp, fp, fn


def evaluate_rule_retrieval(gt_samples: Dict[str, BenchmarkSample], pred_rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    tp = fp = fn = 0

    for sid, gt in gt_samples.items():
        pred = pred_rows.get(sid, {})
        gt_set = set(gt.labels.triggered_rules)
        pred_set = _set_from_rule_list(pred.get("triggered_rules", []))
        a, b, c = _compare_sets(gt_set, pred_set)
        tp += a
        fp += b
        fn += c

    return _prf1(tp, fp, fn)


def evaluate_maneuver_labels(gt_samples: Dict[str, BenchmarkSample], pred_rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    tp = fp = fn = 0

    for sid, gt in gt_samples.items():
        pred = pred_rows.get(sid, {})

        gt_allowed = set(gt.labels.maneuver_allowed.keys())
        gt_forbidden = set(gt.labels.maneuver_forbidden.keys())
        gt_set = {f"A::{x}" for x in gt_allowed} | {f"F::{x}" for x in gt_forbidden}

        pred_allowed = _set_from_source_map(pred.get("maneuver_allowed", {}))
        pred_forbidden = _set_from_source_map(pred.get("maneuver_forbidden", {}))
        pred_set = {f"A::{x}" for x in pred_allowed} | {f"F::{x}" for x in pred_forbidden}

        a, b, c = _compare_sets(gt_set, pred_set)
        tp += a
        fp += b
        fn += c

    return _prf1(tp, fp, fn)


def evaluate_signal_labels(gt_samples: Dict[str, BenchmarkSample], pred_rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    tp = fp = fn = 0

    for sid, gt in gt_samples.items():
        pred = pred_rows.get(sid, {})

        gt_set = set()
        gt_set |= {f"LR::{x}" for x in gt.labels.lights_required.keys()}
        gt_set |= {f"LF::{x}" for x in gt.labels.lights_forbidden.keys()}
        gt_set |= {f"SR::{x}" for x in gt.labels.sounds_required.keys()}
        gt_set |= {f"SF::{x}" for x in gt.labels.sounds_forbidden.keys()}

        pred_set = set()
        pred_set |= {f"LR::{x}" for x in _set_from_source_map(pred.get("lights_required", {}))}
        pred_set |= {f"LF::{x}" for x in _set_from_source_map(pred.get("lights_forbidden", {}))}
        pred_set |= {f"SR::{x}" for x in _set_from_source_map(pred.get("sounds_required", {}))}
        pred_set |= {f"SF::{x}" for x in _set_from_source_map(pred.get("sounds_forbidden", {}))}

        a, b, c = _compare_sets(gt_set, pred_set)
        tp += a
        fp += b
        fn += c

    return _prf1(tp, fp, fn)


def evaluate_arbitration(gt_samples: Dict[str, BenchmarkSample], pred_rows: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Exact-match style metric on final maneuver arbitration result.
    """
    total = 0
    exact = 0

    for sid, gt in gt_samples.items():
        pred = pred_rows.get(sid, {})

        gt_allowed = set(gt.labels.maneuver_allowed.keys())
        gt_forbidden = set(gt.labels.maneuver_forbidden.keys())

        pred_allowed = _set_from_source_map(pred.get("maneuver_allowed", {}))
        pred_forbidden = _set_from_source_map(pred.get("maneuver_forbidden", {}))

        total += 1
        if gt_allowed == pred_allowed and gt_forbidden == pred_forbidden:
            exact += 1

    return {"exact_match": exact / total if total > 0 else 0.0}


def evaluate_benchmark(
    gt_path: str | Path,
    pred_path: str | Path,
) -> Dict[str, Any]:
    gt_samples = _sample_map_from_gt(gt_path)
    pred_rows = _sample_map_from_pred(pred_path)

    rule_metrics = evaluate_rule_retrieval(gt_samples, pred_rows)
    maneuver_metrics = evaluate_maneuver_labels(gt_samples, pred_rows)
    signal_metrics = evaluate_signal_labels(gt_samples, pred_rows)
    arbitration_metrics = evaluate_arbitration(gt_samples, pred_rows)

    aggregate = (
        0.25 * rule_metrics["f1"]
        + 0.35 * maneuver_metrics["f1"]
        + 0.20 * signal_metrics["f1"]
        + 0.20 * arbitration_metrics["exact_match"]
    )

    return {
        "num_gt_samples": len(gt_samples),
        "num_pred_samples": len(pred_rows),
        "rule_retrieval": rule_metrics,
        "maneuver": maneuver_metrics,
        "signals": signal_metrics,
        "arbitration": arbitration_metrics,
        "aggregate_score": aggregate,
    }