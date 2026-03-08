from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .benchmark_schema import (
    BenchmarkSample,
    BenchmarkInputs,
    BenchmarkLabels,
    BenchmarkQA,
    infer_difficulty_level,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_source_map(x: Any) -> Dict[str, List[str]]:
    """
    Normalize labels like:
      {"TURN_PORT": ["R1","R2"]}
      {"TURN_PORT": ("R1","R2")}
      {"TURN_PORT": "R1"}
    to Dict[str, List[str]]
    """
    if not isinstance(x, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for k, v in x.items():
        if v is None:
            out[str(k)] = []
        elif isinstance(v, list):
            out[str(k)] = [str(t) for t in v]
        elif isinstance(v, tuple):
            out[str(k)] = [str(t) for t in v]
        else:
            out[str(k)] = [str(v)]
    return out


def _normalize_suppressed_rules(x: Any) -> List[Dict[str, Any]]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[Dict[str, Any]] = []
        for item in x:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"repr": str(item)})
        return out
    return [{"repr": str(x)}]


def _extract_triggered_rules(label_obj: Dict[str, Any]) -> List[str]:
    gr = label_obj.get("global_resolution", {}) or {}
    kept = gr.get("kept_rules", []) or []
    return [str(x) for x in kept]


def _extract_labels(label_obj: Dict[str, Any]) -> BenchmarkLabels:
    gr = label_obj.get("global_resolution", {}) or {}
    glab = gr.get("labels", {}) or {}

    return BenchmarkLabels(
        triggered_rules=_extract_triggered_rules(label_obj),
        maneuver_allowed=_normalize_source_map(glab.get("maneuver_allowed")),
        maneuver_forbidden=_normalize_source_map(glab.get("maneuver_forbidden")),
        lights_required=_normalize_source_map(glab.get("lights_required")),
        lights_forbidden=_normalize_source_map(glab.get("lights_forbidden")),
        sounds_required=_normalize_source_map(glab.get("sounds_required")),
        sounds_forbidden=_normalize_source_map(glab.get("sounds_forbidden")),
        suppressed_rules=_normalize_suppressed_rules(gr.get("suppressed_rules")),
        explanation_steps=gr.get("explanation", {}).get("steps", []) or [],
    )


def _extract_qa(label_obj: Dict[str, Any]) -> BenchmarkQA:
    gr = label_obj.get("global_resolution", {}) or {}
    meta = label_obj.get("meta", {}) or {}

    return BenchmarkQA(
        sanity_issues=label_obj.get("sanity_issues", []) or [],
        signals_issues=label_obj.get("signals_issues", []) or [],
        sampled_metrics=meta.get("sampled_metrics", {}) or {},
        global_metrics=gr.get("metrics", {}) or {},
    )


def _build_one_sample(
    manifest_row: Dict[str, Any],
    dataset_root: Path,
) -> BenchmarkSample:
    sample_id = str(manifest_row["sample_id"])
    label_rel = manifest_row["label"]["path"]
    label_path = dataset_root / label_rel
    label_obj = _read_json(label_path)

    pattern = str(manifest_row.get("pattern") or manifest_row.get("meta", {}).get("pattern") or "unknown")
    difficulty_score = float(
        manifest_row.get("difficulty")
        if manifest_row.get("difficulty") is not None
        else manifest_row.get("meta", {}).get("difficulty", 0.0)
    )

    qa = _extract_qa(label_obj)
    num_conflicts_maneuver = int(qa.global_metrics.get("num_conflicts_maneuver", 0))
    num_rules = int(qa.global_metrics.get("num_rules", 0))
    difficulty_level = infer_difficulty_level(
        difficulty_score=difficulty_score,
        num_conflicts_maneuver=num_conflicts_maneuver,
        num_rules=num_rules,
    )

    inputs = BenchmarkInputs(
        topdown_image=manifest_row.get("images", {}).get("topdown"),
        radar_image=manifest_row.get("images", {}).get("radar"),
        scene_spec=manifest_row.get("spec", {}) or {},
        render_meta=(manifest_row.get("meta", {}) or {}).get("render", {}) or {},
    )

    labels = _extract_labels(label_obj)

    meta: Dict[str, Any] = {
        "source_manifest_row": manifest_row,
        "label_snapshot_s": (label_obj.get("meta", {}) or {}).get("label_snapshot_s"),
    }

    return BenchmarkSample(
        sample_id=sample_id,
        pattern=pattern,
        split="unspecified",
        difficulty_score=difficulty_score,
        difficulty_level=difficulty_level,
        inputs=inputs,
        labels=labels,
        qa=qa,
        meta=meta,
    )


def build_benchmark_from_dataset(
    dataset_root: str | Path,
    manifest_name: str = "manifest.jsonl",
) -> List[BenchmarkSample]:
    root = Path(dataset_root).resolve()
    manifest_path = root / manifest_name
    rows = _read_jsonl(manifest_path)
    return [_build_one_sample(r, root) for r in rows]


def write_benchmark_jsonl(
    samples: Iterable[BenchmarkSample],
    out_path: str | Path,
) -> None:
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")