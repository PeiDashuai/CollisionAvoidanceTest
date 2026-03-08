from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


SplitName = Literal["train", "val", "test", "unspecified"]
DifficultyName = Literal["easy", "medium", "hard", "unspecified"]


class BenchmarkInputs(BaseModel):
    topdown_image: Optional[str] = None
    radar_image: Optional[str] = None
    scene_spec: Dict[str, Any] = Field(default_factory=dict)
    render_meta: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkLabels(BaseModel):
    triggered_rules: List[str] = Field(default_factory=list)

    maneuver_allowed: Dict[str, List[str]] = Field(default_factory=dict)
    maneuver_forbidden: Dict[str, List[str]] = Field(default_factory=dict)

    lights_required: Dict[str, List[str]] = Field(default_factory=dict)
    lights_forbidden: Dict[str, List[str]] = Field(default_factory=dict)

    sounds_required: Dict[str, List[str]] = Field(default_factory=dict)
    sounds_forbidden: Dict[str, List[str]] = Field(default_factory=dict)

    suppressed_rules: List[Dict[str, Any]] = Field(default_factory=list)
    explanation_steps: List[Dict[str, Any]] = Field(default_factory=list)


class BenchmarkQA(BaseModel):
    sanity_issues: List[Dict[str, Any]] = Field(default_factory=list)
    signals_issues: List[Dict[str, Any]] = Field(default_factory=list)
    sampled_metrics: Dict[str, Any] = Field(default_factory=dict)
    global_metrics: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkSample(BaseModel):
    sample_id: str
    pattern: str = "unknown"
    split: SplitName = "unspecified"

    difficulty_score: float = 0.0
    difficulty_level: DifficultyName = "unspecified"

    inputs: BenchmarkInputs
    labels: BenchmarkLabels
    qa: BenchmarkQA

    meta: Dict[str, Any] = Field(default_factory=dict)


def infer_difficulty_level(
    difficulty_score: float,
    num_conflicts_maneuver: int = 0,
    num_rules: int = 0,
) -> DifficultyName:
    """
    Heuristic mapping from current Part2/Part3 metrics to benchmark difficulty bins.
    """
    if num_conflicts_maneuver >= 2 or difficulty_score >= 6.0 or num_rules >= 12:
        return "hard"
    if num_conflicts_maneuver >= 1 or difficulty_score >= 2.0 or num_rules >= 8:
        return "medium"
    return "easy"