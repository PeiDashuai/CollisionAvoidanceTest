from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from colreg_kernel.aggregate import aggregate, AggregateResult
from colreg_kernel.features import Features

from colreg_reasoner.resolver import resolve, ResolutionResult
from colreg_reasoner.paths import default_rules_path


@dataclass(frozen=True)
class TargetAggregate:
    target_index: int
    aggregate: AggregateResult


@dataclass(frozen=True)
class MultiResolutionResult:
    per_target: Tuple[TargetAggregate, ...]
    global_resolution: ResolutionResult


def resolve_multi(
    features_list: Sequence[Features],
    *,
    rules_path: Optional[str] = None,
    priorities_path: Optional[str] = None,
    strict: bool = True,
) -> MultiResolutionResult:
    """
    Multi-target resolution:
      - Aggregate rules per target (ownship vs target_i) using Part-1 aggregate().
      - Union triggered rule ids across targets.
      - Resolve globally (Part-2) on a representative context feature (defaults to features_list[0]).
    """
    if not features_list:
        raise ValueError("features_list is empty")

    # If caller doesn't provide rules_path, locate Part-1 rules via package path.
    rules_path = rules_path or default_rules_path()

    per: List[TargetAggregate] = []
    seen = set()
    all_ids: List[str] = []

    for i, feats in enumerate(features_list):
        agg = aggregate(feats, strict=False, rules_path=rules_path)
        per.append(TargetAggregate(target_index=i, aggregate=agg))
        for tr in agg.triggered_rules:
            if tr.rule_id not in seen:
                seen.add(tr.rule_id)
                all_ids.append(tr.rule_id)

    global_res = resolve(
        features_list[0],
        all_ids,
        rules_path=rules_path,
        priorities_path=priorities_path,
        strict=strict,
    )

    return MultiResolutionResult(per_target=tuple(per), global_resolution=global_res)
