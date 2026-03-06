"""COLREG reasoning (Part 2): pairwise relations + conflict resolution.

This package depends on `colreg_kernel` (Part 1: rule structuring & aggregation).
"""

from .pairwise import PairRelation, ConflictReason, PairResult, pair_relation
from .resolver import ResolutionResult, resolve, resolve_from_aggregate
from .priority import PriorityPolicy, load_priority_policy, rank_rule

from .multiagent import resolve_multi, MultiResolutionResult
from .metrics import TensionMetrics, compute_tension_metrics
from .graph import RuleGraph, build_rule_graph
from .explain import ExplanationChain, ExplanationStep

__all__ = [
    'PairRelation','ConflictReason','PairResult','pair_relation',
    'PriorityPolicy','load_priority_policy','rank_rule',
    'ResolutionResult','resolve','resolve_from_aggregate',
    'MultiResolutionResult','resolve_multi',
    'TensionMetrics','compute_tension_metrics',
    'RuleGraph','build_rule_graph',
    'ExplanationChain','ExplanationStep',
]

