import json
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import Features
from colreg_reasoner.resolver import resolve_from_aggregate
from colreg_reasoner.priority import default_priorities_path


def _all_sources(dct):
    # primitive -> tuple(rule_id,...)
    for _, srcs in dct.items():
        for s in srcs:
            yield s


def test_resolved_labels_sources_traceable():
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent
    rules_path = str(repo_root / "colreg_kernel" / "rules" / "colregs_atomic.yaml")
    prio_path = default_priorities_path()

    gold_path = repo_root / "colreg_kernel" / "tests" / "gold_cases.jsonl"
    cases = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]

    # sample a subset for speed (use meta difficulty buckets via deterministic stride)
    subset = [cases[i] for i in range(0, len(cases), max(1, len(cases)//25))][:25]

    for c in subset:
        feats = Features.model_validate(c["features"])
        agg = aggregate(feats, strict=False, rules_path=rules_path)
        res = resolve_from_aggregate(feats, agg, strict=False, rules_path=rules_path, priorities_path=prio_path)

        kept = set(res.kept_rules)

        # each label primitive must have at least one source and all sources must be in kept rules
        for mapping in (
            res.labels.maneuver_allowed,
            res.labels.maneuver_forbidden,
            res.labels.lights_required,
            res.labels.lights_forbidden,
            res.labels.sounds_required,
            res.labels.sounds_forbidden,
        ):
            for prim, srcs in mapping.items():
                assert len(srcs) >= 1, f"{c['name']} primitive {prim} has no sources"
                assert set(srcs).issubset(kept), f"{c['name']} primitive {prim} has sources outside kept rules: {srcs}"
