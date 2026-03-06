import json
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import Features
from colreg_reasoner.resolver import resolve_from_aggregate
from colreg_reasoner.priority import default_priorities_path


def test_resolver_regression_20_cases_snapshot():
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent
    rules_path = str(repo_root / "colreg_kernel" / "rules" / "colregs_atomic.yaml")
    prio_path = default_priorities_path()

    gold_path = repo_root / "colreg_kernel" / "tests" / "gold_cases.jsonl"
    cases = {json.loads(l)["name"]: json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()}

    expected = json.loads((base / "tests" / "expected_resolver_20.json").read_text())
    for name, exp in expected.items():
        feats = Features.model_validate(cases[name]["features"])
        agg = aggregate(feats, strict=False, rules_path=rules_path)
        res = resolve_from_aggregate(feats, agg, strict=False, rules_path=rules_path, priorities_path=prio_path)

        got = {
            "triggered_rules": [t.rule_id for t in agg.triggered_rules],
            "kept_rules": list(res.kept_rules),
            "suppressed_rules": [{"rule_id": s.rule_id, "reason": s.reason} for s in res.suppressed_rules],
            "maneuver_allowed": sorted(res.labels.maneuver_allowed.keys()),
            "maneuver_forbidden": sorted(res.labels.maneuver_forbidden.keys()),
            "lights_required": sorted(res.labels.lights_required.keys()),
            "sounds_required": sorted(res.labels.sounds_required.keys()),
        }
        assert got == exp, f"Resolver snapshot mismatch for case: {name}"
