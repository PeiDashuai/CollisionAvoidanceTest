
import json
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import Features


def test_aggregate_regression_20_cases_snapshot():
    base = Path(__file__).resolve().parents[1]
    rules_path = str(base / "rules" / "colregs_atomic.yaml")

    gold_path = base / "tests" / "gold_cases.jsonl"
    cases = {json.loads(l)["name"]: json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines()}

    exp_path = base / "tests" / "expected_aggregate_20.json"
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    selected = exp["selected"]
    snapshots = exp["snapshots"]

    for name in selected:
        item = cases[name]
        feats = Features.model_validate(item["features"])
        res = aggregate(feats, rules_path=rules_path, strict=False)
        got = {
            "triggered_rules": [t.rule_id for t in res.triggered_rules],
            "maneuver_allowed": sorted(res.labels.maneuver_allowed.keys()),
            "maneuver_forbidden": sorted(res.labels.maneuver_forbidden.keys()),
            "lights_required": sorted(res.labels.lights_required.keys()),
            "sounds_required": sorted(res.labels.sounds_required.keys()),
            "explanation_chain": list(res.explanation_chain),
        }
        assert got == snapshots[name], f"aggregate snapshot mismatch for case={name}"
