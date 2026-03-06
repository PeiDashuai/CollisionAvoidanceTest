import json
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import Features
from colreg_reasoner.resolver import resolve_from_aggregate
from colreg_reasoner.priority import default_priorities_path


def test_explanation_schema_has_required_stages():
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent
    rules_path = str(repo_root / "colreg_kernel" / "rules" / "colregs_atomic.yaml")
    prio_path = default_priorities_path()

    gold_path = repo_root / "colreg_kernel" / "tests" / "gold_cases.jsonl"
    case = json.loads(gold_path.read_text().splitlines()[0])
    feats = Features.model_validate(case["features"])
    agg = aggregate(feats, strict=False, rules_path=rules_path)
    res = resolve_from_aggregate(feats, agg, strict=False, rules_path=rules_path, priorities_path=prio_path)

    d = res.explanation.to_dict()
    assert "steps" in d and isinstance(d["steps"], list) and d["steps"]
    stages = {s["stage"] for s in d["steps"]}
    # baseline required stages
    for required in ["domain", "responsibility", "encounter", "area", "signals"]:
        assert required in stages
