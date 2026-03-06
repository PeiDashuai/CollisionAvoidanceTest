import json
from pathlib import Path

from colreg_kernel.engine import evaluate_all
from colreg_kernel.features import Features
from colreg_kernel.rules import load_rules
from colreg_kernel.tri import Tri


def test_gold_cases_expected_rules_trigger():
    base = Path(__file__).resolve().parents[1]
    rules = load_rules(base / "rules" / "colregs_atomic.yaml")

    gold_path = base / "tests" / "gold_cases.jsonl"
    lines = gold_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 5

    for line in lines:
        item = json.loads(line)
        # Curriculum metadata (used for stratified sampling later)
        meta = item.get("meta")
        assert isinstance(meta, dict), f"case={item.get('name')} missing meta"
        for k in ("difficulty_level", "n_vessels", "domain", "vessel_mix"):
            assert k in meta, f"case={item.get('name')} meta missing key: {k}"
        feats = Features.model_validate(item["features"])
        evals = evaluate_all(rules, feats)
        m = {e.rule_id: e for e in evals}

        for rid in item.get("expected_true", []):
            assert m[rid].value == Tri.TRUE, f"case={item['name']} expected TRUE: {rid} got {m[rid].value} missing={m[rid].missing_fields}"

        for rid in item.get("expected_false", []):
            assert m[rid].value == Tri.FALSE, f"case={item['name']} expected FALSE: {rid} got {m[rid].value} missing={m[rid].missing_fields}"
