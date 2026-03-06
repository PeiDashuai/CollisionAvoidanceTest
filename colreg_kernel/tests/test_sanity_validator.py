import json
from pathlib import Path

from colreg_kernel.features import Features
from colreg_kernel.validator import validate_feature_consistency


def test_gold_cases_pass_sanity_validator():
    base = Path(__file__).resolve().parents[1]
    gold_path = base / "tests" / "gold_cases.jsonl"
    lines = gold_path.read_text(encoding="utf-8").strip().splitlines()

    failures = []
    for line in lines:
        item = json.loads(line)
        feats = Features.model_validate(item["features"])
        issues = validate_feature_consistency(feats)
        if issues:
            failures.append((item.get("name"), [(i.code, i.message) for i in issues]))

    assert not failures, f"Sanity validator failures (first 5): {failures[:5]}"
