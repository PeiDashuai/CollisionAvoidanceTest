from pathlib import Path

from colreg_kernel.rules import load_rules


def test_rule_schema_validates():
    rules_path = Path(__file__).resolve().parents[1] / "rules" / "colregs_atomic.yaml"
    rules = load_rules(rules_path)
    assert len(rules) >= 10
    # IDs unique is enforced by loader
