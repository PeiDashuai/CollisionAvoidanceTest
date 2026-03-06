import json
from pathlib import Path

from colreg_kernel.engine import evaluate_all
from colreg_kernel.features import Features
from colreg_kernel.rules import load_rules
from colreg_kernel.tri import Tri


def test_rule_coverage_on_gold_cases():
    """Ensure the gold suite triggers almost all atomic rules at least once.

    This prevents "dead" rules that are never exercised and silently rot.
    """
    base = Path(__file__).resolve().parents[1]
    rules = load_rules(base / "rules" / "colregs_atomic.yaml")

    gold_path = base / "tests" / "gold_cases.jsonl"
    lines = gold_path.read_text(encoding="utf-8").strip().splitlines()

    triggered = {rid: 0 for rid in rules.keys()}

    for line in lines:
        item = json.loads(line)
        feats = Features.model_validate(item["features"])
        evals = evaluate_all(rules, feats)
        for e in evals:
            if e.value == Tri.TRUE:
                triggered[e.rule_id] += 1

    covered = [rid for rid, c in triggered.items() if c > 0]
    coverage = len(covered) / max(1, len(rules))

    # Allow a small fraction of edge rules to remain uncovered (e.g., rare branches).
    assert coverage >= 0.98, f"coverage={coverage:.3f} covered={len(covered)}/{len(rules)} uncovered={[rid for rid,c in triggered.items() if c==0][:10]}"
