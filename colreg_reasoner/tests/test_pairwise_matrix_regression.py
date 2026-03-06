import json
import itertools
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import Features
from colreg_reasoner.pairwise import pair_relation
from colreg_kernel.rules import load_rules


def test_pairwise_matrix_regression_20_cases_snapshot():
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent
    rules_path = str(repo_root / "colreg_kernel" / "rules" / "colregs_atomic.yaml")

    gold_path = repo_root / "colreg_kernel" / "tests" / "gold_cases.jsonl"
    cases = {json.loads(l)["name"]: json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()}

    expected = json.loads((base / "tests" / "expected_pairwise_20.json").read_text())

    rules = load_rules(rules_path)

    for name, exp_edges in expected.items():
        feats = Features.model_validate(cases[name]["features"])
        agg = aggregate(feats, strict=False, rules_path=rules_path)

        trig = [t.rule_id for t in agg.triggered_rules]
        got_edges = []
        for a, b in itertools.combinations(sorted(trig), 2):
            pr = pair_relation(rules[a], rules[b], feats, assume_applicable_true=True)
            got_edges.append([a, b, pr.relation.value, pr.reason.value])

        assert got_edges == exp_edges, f"Pairwise snapshot mismatch for case: {name}"
