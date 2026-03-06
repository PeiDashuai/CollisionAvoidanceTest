import json
from pathlib import Path

from colreg_kernel.engine import evaluate_all
from colreg_kernel.features import Features
from colreg_kernel.rules import load_rules
from colreg_kernel.tri import Tri
from colreg_kernel.validator import validate_signals_outputs


def test_signals_outputs_no_contradictions_on_gold_cases():
    base = Path(__file__).resolve().parents[1]
    rules = load_rules(base / "rules" / "colregs_atomic.yaml")

    gold_path = base / "tests" / "gold_cases.jsonl"
    lines = gold_path.read_text(encoding="utf-8").strip().splitlines()

    failures = []
    for line in lines:
        item = json.loads(line)
        feats = Features.model_validate(item["features"])
        evals = evaluate_all(rules, feats)

        lights_req = []
        lights_forb = []
        sounds_req = []
        sounds_forb = []

        for e in evals:
            if e.value != Tri.TRUE:
                continue
            r = rules[e.rule_id]
            lights_req.extend([c.primitive for c in r.actions.signals.lights_shapes_required])
            lights_forb.extend([c.primitive for c in r.actions.signals.lights_shapes_forbidden])
            sounds_req.extend([c.primitive for c in r.actions.signals.sounds_required])
            sounds_forb.extend([c.primitive for c in r.actions.signals.sounds_forbidden])

        issues = validate_signals_outputs(
            features=feats,
            lights_required=lights_req,
            lights_forbidden=lights_forb,
            sounds_required=sounds_req,
            sounds_forbidden=sounds_forb,
        )
        if issues:
            failures.append((item.get("name"), [(i.code, i.message) for i in issues]))

    assert not failures, f"Signals legality failures (first 5): {failures[:5]}"
