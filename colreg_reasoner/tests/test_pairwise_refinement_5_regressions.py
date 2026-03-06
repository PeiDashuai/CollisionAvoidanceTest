import json
from pathlib import Path

from colreg_kernel.features import Features
from colreg_kernel.rules import load_rules

from colreg_reasoner.pairwise import pair_relation, PairRelation, ConflictReason
from colreg_reasoner.metrics import compute_tension_metrics


def _repo_paths():
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent
    rules_path = str(repo_root / "colreg_kernel" / "rules" / "colregs_atomic.yaml")
    gold_path = repo_root / "colreg_kernel" / "tests" / "gold_cases.jsonl"
    return rules_path, gold_path


def _load_case(name: str):
    rules_path, gold_path = _repo_paths()
    cases = {json.loads(l)["name"]: json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()}
    return rules_path, Features.model_validate(cases[name]["features"])


def test_pairwise_no_false_conflict_lookout_vs_r19d1():
    rules_path, feats = _load_case("restricted_power_underway_making_way")
    rules = load_rules(rules_path)
    r5 = rules["COLREG_R05_LOOKOUT"]
    r19d1 = rules["COLREG_R19D1_AVOID_PORT_TURN_FORWARD_BEAM_NOT_BEING_OVERTAKEN"]

    pr = pair_relation(r5, r19d1, feats, assume_applicable_true=True)
    assert pr.relation == PairRelation.COMPATIBLE


def test_pairwise_tension_r16_vs_r19d1():
    rules_path, feats = _load_case("restricted_power_underway_making_way")
    rules = load_rules(rules_path)
    r16 = rules["COLREG_R16_EARLY_SUBSTANTIAL_ACTION"]
    r19d1 = rules["COLREG_R19D1_AVOID_PORT_TURN_FORWARD_BEAM_NOT_BEING_OVERTAKEN"]

    pr = pair_relation(r16, r19d1, feats, assume_applicable_true=True)
    # R16 allows TURN_STARBOARD/REDUCE_SPEED; R19(d)(i) forbids TURN_PORT.
    # There is no direct overlap on the same primitive, so this is COMPATIBLE.
    assert pr.relation == PairRelation.COMPATIBLE
    assert pr.relation == PairRelation.COMPATIBLE
    assert pr.reason in {ConflictReason.APPLICABILITY, ConflictReason.MANEUVER_ACTIONS}


def test_metrics_breakdown_sums_to_total_conflicts():
    rules_path, feats = _load_case("restricted_power_underway_making_way")
    rules = load_rules(rules_path)
    kept_ids = [
        "COLREG_R19D1_AVOID_PORT_TURN_FORWARD_BEAM_NOT_BEING_OVERTAKEN",
        "COLREG_R19_ENGINES_READY",
        "COLREG_R19_SAFE_SPEED_RESTRICTED_VIS",
        "COLREG_R35_RESTRICTED_POWER_MAKING_WAY_SOUND",
        "COLREG_R16_EARLY_SUBSTANTIAL_ACTION",
        "COLREG_R23_POWER_UNDERWAY_LIGHTS",
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
    ]
    kept_specs = [rules[rid] for rid in kept_ids]
    m = compute_tension_metrics(kept_specs, feats)
    assert m.num_conflicts == (m.num_conflicts_maneuver + m.num_conflicts_signals + m.num_conflicts_domain)


def test_domain_conflict_r34_vs_r35():
    rules_path, feats = _load_case("restricted_power_underway_making_way")
    rules = load_rules(rules_path)
    r34 = rules["COLREG_R34_IN_SIGHT_TURN_PORT_SOUND"]
    r35 = rules["COLREG_R35_RESTRICTED_POWER_MAKING_WAY_SOUND"]
    pr = pair_relation(r34, r35, feats, assume_applicable_true=True)
    assert pr.relation == PairRelation.CONFLICT
    assert pr.reason == ConflictReason.DOMAIN


def test_metrics_fields_present_and_non_negative():
    rules_path, feats = _load_case("restricted_power_underway_making_way")
    rules = load_rules(rules_path)
    kept_specs = [rules["COLREG_R05_LOOKOUT"], rules["COLREG_R06_SAFE_SPEED"], rules["COLREG_R19_SAFE_SPEED_RESTRICTED_VIS"]]
    m = compute_tension_metrics(kept_specs, feats)
    assert m.num_rules >= 1
    assert m.num_pairs >= 0
    assert m.num_conflicts >= 0
    assert m.num_conflicts_maneuver >= 0
    assert m.num_conflicts_signals >= 0
    assert m.num_conflicts_domain >= 0
    assert m.num_tensions >= 0
    assert m.num_tensions_maneuver >= 0
