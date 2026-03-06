import json
from pathlib import Path

from colreg_kernel.features import Features
from colreg_reasoner.multiagent import resolve_multi


def _load_cases():
    base = Path(__file__).resolve().parents[1]
    repo_root = base.parent
    gold_path = repo_root / "colreg_kernel" / "tests" / "gold_cases.jsonl"
    cases = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
    return {c["name"]: c for c in cases}


def _merge_ownship(src_features: dict, ownship: dict) -> dict:
    f = dict(src_features)
    f["ownship"] = ownship
    return f


def test_multi_target_basic_invariants_10_scenarios():
    cases = _load_cases()
    # choose 10 diverse cases by name prefix if available; fall back to first ones
    names = list(cases.keys())
    picked = []
    for n in names:
        if any(k in n.lower() for k in ["cross", "tss", "channel", "restricted", "nuc", "ram", "fishing", "sailing"]):
            picked.append(n)
        if len(picked) >= 20:
            break
    if len(picked) < 20:
        picked = names[:20]

    # build 10 multi scenarios, each with 2 targets
    for i in range(10):
        a = cases[picked[2*i]]["features"]
        b = cases[picked[2*i+1]]["features"]
        # unify ownship to a's ownship for consistency
        b2 = _merge_ownship(b, a.get("ownship", {}))
        feats_list = [Features.model_validate(a), Features.model_validate(b2)]
        res = resolve_multi(feats_list, strict=False)

        # invariants: global kept rules should include union of per-target triggered ids (or a superset after suppression)
        per_union = set()
        for ta in res.per_target:
            per_union |= {t.rule_id for t in ta.aggregate.triggered_rules}
        kept = set(res.global_resolution.kept_rules)
        assert per_union.issubset(kept), "global kept_rules must cover per-target triggered rules (baseline policy)"

        # determinism: calling resolve_multi twice yields same kept_rules
        res2 = resolve_multi(feats_list, strict=False)
        assert res.global_resolution.kept_rules == res2.global_resolution.kept_rules
