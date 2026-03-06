
import json
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import Features


def _flatten_sources(mapping: dict[str, tuple[str, ...]]) -> set[str]:
    out: set[str] = set()
    for _, srcs in mapping.items():
        out.update(srcs)
    return out


def test_label_sources_trace_to_triggered_rules():
    base = Path(__file__).resolve().parents[1]
    rules_path = str(base / "rules" / "colregs_atomic.yaml")

    gold_path = base / "tests" / "gold_cases.jsonl"
    lines = gold_path.read_text(encoding="utf-8").splitlines()

    # check a representative subset to keep test fast but meaningful
    subset_names = {
        "crossing_power_power_starboard_risk",
        "restricted_power_underway_making_way",
        "narrow_channel_crossing_impedes",
        "tss_in_sep_zone",
        "sailing_diff_tack_port_giveway",
        "lights_nuc",
        "restricted_sound_ram",
    }

    for line in lines:
        item = json.loads(line)
        name = item["name"]
        if name not in subset_names:
            continue

        feats = Features.model_validate(item["features"])
        res = aggregate(feats, rules_path=rules_path, strict=False)

        triggered = {t.rule_id for t in res.triggered_rules}

        # Every label primitive must be traceable to at least one triggered rule,
        # and sources must be a subset of triggered rules.
        for mapping in (
            res.labels.maneuver_allowed,
            res.labels.maneuver_forbidden,
            res.labels.lights_required,
            res.labels.lights_forbidden,
            res.labels.sounds_required,
            res.labels.sounds_forbidden,
        ):
            for prim, srcs in mapping.items():
                assert len(srcs) >= 1, f"case={name} prim={prim} has empty sources"
                assert set(srcs).issubset(triggered), f"case={name} prim={prim} sources not subset of triggered rules: {srcs} vs {triggered}"
