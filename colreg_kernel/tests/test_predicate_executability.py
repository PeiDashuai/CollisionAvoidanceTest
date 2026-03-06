import random
from pathlib import Path

from colreg_kernel.features import (
    AreaType,
    BearingSector,
    EncounterType,
    Features,
    TrafficLaneRelation,
    VesselInfo,
    VesselClass,
    Visibility,
)
from colreg_kernel.engine import applicable
from colreg_kernel.rules import load_rules


def _rand_choice(enum_cls):
    return random.choice(list(enum_cls))


def test_all_rules_executable_no_exceptions():
    random.seed(0)
    rules_path = Path(__file__).resolve().parents[1] / "rules" / "colregs_atomic.yaml"
    rules = load_rules(rules_path)

    for _ in range(50):
        feats = Features(
            visibility=_rand_choice(Visibility),
            area_type=_rand_choice(AreaType),
            traffic_lane_relation=_rand_choice(TrafficLaneRelation),
            ownship=VesselInfo(vessel_class=_rand_choice(VesselClass)),
            target=VesselInfo(vessel_class=_rand_choice(VesselClass)),
            encounter_type=_rand_choice(EncounterType),
            relative_bearing_sector=_rand_choice(BearingSector),
            is_target_on_starboard=random.choice([True, False]),
            closing=random.choice([True, False]),
        )
        for r in rules.values():
            tv = applicable(r, feats)
            assert tv.value.value in {"true", "false", "unknown"}
