from __future__ import annotations
from enum import Enum
from typing import get_args, get_origin, Union
from typing import Optional, Any, Dict

from colreg_kernel.features import Features, VesselInfo
from colreg_kernel.derive_features import (
    load_operational_params,
    derive_risk_of_collision,
    derive_collision_imminent,
    derive_impedes,
    derive_tss_crossing_ok,
)

from .spec import SceneSpec, VesselSpec
from .geometry import PairwiseGeometry

from enum import Enum

def _coerce_enum_field(field: str, value: Any) -> Any:
    """
    Coerce a string into the Enum type used by colreg_kernel.features.Features.<field>.
    Handles Optional[Enum] / Union[Enum, None] annotations.
    """
    ann = Features.model_fields[field].annotation

    # unwrap Optional/Union
    origin = get_origin(ann)
    if origin is Union:
        for t in get_args(ann):
            if isinstance(t, type) and issubclass(t, Enum):
                return t(value)
        return value

    # direct Enum
    if isinstance(ann, type) and issubclass(ann, Enum):
        return ann(value)

    return value

def _coerce_feature_field(field: str, value: Any) -> Any:
    """
    Coerce value to the type expected by colreg_kernel.features.Features for a given field.
    If the field annotation is an Enum, cast Enum(value). Otherwise return as-is.
    """
    ann = Features.model_fields[field].annotation  # pydantic v2
    if isinstance(value, Enum):
        return value
    if isinstance(ann, type) and issubclass(ann, Enum):
        return ann(value)
    return value

def _norm_vessel_class(vc: str) -> str:
    s = (vc or "").strip().lower()
    mapping = {
        "ram": "ram",
        "nuc": "nuc",
        "cbd": "cbd",
        "constrained_by_draught": "constrained_by_draft",
        "constrained_by_draft": "constrained_by_draft",
        "power-driven": "power",
        "power_driven": "power",
    }
    return mapping.get(s, s)


def _to_vesselinfo(v: VesselSpec) -> VesselInfo:
    return VesselInfo(
        vessel_class=_norm_vessel_class(v.vessel_class),
        length_m=v.length_m,
        nav_status=v.nav_status,
        fishing_mode=v.fishing_mode,
        is_towing=v.is_towing,
        tow_length_m=v.tow_length_m,
        pilot_on_duty=v.pilot_on_duty,
        sailing_using_engine=v.sailing_using_engine,
        sailing_tack=None,
        is_windward=None,
    )


def build_features_for_target(
    scene: SceneSpec,
    target: VesselSpec,
    geom: PairwiseGeometry,
    operational_params_path: Optional[str] = None,
) -> Features:
    params: Dict[str, Any] = (
        load_operational_params(operational_params_path)
        if operational_params_path
        else load_operational_params()
    )

    own = scene.ownship
    own_length_m = float(own.length_m)

    # IMPORTANT: Part1 signature uses own_length_m
    risk = derive_risk_of_collision(
        cpa_m=geom.cpa_m,
        tcpa_s=geom.tcpa_s,
        own_length_m=own_length_m,
        params=params,
    )
    imminent = derive_collision_imminent(
        cpa_m=geom.cpa_m,
        tcpa_s=geom.tcpa_s,
        own_length_m=own_length_m,
        params=params,
    )

    imp_channel = None
    imp_tss = None
    if scene.domain.value in ["narrow_channel", "tss"]:
        # NOTE: if derive_impedes signature differs, run inspect.signature(derive_impedes)
        imp = derive_impedes(
            cpa_m=geom.cpa_m,
            tcpa_s=geom.tcpa_s,
            own_length_m=own_length_m,
            area_type=scene.domain.value,
            params=params,
        )
        if isinstance(imp, dict):
            imp_channel = bool(imp.get("impedes_channel_vessel")) if "impedes_channel_vessel" in imp else None
            imp_tss = bool(imp.get("impedes_tss_traffic")) if "impedes_tss_traffic" in imp else None

    tss_ok = None
    if scene.domain.value == "tss" and geom.tss_crossing_angle_deg is not None:
        # NOTE: if derive_tss_crossing_ok signature differs, run inspect.signature(derive_tss_crossing_ok)
        tss_ok = derive_tss_crossing_ok(
            angle_deg=float(geom.tss_crossing_angle_deg),
            params=params,
        )

    features = Features(
        visibility=_coerce_feature_field("visibility", scene.visibility),
        in_sight=scene.in_sight,
        area_type=_coerce_feature_field("area_type", scene.domain.value if scene.domain.value != "restricted" else "open"),
        traffic_lane_relation="not_in_tss",  # MVP
        encounter_type=geom.encounter_type,
        relative_bearing_sector=geom.relative_bearing_sector,
        is_target_on_starboard=geom.is_target_on_starboard,
        closing=geom.closing,
        cpa_m=geom.cpa_m,
        tcpa_s=geom.tcpa_s,
        risk_of_collision=bool(risk) if risk is not None else None,
        collision_imminent=bool(imminent) if imminent is not None else None,
        tss_crossing_angle_deg=geom.tss_crossing_angle_deg,
        tss_crossing_ok=tss_ok,
        impedes_channel_vessel=imp_channel,
        impedes_tss_traffic=imp_tss,
        ownship=_to_vesselinfo(scene.ownship),
        target=_to_vesselinfo(target),
        intent_turn=None,
        intent_overtake=None,
        intent_astern_propulsion=None,
        give_way_taking_action=None,
        being_overtaken=None,
        intends_cross_channel=None,
        intends_anchor=None,
        agreement_received_for_overtake=None,
        sailing_tack_relation=None,
    )
    return features