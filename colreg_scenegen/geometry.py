from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import numpy as np

from .sim_kinematics import VesselState, _heading_to_unit


BearingSector = Literal[
    "ahead",
    "starboard_bow",
    "starboard_beam",
    "starboard_quarter",
    "astern",
    "port_quarter",
    "port_beam",
    "port_bow",
]

EncounterType = Literal["crossing", "head_on", "overtaking", "other"]


@dataclass(frozen=True)
class PairwiseGeometry:
    encounter_type: EncounterType
    is_target_on_starboard: bool
    relative_bearing_deg: float
    relative_bearing_sector: BearingSector
    closing: bool
    cpa_m: float
    tcpa_s: float
    tss_crossing_angle_deg: Optional[float] = None


def _wrap_deg(x: float) -> float:
    x = x % 360.0
    return x if x >= 0 else x + 360.0


def _bearing_sector(bearing_deg: float) -> BearingSector:
    # bearing_deg: 0 ahead, 90 starboard beam, 180 astern, 270 port beam
    b = _wrap_deg(bearing_deg)
    # 8 sectors, 45° each
    if b < 22.5 or b >= 337.5:
        return "ahead"
    if 22.5 <= b < 67.5:
        return "starboard_bow"
    if 67.5 <= b < 112.5:
        return "starboard_beam"
    if 112.5 <= b < 157.5:
        return "starboard_quarter"
    if 157.5 <= b < 202.5:
        return "astern"
    if 202.5 <= b < 247.5:
        return "port_quarter"
    if 247.5 <= b < 292.5:
        return "port_beam"
    return "port_bow"


def _relative_bearing(own_heading_deg: float, rel_vec_xy: np.ndarray) -> float:
    # bearing relative to ownship heading: 0 ahead, 90 starboard, 180 astern, 270 port
    # own heading unit:
    u = _heading_to_unit(own_heading_deg)
    # right unit:
    r = np.array([u[1], -u[0]], dtype=float)  # rotate u clockwise
    forward = float(np.dot(rel_vec_xy, u))
    right = float(np.dot(rel_vec_xy, r))
    ang = np.rad2deg(np.arctan2(right, forward))
    return _wrap_deg(ang)


def cpa_tcpa_from_states(own: VesselState, tgt: VesselState) -> tuple[float, float]:
    p = np.array([own.x_m, own.y_m], dtype=float)
    q = np.array([tgt.x_m, tgt.y_m], dtype=float)
    v = _heading_to_unit(own.heading_deg) * own.speed_mps
    w = _heading_to_unit(tgt.heading_deg) * tgt.speed_mps

    r0 = q - p
    vr = w - v
    vr2 = float(np.dot(vr, vr))
    if vr2 < 1e-9:
        # nearly same velocity
        return float(np.linalg.norm(r0)), float("inf")

    tcpa = -float(np.dot(r0, vr)) / vr2
    tcpa = max(tcpa, 0.0)  # only future
    cpa_vec = r0 + vr * tcpa
    dcpa = float(np.linalg.norm(cpa_vec))
    return dcpa, tcpa


def infer_encounter_type(own: VesselState, tgt: VesselState, rel_bearing_deg: float) -> EncounterType:
    # very simple heuristic:
    # - overtaking if target is roughly ahead of ownship? Actually overtaking is ownship approaching target from >112.5° abaft target beam.
    # Here we approximate using rel bearing from ownship perspective:
    #   if target is within astern sector => "overtaking" (ownship sees target astern => ownship being overtaken)
    # Better: use target's relative bearing of ownship, but we don't have it. Keep minimal.
    # - head_on if headings opposite-ish and target ahead-ish
    hdg_diff = abs(_wrap_deg(own.heading_deg - tgt.heading_deg))
    hdg_diff = min(hdg_diff, 360.0 - hdg_diff)

    if (hdg_diff > 150.0) and (_bearing_sector(rel_bearing_deg) in ["ahead", "starboard_bow", "port_bow"]):
        return "head_on"

    # approximate overtaking: target is close to ahead in target frame is hard; use own rel bearing near ahead but target is slower and closing
    return "crossing" if _bearing_sector(rel_bearing_deg) not in ["astern"] else "overtaking"


def compute_pairwise_geometry(
    tracks: Dict[str, List[VesselState]],
    ownship_id: str,
    target_id: str,
    t_index: int = -1,
    domain: str = "open",
    tss_lane_heading_deg: Optional[float] = None,
) -> PairwiseGeometry:
    own = tracks[ownship_id][t_index]
    tgt = tracks[target_id][t_index]

    rel = np.array([tgt.x_m - own.x_m, tgt.y_m - own.y_m], dtype=float)
    rel_bearing = _relative_bearing(own.heading_deg, rel)
    sector = _bearing_sector(rel_bearing)
    is_starboard = sector in ["starboard_bow", "starboard_beam", "starboard_quarter"]

    dcpa, tcpa = cpa_tcpa_from_states(own, tgt)
    # closing: range decreasing now? approximate by dot(r0, vr) < 0
    p = np.array([own.x_m, own.y_m], dtype=float)
    q = np.array([tgt.x_m, tgt.y_m], dtype=float)
    v = _heading_to_unit(own.heading_deg) * own.speed_mps
    w = _heading_to_unit(tgt.heading_deg) * tgt.speed_mps
    r0 = q - p
    vr = w - v
    closing = float(np.dot(r0, vr)) < 0.0

    encounter = infer_encounter_type(own, tgt, rel_bearing)

    tss_angle = None
    if domain == "tss" and tss_lane_heading_deg is not None:
        # crossing angle: abs(own heading - lane heading) modulo 180
        d = abs(_wrap_deg(own.heading_deg - tss_lane_heading_deg))
        d = min(d, 360.0 - d)
        d = min(d, 180.0 - abs(d - 180.0))
        tss_angle = float(d)

    return PairwiseGeometry(
        encounter_type=encounter,
        is_target_on_starboard=is_starboard,
        relative_bearing_deg=float(rel_bearing),
        relative_bearing_sector=sector,
        closing=bool(closing),
        cpa_m=float(dcpa),
        tcpa_s=float(tcpa),
        tss_crossing_angle_deg=tss_angle,
    )