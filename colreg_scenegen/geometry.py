from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import numpy as np

from .sim_kinematics import VesselState


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


def _heading_to_unit(heading_deg: float) -> np.ndarray:
    rad = np.deg2rad(heading_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def _bearing_sector(bearing_deg: float) -> BearingSector:
    b = _wrap_deg(bearing_deg)
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
    """
    Bearing relative to ownship heading:
    0 ahead, 90 starboard, 180 astern, 270 port
    """
    u = _heading_to_unit(own_heading_deg)
    r = np.array([u[1], -u[0]], dtype=float)  # right
    forward = float(np.dot(rel_vec_xy, u))
    right = float(np.dot(rel_vec_xy, r))
    ang = np.rad2deg(np.arctan2(right, forward))
    return _wrap_deg(ang)


def cpa_tcpa_from_states(own: VesselState, tgt: VesselState) -> tuple[float, float, float]:
    """
    Returns:
      dcpa_future, tcpa_future, tcpa_raw

    tcpa_raw can be negative (closest approach in the past).
    tcpa_future is clamped to >=0 for downstream rule features.
    """
    p_own = np.array([own.x_m, own.y_m], dtype=float)
    p_tgt = np.array([tgt.x_m, tgt.y_m], dtype=float)

    v_own = _heading_to_unit(own.heading_deg) * float(own.speed_mps)
    v_tgt = _heading_to_unit(tgt.heading_deg) * float(tgt.speed_mps)

    r = p_tgt - p_own
    v_rel = v_tgt - v_own
    v_rel2 = float(np.dot(v_rel, v_rel))

    if v_rel2 < 1e-9:
        return float(np.linalg.norm(r)), float("inf"), float("inf")

    tcpa_raw = -float(np.dot(r, v_rel)) / v_rel2
    tcpa_future = max(tcpa_raw, 0.0)
    cpa_vec = r + v_rel * tcpa_future
    dcpa = float(np.linalg.norm(cpa_vec))
    return dcpa, tcpa_future, tcpa_raw


def infer_encounter_type(
    own: VesselState,
    tgt: VesselState,
    rel_bearing_deg: float,
    tcpa_s: float,
    closing: bool,
) -> EncounterType:
    sector = _bearing_sector(rel_bearing_deg)
    hdg_diff = abs(_wrap_deg(own.heading_deg - tgt.heading_deg))
    hdg_diff = min(hdg_diff, 360.0 - hdg_diff)

    # head-on
    if closing and np.isfinite(tcpa_s) and tcpa_s > 0 and hdg_diff > 150.0 and sector in ["ahead", "starboard_bow", "port_bow"]:
        return "head_on"

    # overtaking (simplified)
    if closing and np.isfinite(tcpa_s) and tcpa_s > 0 and hdg_diff < 25.0 and sector in ["ahead", "starboard_bow", "port_bow"] and own.speed_mps > tgt.speed_mps:
        return "overtaking"

    # crossing
    if closing and np.isfinite(tcpa_s) and tcpa_s > 0 and sector in [
        "starboard_bow", "starboard_beam", "port_bow", "port_beam"
    ]:
        return "crossing"

    return "other"


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

    dcpa, tcpa_future, tcpa_raw = cpa_tcpa_from_states(own, tgt)

    # closing semantics: only if future closest approach is ahead in time
    closing = bool(np.isfinite(tcpa_raw) and tcpa_raw > 0.0)

    encounter = infer_encounter_type(own, tgt, rel_bearing, tcpa_future, closing)

    tss_angle = None
    if domain == "tss" and tss_lane_heading_deg is not None:
        d = abs(_wrap_deg(own.heading_deg - tss_lane_heading_deg))
        d = min(d, 360.0 - d)
        d = min(d, 180.0 - abs(d - 180.0))
        tss_angle = float(d)

    return PairwiseGeometry(
        encounter_type=encounter,
        is_target_on_starboard=is_starboard,
        relative_bearing_deg=float(rel_bearing),
        relative_bearing_sector=sector,
        closing=closing,
        cpa_m=float(dcpa),
        tcpa_s=float(tcpa_future),
        tss_crossing_angle_deg=tss_angle,
    )