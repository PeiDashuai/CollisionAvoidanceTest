from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .spec import SceneSpec, VesselSpec


@dataclass(frozen=True)
class VesselState:
    t_s: float
    x_m: float
    y_m: float
    heading_deg: float
    speed_mps: float


def _heading_to_unit(heading_deg: float) -> np.ndarray:
    # 0 deg = North (y+), 90 deg = East (x+)
    rad = np.deg2rad(heading_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def simulate_vessel(v: VesselSpec, duration_s: float, dt_s: float) -> List[VesselState]:
    n = int(np.floor(duration_s / dt_s)) + 1
    pos = np.array([v.x_m, v.y_m], dtype=float)
    u = _heading_to_unit(v.heading_deg)
    out: List[VesselState] = []
    for k in range(n):
        t = k * dt_s
        out.append(VesselState(t_s=t, x_m=float(pos[0]), y_m=float(pos[1]),
                               heading_deg=float(v.heading_deg), speed_mps=float(v.speed_mps)))
        pos = pos + u * v.speed_mps * dt_s
    return out


def simulate_scene(spec: SceneSpec) -> Dict[str, List[VesselState]]:
    tracks: Dict[str, List[VesselState]] = {}
    tracks[spec.ownship.vessel_id] = simulate_vessel(spec.ownship, spec.duration_s, spec.dt_s)
    for t in spec.targets:
        tracks[t.vessel_id] = simulate_vessel(t, spec.duration_s, spec.dt_s)
    return tracks