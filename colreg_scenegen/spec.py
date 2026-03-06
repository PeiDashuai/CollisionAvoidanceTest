from __future__ import annotations

from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Domain(str, Enum):
    OPEN = "open"
    NARROW_CHANNEL = "narrow_channel"
    TSS = "tss"
    RESTRICTED = "restricted"


VesselClass = Literal[
    "power",
    "sailing",
    "fishing",
    "ram",
    "nuc",
    "cbd",
    "constrained_by_draft",
    "aground",
]

NavStatus = Literal[
    "underway_making_way",
    "underway_not_making_way",
    "at_anchor",
    "aground",
]


class VesselSpec(BaseModel):
    vessel_id: str
    vessel_class: VesselClass = "power"
    length_m: float = 80.0
    nav_status: NavStatus = "underway_making_way"

    # Initial state (ENU meters, heading degrees (0=N, 90=E), speed m/s)
    x_m: float = 0.0
    y_m: float = 0.0
    heading_deg: float = 0.0
    speed_mps: float = 6.0

    # Optional flags
    fishing_mode: Optional[Literal["none", "trawling", "other"]] = "none"
    is_towing: bool = False
    tow_length_m: Optional[float] = None
    pilot_on_duty: bool = False
    sailing_using_engine: bool = False


class SceneSpec(BaseModel):
    seed: int = 0
    domain: Domain = Domain.OPEN
    visibility: Literal["clear", "restricted"] = "clear"
    in_sight: bool = True

    duration_s: float = 120.0
    dt_s: float = 1.0

    ownship: VesselSpec
    targets: List[VesselSpec] = Field(default_factory=list)

    # For TSS domain (optional, simplified)
    # lane direction heading in deg (0=N, 90=E)
    tss_lane_heading_deg: float = 0.0