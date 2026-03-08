from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal
import numpy as np

from colreg_kernel.features import Features
from colreg_reasoner.multiagent import resolve_multi

from .spec import SceneSpec, VesselSpec, Domain
from .sim_kinematics import simulate_scene
from .geometry import (
    compute_pairwise_geometry,
    cpa_tcpa_from_states,
    _relative_bearing,
    _bearing_sector,
)
from .feature_builder import build_features_for_target


TrafficLaneRelation = Literal[
    "not_in_tss",
    "in_lane_same_dir",
    "in_lane_opposite_dir",
    "crossing_lane",
    "in_separation_zone",
    "inshore_zone",
]


@dataclass(frozen=True)
class ConflictConstraints:
    min_rules: int = 8
    min_conflicts: int = 1
    min_tensions: int = 0
    min_complexity_score: float = 2.0

    min_conflicts_maneuver: int = 1
    min_conflicts_signals: int = 0
    min_conflicts_domain: int = 0

    required_rule_ids: Tuple[str, ...] = ()
    required_rule_prefix: Tuple[str, ...] = ()
    required_target_classes: Tuple[str, ...] = ()

    max_tries: int = 800


@dataclass(frozen=True)
class PatternSpec:
    name: str
    domain: Domain
    encounter: Literal["crossing", "head_on", "overtaking", "mixed"] = "mixed"

    visibility: Literal["clear", "restricted"] = "clear"
    in_sight: bool = True

    traffic_lane_relation: TrafficLaneRelation = "not_in_tss"

    intends_cross_channel: Optional[bool] = None
    intends_anchor: Optional[bool] = None
    agreement_received_for_overtake: Optional[bool] = None

    force_conflict: bool = False


def _norm_class(x: str) -> str:
    return (x or "").strip().lower()


def _default_rules_path() -> str:
    import colreg_kernel  # type: ignore
    from pathlib import Path

    pkg_file = Path(colreg_kernel.__file__).resolve()
    repo_root = pkg_file.parents[2]
    p = repo_root / "rules" / "colregs_atomic.yaml"
    return str(p)

def _snapshot_index_for_label(scene: SceneSpec) -> int:
    """
    Choose a labeling snapshot early enough that future-TCPA contacts
    are still active for restricted_multi.
    For duration=120s, use ~30s.
    """
    k = int(round(30.0 / scene.dt_s))
    k = max(0, min(int(scene.duration_s / scene.dt_s) - 1, k))
    return k


def _heading_to_unit(heading_deg: float) -> np.ndarray:
    rad = np.deg2rad(heading_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def _make_ownship(rng: np.random.Generator) -> VesselSpec:
    return VesselSpec(
        vessel_id="ownship",
        vessel_class="power",
        length_m=float(rng.integers(60, 200)),
        nav_status="underway_making_way",
        x_m=0.0,
        y_m=0.0,
        heading_deg=float(rng.choice([0.0, 90.0])),
        speed_mps=float(rng.uniform(6.0, 10.0)),
    )


def _finalize_target(t: VesselSpec, rng: np.random.Generator) -> VesselSpec:
    t.vessel_class = _norm_class(t.vessel_class)
    if t.vessel_class == "fishing":
        t.fishing_mode = str(rng.choice(["trawling", "other"]))
    return t

def _body_axes(heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (right, forward) unit vectors in world frame.
    """
    fwd = _heading_to_unit(heading_deg)
    right = np.array([fwd[1], -fwd[0]], dtype=float)
    return right, fwd

def _construct_target_from_relative_body_position(
    own: VesselSpec,
    *,
    vessel_id: str,
    vessel_class: str,
    length_m: float,
    nav_status: str,
    rel_right_m: float,
    rel_forward_m: float,
    target_heading_deg: float,
    target_speed_mps: float,
) -> VesselSpec:
    """
    Construct a target directly from ownship body-frame relative position:
      rel_right_m   > 0 => starboard
      rel_forward_m > 0 => ahead
    """
    right, fwd = _body_axes(own.heading_deg)
    p_own = np.array([own.x_m, own.y_m], dtype=float)
    p_tgt = p_own + right * float(rel_right_m) + fwd * float(rel_forward_m)

    return VesselSpec(
        vessel_id=vessel_id,
        vessel_class=vessel_class,
        length_m=float(length_m),
        nav_status=nav_status,
        x_m=float(p_tgt[0]),
        y_m=float(p_tgt[1]),
        heading_deg=float(target_heading_deg),
        speed_mps=float(target_speed_mps),
    )

def _construct_target_from_future_cpa(
    own: VesselSpec,
    *,
    vessel_id: str,
    vessel_class: str,
    length_m: float,
    nav_status: str,
    target_heading_deg: float,
    target_speed_mps: float,
    tcpa_target_s: float,
    dcpa_target_m: float,
    side_sign: float,
) -> VesselSpec:
    """
    Construct target current position so that:
      - future TCPA is approximately tcpa_target_s
      - future DCPA is approximately dcpa_target_m
    using:
        r0 = -v_rel * tcpa + sign * dcpa * n_perp(v_rel)

    side_sign = +1 or -1 controls which side of relative-velocity normal.
    """
    p_own = np.array([own.x_m, own.y_m], dtype=float)
    v_own = _heading_to_unit(own.heading_deg) * float(own.speed_mps)
    v_tgt = _heading_to_unit(target_heading_deg) * float(target_speed_mps)
    v_rel = v_tgt - v_own

    v_rel_norm = float(np.linalg.norm(v_rel))
    if v_rel_norm < 1e-6:
        # avoid degenerate relative velocity
        v_rel = v_rel + np.array([0.1, 0.0], dtype=float)
        v_rel_norm = float(np.linalg.norm(v_rel))

    # perpendicular unit normal to relative velocity
    n_perp = np.array([-v_rel[1], v_rel[0]], dtype=float) / v_rel_norm

    r0 = -v_rel * float(tcpa_target_s) + float(side_sign) * float(dcpa_target_m) * n_perp
    p_tgt = p_own + r0

    return VesselSpec(
        vessel_id=vessel_id,
        vessel_class=vessel_class,
        length_m=float(length_m),
        nav_status=nav_status,
        x_m=float(p_tgt[0]),
        y_m=float(p_tgt[1]),
        heading_deg=float(target_heading_deg),
        speed_mps=float(target_speed_mps),
    )

def _construct_restricted_multi_targets_deterministic(
    own: VesselSpec,
    rng: np.random.Generator,
    n_targets: int,
) -> list[VesselSpec]:
    """
    Deterministic 3-role construction for restricted_multi:
      t1: crossing from starboard  -> R15/R16
      t2: starboard beam/quarter   -> R19D2
      t3: port beam/quarter        -> R19D3

    Design choice:
      - t1 uses future-CPA construction because it is the main encounter target.
      - t2 / t3 use direct body-frame placement to stabilize current bearing sector.
    """
    targets: list[VesselSpec] = []

    # -------------------------------------------------
    # t1: crossing from starboard, future TCPA ~ 55s
    # -------------------------------------------------
    if abs(own.heading_deg - 0.0) < 1e-6:
        t1_hdg = 270.0   # westbound crossing
    else:
        t1_hdg = 0.0     # northbound crossing

    t1 = _construct_target_from_future_cpa(
        own,
        vessel_id="t1",
        vessel_class=str(rng.choice(["power", "sailing", "fishing"])),
        length_m=float(rng.integers(20, 140)),
        nav_status="underway_making_way",
        target_heading_deg=t1_hdg,
        target_speed_mps=float(rng.uniform(4.5, 8.0)),
        tcpa_target_s=55.0,
        dcpa_target_m=120.0,
        side_sign=+1.0,
    )
    t1 = _finalize_target(t1, rng)
    targets.append(t1)

    # -------------------------------------------------
    # t2: starboard beam / quarter, stabilize sector for R19D2
    # current sector target: starboard_beam / starboard_quarter
    # -------------------------------------------------
    if abs(own.heading_deg - 0.0) < 1e-6:
        t2_hdg = float(rng.choice([0.0, 8.0, 352.0]))     # roughly northbound
    else:
        t2_hdg = float(rng.choice([90.0, 98.0, 82.0]))    # roughly eastbound

    t2 = _construct_target_from_relative_body_position(
        own,
        vessel_id="t2",
        vessel_class=str(rng.choice(["power", "ram", "nuc"])),
        length_m=float(rng.integers(30, 180)),
        nav_status="underway_making_way",
        rel_right_m=float(rng.uniform(500.0, 900.0)),
        rel_forward_m=float(rng.uniform(-500.0, -120.0)),
        target_heading_deg=t2_hdg,
        target_speed_mps=float(rng.uniform(2.5, 6.0)),
    )
    t2 = _finalize_target(t2, rng)
    targets.append(t2)

    # -------------------------------------------------
    # t3: port beam / quarter, stabilize sector for R19D3
    # current sector target: port_beam / port_quarter
    # -------------------------------------------------
    if abs(own.heading_deg - 0.0) < 1e-6:
        t3_hdg = float(rng.choice([180.0, 188.0, 172.0]))   # roughly southbound
    else:
        t3_hdg = float(rng.choice([270.0, 278.0, 262.0]))   # roughly westbound

    t3 = _construct_target_from_relative_body_position(
        own,
        vessel_id="t3",
        vessel_class=str(rng.choice(["power", "ram", "nuc"])),
        length_m=float(rng.integers(30, 180)),
        nav_status="underway_making_way",
        rel_right_m=-float(rng.uniform(500.0, 900.0)),
        rel_forward_m=float(rng.uniform(-500.0, -120.0)),
        target_heading_deg=t3_hdg,
        target_speed_mps=float(rng.uniform(2.5, 6.0)),
    )
    t3 = _finalize_target(t3, rng)
    targets.append(t3)

    # -------------------------------------------------
    # Extra targets
    # -------------------------------------------------
    for i in range(4, n_targets + 1):
        if abs(own.heading_deg - 0.0) < 1e-6:
            hdg = float(rng.choice([255.0, 270.0, 285.0]))
        else:
            hdg = float(rng.choice([345.0, 0.0, 15.0]))

        tx = _construct_target_from_future_cpa(
            own,
            vessel_id=f"t{i}",
            vessel_class=str(rng.choice(["power", "sailing", "fishing"])),
            length_m=float(rng.integers(20, 140)),
            nav_status="underway_making_way",
            target_heading_deg=hdg,
            target_speed_mps=float(rng.uniform(3.0, 7.5)),
            tcpa_target_s=float(rng.uniform(40.0, 90.0)),
            dcpa_target_m=float(rng.uniform(150.0, 350.0)),
            side_sign=float(rng.choice([+1.0, -1.0])),
        )
        tx = _finalize_target(tx, rng)
        targets.append(tx)

    return targets

def _try_build_target_with_sector(
    own: VesselSpec,
    rng: np.random.Generator,
    *,
    idx: int,
    vessel_classes: list[str],
    heading_candidates_deg: list[float],
    speed_range: tuple[float, float],
    tcpa_target_s: float,
    dcpa_target_m: float,
    desired_sectors: tuple[str, ...],
    side_sign_candidates: tuple[float, ...] = (+1.0, -1.0),
    tcpa_low_s: float = 20.0,
    tcpa_high_s: float = 150.0,
    max_random_trials: int = 120,
) -> Optional[VesselSpec]:
    """
    More robust search:
    1) keep only candidates that match desired sector and have positive future tcpa
    2) among them, prefer ones inside [low, high]
    3) otherwise choose the one closest to tcpa_target_s
    """
    class _Tmp:
        def __init__(self, x, y, h, s):
            self.x_m = x
            self.y_m = y
            self.heading_deg = h
            self.speed_mps = s

    own_state_tmp = _Tmp(own.x_m, own.y_m, own.heading_deg, own.speed_mps)

    headings = heading_candidates_deg[:]
    rng.shuffle(headings)

    trials: list[tuple[float, float]] = []
    for hdg in headings:
        for side_sign in side_sign_candidates:
            trials.append((float(hdg), float(side_sign)))

    for _ in range(max_random_trials):
        base = float(rng.choice(headings))
        jitter = float(rng.uniform(-30.0, 30.0))
        side_sign = float(rng.choice(side_sign_candidates))
        trials.append((base + jitter, side_sign))

    best_in_window: Optional[tuple[float, VesselSpec]] = None
    best_relaxed: Optional[tuple[float, VesselSpec]] = None

    for hdg, side_sign in trials:
        sp = float(rng.uniform(speed_range[0], speed_range[1]))
        t = _construct_target_from_future_cpa(
            own,
            vessel_id=f"t{idx}",
            vessel_class=str(rng.choice(vessel_classes)),
            length_m=float(rng.integers(20, 180)),
            nav_status="underway_making_way",
            target_heading_deg=float(hdg),
            target_speed_mps=sp,
            tcpa_target_s=float(tcpa_target_s),
            dcpa_target_m=float(dcpa_target_m),
            side_sign=float(side_sign),
        )

        rel = np.array([t.x_m - own.x_m, t.y_m - own.y_m], dtype=float)
        b = _relative_bearing(own.heading_deg, rel)
        sec = _bearing_sector(b)

        if sec not in desired_sectors:
            continue

        tgt_state_tmp = _Tmp(t.x_m, t.y_m, t.heading_deg, t.speed_mps)
        _, tcpa_future, tcpa_raw = cpa_tcpa_from_states(own_state_tmp, tgt_state_tmp)

        if not np.isfinite(tcpa_future) or not np.isfinite(tcpa_raw):
            continue
        if tcpa_raw <= 0.0:
            continue
        if tcpa_future <= 5.0:
            continue

        score = abs(tcpa_future - tcpa_target_s)

        if tcpa_low_s <= tcpa_future <= tcpa_high_s:
            if best_in_window is None or score < best_in_window[0]:
                best_in_window = (score, t)
        else:
            if best_relaxed is None or score < best_relaxed[0]:
                best_relaxed = (score, t)

    if best_in_window is not None:
        return best_in_window[1]

    if best_relaxed is not None:
        return best_relaxed[1]

    return None

def _template_target_forward_contact_future(own: VesselSpec, rng: np.random.Generator, idx: int) -> Optional[VesselSpec]:
    """
    Crossing target from starboard to trigger R15/R16.
    """
    if abs(own.heading_deg - 0.0) < 1e-6:
        heading_candidates = [255.0, 270.0, 285.0]
    else:
        heading_candidates = [345.0, 0.0, 15.0]

    return _try_build_target_with_sector(
        own,
        rng,
        idx=idx,
        vessel_classes=["power", "sailing", "fishing", "ram", "nuc"],
        heading_candidates_deg=heading_candidates,
        speed_range=(4.0, 9.0),
        tcpa_target_s=55.0,
        dcpa_target_m=150.0,
        desired_sectors=("starboard_bow", "starboard_beam"),
        side_sign_candidates=(+1.0, -1.0),
        tcpa_low_s=30.0,
        tcpa_high_s=120.0,
    )

def _template_target_starboard_beam_future(own: VesselSpec, rng: np.random.Generator, idx: int) -> Optional[VesselSpec]:
    """
    Starboard beam / quarter future contact for R19D2.
    """
    if abs(own.heading_deg - 0.0) < 1e-6:
        heading_candidates = [0.0, 15.0, 345.0, 20.0, 340.0]
    else:
        heading_candidates = [90.0, 105.0, 75.0, 110.0, 70.0]

    return _try_build_target_with_sector(
        own,
        rng,
        idx=idx,
        vessel_classes=["power", "ram", "nuc"],
        heading_candidates_deg=heading_candidates,
        speed_range=(2.0, 7.0),
        tcpa_target_s=70.0,
        dcpa_target_m=300.0,
        desired_sectors=("starboard_beam", "starboard_quarter"),
        side_sign_candidates=(+1.0, -1.0),
        tcpa_low_s=20.0,
        tcpa_high_s=150.0,
    )

def _template_target_port_beam_future(own: VesselSpec, rng: np.random.Generator, idx: int) -> Optional[VesselSpec]:
    """
    Port beam / quarter future contact for R19D3.
    """
    if abs(own.heading_deg - 0.0) < 1e-6:
        heading_candidates = [180.0, 200.0, 160.0, 185.0, 175.0]
    else:
        heading_candidates = [270.0, 250.0, 290.0, 265.0, 275.0]

    return _try_build_target_with_sector(
        own,
        rng,
        idx=idx,
        vessel_classes=["power", "ram", "nuc"],
        heading_candidates_deg=heading_candidates,
        speed_range=(2.0, 7.0),
        tcpa_target_s=75.0,
        dcpa_target_m=300.0,
        desired_sectors=("port_beam", "port_quarter"),
        side_sign_candidates=(+1.0, -1.0),
        tcpa_low_s=20.0,
        tcpa_high_s=150.0,
    )

def _template_target_crossing_starboard_future(own: VesselSpec, rng: np.random.Generator, idx: int) -> VesselSpec:
    """
    Crossing target from starboard to trigger R15/R16.
    Desired current sector: starboard_bow / starboard_beam
    """
    if abs(own.heading_deg - 0.0) < 1e-6:
        heading_candidates = [260.0, 270.0, 280.0]
    else:
        heading_candidates = [350.0, 0.0, 10.0]

    return _try_build_target_with_sector(
        own,
        rng,
        idx=idx,
        vessel_classes=["power", "sailing", "fishing", "ram", "nuc"],
        heading_candidates_deg=heading_candidates,
        speed_range=(3.0, 8.0),
        tcpa_target_s=60.0,
        dcpa_target_m=200.0,
        desired_sectors=("starboard_bow", "starboard_beam"),
        side_sign_candidates=(+1.0, -1.0),
    )




def _template_target_overtaking_future(own: VesselSpec, rng: np.random.Generator, idx: int) -> VesselSpec:
    if abs(own.heading_deg - 0.0) < 1e-6:
        tx = float(rng.uniform(-80.0, 80.0))
        ty = float(rng.uniform(700.0, 1400.0))
    else:
        tx = float(rng.uniform(700.0, 1400.0))
        ty = float(rng.uniform(-80.0, 80.0))
    sp = max(1.0, own.speed_mps - float(rng.uniform(1.5, 3.5)))
    return VesselSpec(
        vessel_id=f"t{idx}",
        vessel_class=str(rng.choice(["power", "fishing", "sailing"])),
        length_m=float(rng.integers(10, 120)),
        nav_status="underway_making_way",
        x_m=tx,
        y_m=ty,
        heading_deg=float(own.heading_deg),
        speed_mps=sp,
    )


def _enforce_positive_tcpa_window(
    own: VesselSpec,
    tgt: VesselSpec,
    target_tcpa_s: float = 60.0,
    low: float = 30.0,
    high: float = 120.0,
    max_iter: int = 6,
) -> VesselSpec:
    """
    Iteratively shift target along its own velocity direction so that future TCPA
    tends into [low, high]. This is a lightweight geometry correction for scenegen.
    """
    class _Tmp:
        def __init__(self, x, y, h, s):
            self.x_m = x
            self.y_m = y
            self.heading_deg = h
            self.speed_mps = s

    cur = tgt
    u_t = _heading_to_unit(cur.heading_deg)

    for _ in range(max_iter):
        own_s = _Tmp(own.x_m, own.y_m, own.heading_deg, own.speed_mps)
        tgt_s = _Tmp(cur.x_m, cur.y_m, cur.heading_deg, cur.speed_mps)

        _, tcpa_future, tcpa_raw = cpa_tcpa_from_states(own_s, tgt_s)

        if not np.isfinite(tcpa_raw):
            break
        if low <= tcpa_future <= high:
            break

        delta_t = float(target_tcpa_s - tcpa_future)
        # damped shift to avoid overshoot
        shift = u_t * cur.speed_mps * (0.75 * delta_t)

        cur = VesselSpec(
            vessel_id=cur.vessel_id,
            vessel_class=cur.vessel_class,
            length_m=cur.length_m,
            nav_status=cur.nav_status,
            x_m=float(cur.x_m + shift[0]),
            y_m=float(cur.y_m + shift[1]),
            heading_deg=cur.heading_deg,
            speed_mps=cur.speed_mps,
            fishing_mode=cur.fishing_mode,
            is_towing=cur.is_towing,
            tow_length_m=cur.tow_length_m,
            pilot_on_duty=cur.pilot_on_duty,
            sailing_using_engine=cur.sailing_using_engine,
        )

    return cur


def _make_targets(
    own: VesselSpec,
    rng: np.random.Generator,
    n_targets: int,
    encounter: str,
    force_conflict: bool,
) -> List[VesselSpec]:
    targets: List[VesselSpec] = []

    if force_conflict and n_targets >= 3:
        return _construct_restricted_multi_targets_deterministic(own, rng, n_targets)

    for i in range(1, n_targets + 1):
        if encounter == "crossing":
            t = _template_target_crossing_starboard_future(own, rng, i)

        elif encounter == "head_on":
            t = _template_target_forward_contact_future(own, rng, i)

        elif encounter == "overtaking":
            t = _template_target_overtaking_future(own, rng, i)

        else:
            choice = ["crossing", "head_on", "overtaking"][((i - 1) % 3)]
            if choice == "crossing":
                t = _template_target_crossing_starboard_future(own, rng, i)

            elif choice == "head_on":
                t = _template_target_forward_contact_future(own, rng, i)

            else:
                t = _template_target_overtaking_future(own, rng, i)


        targets.append(_finalize_target(t, rng))

    return targets


def _inject_feature_flags(feats: Features, pat: PatternSpec) -> Features:
    updates: Dict[str, Any] = {}

    if pat.domain == Domain.TSS:
        updates["area_type"] = "tss"
        updates["traffic_lane_relation"] = pat.traffic_lane_relation
    elif pat.domain == Domain.NARROW_CHANNEL:
        updates["area_type"] = "narrow_channel"

    if pat.intends_cross_channel is not None:
        updates["intends_cross_channel"] = pat.intends_cross_channel
    if pat.intends_anchor is not None:
        updates["intends_anchor"] = pat.intends_anchor
    if pat.agreement_received_for_overtake is not None:
        updates["agreement_received_for_overtake"] = pat.agreement_received_for_overtake

    updates["in_sight"] = pat.in_sight
    return feats.model_copy(update=updates)


def _metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    if metrics is None:
        return {}
    if hasattr(metrics, "to_dict"):
        return metrics.to_dict()
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    if hasattr(metrics, "__dict__"):
        return {k: v for k, v in metrics.__dict__.items() if not k.startswith("_")}
    return {"_repr": str(metrics)}


def _evaluate_scene(
    scene: SceneSpec,
    rules_path: str,
    pat: PatternSpec,
    cons: ConflictConstraints,
) -> Tuple[bool, Dict[str, Any]]:
    tracks = simulate_scene(scene)

    feats_list: List[Features] = []
    tcpa_list: List[float] = []

    label_k = _snapshot_index_for_label(scene)

    for tgt in scene.targets:
        geom = compute_pairwise_geometry(
            tracks,
            ownship_id=scene.ownship.vessel_id,
            target_id=tgt.vessel_id,
            t_index=label_k,
            domain=scene.domain.value,
            tss_lane_heading_deg=scene.tss_lane_heading_deg,
        )
        tcpa_list.append(float(geom.tcpa_s))

        feats = build_features_for_target(scene, tgt, geom, operational_params_path=None)
        feats = _inject_feature_flags(feats, pat)
        feats_list.append(feats)

    res = resolve_multi(feats_list, strict=False, rules_path=rules_path)
    global_res = getattr(res, "global_resolution", res)

    kept_rules = getattr(global_res, "kept_rules", []) or []
    metrics_dict = _metrics_to_dict(getattr(global_res, "metrics", None))

    ok = True

    if len(kept_rules) < cons.min_rules:
        ok = False

    for rid in cons.required_rule_ids:
        if rid not in kept_rules:
            ok = False
            break

    for pref in cons.required_rule_prefix:
        if not any(r.startswith(pref) for r in kept_rules):
            ok = False
            break

    if cons.required_target_classes:
        classes = {_norm_class(t.vessel_class) for t in scene.targets}
        for c in cons.required_target_classes:
            if _norm_class(c) not in classes:
                ok = False
                break

    if int(metrics_dict.get("num_conflicts", 0)) < cons.min_conflicts:
        ok = False
    if int(metrics_dict.get("num_tensions", 0)) < cons.min_tensions:
        ok = False
    if float(metrics_dict.get("complexity_score", 0.0)) < cons.min_complexity_score:
        ok = False
    if int(metrics_dict.get("num_conflicts_maneuver", 0)) < cons.min_conflicts_maneuver:
        ok = False
    if int(metrics_dict.get("num_conflicts_signals", 0)) < cons.min_conflicts_signals:
        ok = False
    if int(metrics_dict.get("num_conflicts_domain", 0)) < cons.min_conflicts_domain:
        ok = False

    # Stronger TCPA gate for restricted_multi:
    # - t1, t2 must both be in [30,120]
    # - remaining targets must have tcpa > 10
    if pat.name == "restricted_multi":
        # t1 is the main future encounter target
        if len(tcpa_list) >= 1:
            if not (20.0 <= tcpa_list[0] <= 120.0):
                ok = False

        # t2 / t3 are restricted-sector contacts; allow broader future window
        if len(tcpa_list) >= 2:
            if not (np.isfinite(tcpa_list[1]) and tcpa_list[1] >= 0.0):
                ok = False
        if len(tcpa_list) >= 3:
            if not (np.isfinite(tcpa_list[2]) and tcpa_list[2] >= 0.0):
                ok = False

        for t in tcpa_list[3:]:
            if not (np.isfinite(t) and t > 10.0):
                ok = False

    debug = {
        "kept_rules": tuple(kept_rules),
        "metrics": metrics_dict,
        "scene": scene.model_dump(),
        "tcpa_s_list": tcpa_list,
    }
    return ok, debug


def sample_scene_advanced(
    *,
    seed: int,
    pattern: PatternSpec,
    n_targets: int,
    constraints: ConflictConstraints,
    rules_path: Optional[str] = None,
) -> Tuple[SceneSpec, Dict[str, Any]]:
    rp = rules_path or _default_rules_path()
    rng = np.random.default_rng(seed)

    last_debug: Dict[str, Any] = {}

    for attempt in range(constraints.max_tries):
        own = _make_ownship(rng)

        targets = _make_targets(
            own=own,
            rng=rng,
            n_targets=n_targets,
            encounter=pattern.encounter,
            force_conflict=pattern.force_conflict,
        )

        if targets is None:
            continue

        scene = SceneSpec(
            seed=seed + attempt,
            domain=pattern.domain,
            visibility=pattern.visibility,
            in_sight=pattern.in_sight,
            duration_s=120.0,
            dt_s=1.0,
            ownship=own,
            targets=targets,
            tss_lane_heading_deg=float(rng.choice([0.0, 90.0])),
        )

        ok, dbg = _evaluate_scene(scene, rp, pattern, constraints)
        last_debug = dbg
        if ok:
            return scene, dbg

    raise RuntimeError(
        f"Failed to sample a scene satisfying constraints after {constraints.max_tries} tries. "
        f"Last metrics: {last_debug.get('metrics')}, "
        f"last kept_rules: {last_debug.get('kept_rules')}, "
        f"last tcpa_s_list: {last_debug.get('tcpa_s_list')}"
    )


def preset_patterns() -> Dict[str, PatternSpec]:
    return {
        "restricted_multi": PatternSpec(
            name="restricted_multi",
            domain=Domain.RESTRICTED,
            encounter="mixed",
            visibility="restricted",
            in_sight=False,
            force_conflict=True,
        ),
        "channel_crossing_impede": PatternSpec(
            name="channel_crossing_impede",
            domain=Domain.NARROW_CHANNEL,
            encounter="crossing",
            visibility="clear",
            in_sight=True,
            intends_cross_channel=True,
            force_conflict=False,
        ),
        "tss_crossing": PatternSpec(
            name="tss_crossing",
            domain=Domain.TSS,
            encounter="crossing",
            visibility="clear",
            in_sight=True,
            traffic_lane_relation="crossing_lane",
            force_conflict=False,
        ),
        "responsibility_mix": PatternSpec(
            name="responsibility_mix",
            domain=Domain.OPEN,
            encounter="crossing",
            visibility="clear",
            in_sight=True,
            force_conflict=False,
        ),
    }


def sample_scene(seed: int = 0, domain: Domain = Domain.OPEN, n_targets: int = 2) -> SceneSpec:
    pat = PatternSpec(
        name="legacy",
        domain=domain,
        encounter="mixed",
        visibility="restricted" if domain == Domain.RESTRICTED else "clear",
        in_sight=False if domain == Domain.RESTRICTED else True,
        force_conflict=False,
    )
    cons = ConflictConstraints(
        min_rules=0,
        min_conflicts=0,
        min_tensions=0,
        min_complexity_score=0.0,
        min_conflicts_maneuver=0,
        max_tries=50,
    )
    scene, _ = sample_scene_advanced(seed=seed, pattern=pat, n_targets=n_targets, constraints=cons)
    return scene