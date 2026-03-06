from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal
import numpy as np

from colreg_kernel.features import Features
from colreg_reasoner.multiagent import resolve_multi

from .spec import SceneSpec, VesselSpec, Domain
from .sim_kinematics import simulate_scene
from .geometry import compute_pairwise_geometry
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
    # size/complexity targets
    min_rules: int = 8
    min_conflicts: int = 1
    min_tensions: int = 0
    min_complexity_score: float = 2.0

    # optional stricter breakdown
    min_conflicts_maneuver: int = 1
    min_conflicts_signals: int = 0
    min_conflicts_domain: int = 0

    # require presence of certain rules in kept_rules
    required_rule_ids: Tuple[str, ...] = ()
    required_rule_prefix: Tuple[str, ...] = ()

    # require target vessel classes among targets (lowercase expected: "nuc","ram","fishing","sailing","power")
    required_target_classes: Tuple[str, ...] = ()

    max_tries: int = 800


@dataclass(frozen=True)
class PatternSpec:
    """
    Pattern recipe for generating multi-ship conflict scenes.
    """
    name: str
    domain: Domain
    encounter: Literal["crossing", "head_on", "overtaking", "mixed"] = "mixed"

    # Scene-level switches (do NOT inject into Features.visibility; that's handled in feature_builder)
    visibility: Literal["clear", "restricted"] = "clear"
    in_sight: bool = True

    # For TSS: lane relation injection
    traffic_lane_relation: TrafficLaneRelation = "not_in_tss"

    # Optional feature injections to trigger specific sub-clauses
    intends_cross_channel: Optional[bool] = None
    intends_anchor: Optional[bool] = None
    agreement_received_for_overtake: Optional[bool] = None

    # If True, use deterministic target templates to force maneuver conflicts (for early-stage data)
    force_conflict: bool = False


def _norm_class(x: str) -> str:
    return (x or "").strip().lower()


def _default_rules_path() -> str:
    # locate Part1 rule YAML via installed colreg_kernel path (editable-safe)
    import colreg_kernel  # type: ignore
    from pathlib import Path

    pkg_file = Path(colreg_kernel.__file__).resolve()
    repo_root = pkg_file.parents[2]  # .../colreg_kernel
    p = repo_root / "rules" / "colregs_atomic.yaml"
    return str(p)


def _make_ownship(rng: np.random.Generator) -> VesselSpec:
    return VesselSpec(
        vessel_id="ownship",
        vessel_class="power",
        length_m=float(rng.integers(60, 200)),
        nav_status="underway_making_way",
        x_m=0.0,
        y_m=0.0,
        heading_deg=float(rng.choice([0.0, 90.0])),  # N or E
        speed_mps=float(rng.uniform(6.0, 11.0)),
    )


def _template_target_forward_contact(own: VesselSpec, rng: np.random.Generator, idx: int) -> VesselSpec:
    """
    Put target roughly ahead to encourage R19(d)(i)-style "avoid port turn forward of beam"
    and also to create risk of collision more often.
    """
    r = float(rng.uniform(900.0, 1800.0))
    if abs(own.heading_deg - 0.0) < 1e-6:
        tx, ty = float(rng.uniform(-80.0, 80.0)), r
        th = 180.0  # opposite-ish
    else:
        tx, ty = r, float(rng.uniform(-80.0, 80.0))
        th = 270.0
    sp = float(rng.uniform(4.0, 10.0))
    return VesselSpec(
        vessel_id=f"t{idx}",
        vessel_class=str(rng.choice(["power", "ram", "nuc"])),
        length_m=float(rng.integers(20, 180)),
        nav_status="underway_making_way",
        x_m=tx,
        y_m=ty,
        heading_deg=th,
        speed_mps=sp,
    )


def _template_target_starboard_beam_contact(own: VesselSpec, rng: np.random.Generator, idx: int) -> VesselSpec:
    """
    Put target on starboard beam/quadrant to encourage R19(d)(ii)-style clauses that can forbid
    turning toward a vessel abeam/abaft the beam (often forbids TURN_STARBOARD).
    """
    r = float(rng.uniform(900.0, 1700.0))
    if abs(own.heading_deg - 0.0) < 1e-6:
        # own north: starboard is +x. place near beam: (x=+r, y small)
        tx, ty = r, float(rng.uniform(-120.0, 120.0))
        th = float(rng.choice([0.0, 180.0]))
    else:
        # own east: starboard is -y. place near beam: (y=-r, x small)
        tx, ty = float(rng.uniform(-120.0, 120.0)), -r
        th = float(rng.choice([90.0, 270.0]))
    sp = float(rng.uniform(2.0, 9.0))
    return VesselSpec(
        vessel_id=f"t{idx}",
        vessel_class=str(rng.choice(["power", "ram", "nuc"])),
        length_m=float(rng.integers(20, 180)),
        nav_status="underway_making_way",
        x_m=tx,
        y_m=ty,
        heading_deg=th,
        speed_mps=sp,
    )


def _template_target_crossing_starboard(own: VesselSpec, rng: np.random.Generator, idx: int) -> VesselSpec:
    """
    Put target on starboard side crossing to encourage R15 give-way.
    """
    r = float(rng.uniform(900.0, 2000.0))
    if abs(own.heading_deg - 0.0) < 1e-6:
        tx, ty = r, float(rng.uniform(-300.0, 300.0))
        th = 270.0
    else:
        tx, ty = float(rng.uniform(-300.0, 300.0)), -r
        th = 0.0
    sp = float(rng.uniform(3.0, 9.0))
    return VesselSpec(
        vessel_id=f"t{idx}",
        vessel_class=str(rng.choice(["power", "sailing", "fishing", "ram", "nuc"])),
        length_m=float(rng.integers(10, 150)),
        nav_status="underway_making_way",
        x_m=tx,
        y_m=ty,
        heading_deg=th,
        speed_mps=sp,
    )


def _template_target_overtaking_ahead(own: VesselSpec, rng: np.random.Generator, idx: int) -> VesselSpec:
    r = float(rng.uniform(700.0, 1500.0))
    u = float(rng.uniform(-120.0, 120.0))
    if abs(own.heading_deg - 0.0) < 1e-6:
        tx, ty = u, r
    else:
        tx, ty = r, u
    sp = max(1.0, own.speed_mps - float(rng.uniform(1.0, 4.0)))
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


def _finalize_target(t: VesselSpec, rng: np.random.Generator) -> VesselSpec:
    t.vessel_class = _norm_class(t.vessel_class)
    if t.vessel_class == "fishing":
        t.fishing_mode = str(rng.choice(["trawling", "other"]))
    return t


def _make_targets(
    own: VesselSpec,
    rng: np.random.Generator,
    n_targets: int,
    encounter: str,
    force_conflict: bool,
) -> List[VesselSpec]:
    targets: List[VesselSpec] = []

    if force_conflict and n_targets >= 2:
        # 1) forward contact
        targets.append(_finalize_target(_template_target_forward_contact(own, rng, 1), rng))
        # 2) starboard beam contact
        targets.append(_finalize_target(_template_target_starboard_beam_contact(own, rng, 2), rng))
        # fill the rest with crossing starboard or overtaking to increase rule variety
        for i in range(3, n_targets + 1):
            t = _template_target_crossing_starboard(own, rng, i) if (i % 2 == 1) else _template_target_overtaking_ahead(own, rng, i)
            targets.append(_finalize_target(t, rng))
        return targets

    # non-forced: mixed templates
    for i in range(1, n_targets + 1):
        if encounter == "crossing":
            t = _template_target_crossing_starboard(own, rng, i)
        elif encounter == "head_on":
            t = _template_target_forward_contact(own, rng, i)
        elif encounter == "overtaking":
            t = _template_target_overtaking_ahead(own, rng, i)
        else:
            choice = ["crossing", "head_on", "overtaking"][((i - 1) % 3)]
            t = _template_target_crossing_starboard(own, rng, i) if choice == "crossing" else (
                _template_target_forward_contact(own, rng, i) if choice == "head_on" else _template_target_overtaking_ahead(own, rng, i)
            )
        targets.append(_finalize_target(t, rng))

    return targets


def _inject_feature_flags(feats: Features, pat: PatternSpec) -> Features:
    """
    Post-process Features to trigger R9/R10 sub-clauses.
    IMPORTANT: do NOT override feats.visibility here (avoid Enum/string mismatch).
    """
    updates: Dict[str, Any] = {}

    # domain-specific
    if pat.domain == Domain.TSS:
        updates["area_type"] = "tss"
        updates["traffic_lane_relation"] = pat.traffic_lane_relation
    elif pat.domain == Domain.NARROW_CHANNEL:
        updates["area_type"] = "narrow_channel"

    # sub-clause toggles
    if pat.intends_cross_channel is not None:
        updates["intends_cross_channel"] = pat.intends_cross_channel
    if pat.intends_anchor is not None:
        updates["intends_anchor"] = pat.intends_anchor
    if pat.agreement_received_for_overtake is not None:
        updates["agreement_received_for_overtake"] = pat.agreement_received_for_overtake

    # in_sight is a bool, safe to inject
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
    for tgt in scene.targets:
        geom = compute_pairwise_geometry(
            tracks,
            ownship_id=scene.ownship.vessel_id,
            target_id=tgt.vessel_id,
            t_index=-1,
            domain=scene.domain.value,
            tss_lane_heading_deg=scene.tss_lane_heading_deg,
        )
        feats = build_features_for_target(scene, tgt, geom, operational_params_path=None)
        feats = _inject_feature_flags(feats, pat)
        feats_list.append(feats)

    res = resolve_multi(feats_list, strict=False, rules_path=rules_path)
    global_res = getattr(res, "global_resolution", res)

    kept_rules = getattr(global_res, "kept_rules", []) or []
    metrics_dict = _metrics_to_dict(getattr(global_res, "metrics", None))

    # checks
    ok = True
    if len(kept_rules) < cons.min_rules:
        ok = False

    # required rules
    for rid in cons.required_rule_ids:
        if rid not in kept_rules:
            ok = False
            break
    for pref in cons.required_rule_prefix:
        if not any(r.startswith(pref) for r in kept_rules):
            ok = False
            break

    # class presence
    if cons.required_target_classes:
        classes = {_norm_class(t.vessel_class) for t in scene.targets}
        for c in cons.required_target_classes:
            if _norm_class(c) not in classes:
                ok = False
                break

    # metrics constraints
    if int(metrics_dict.get("num_conflicts", 0)) < cons.min_conflicts:
        ok = False
    if int(metrics_dict.get("num_tensions", 0)) < cons.min_tensions:
        ok = False
    if float(metrics_dict.get("complexity_score", 0.0)) < cons.min_complexity_score:
        ok = False

    # refined conflict types
    if int(metrics_dict.get("num_conflicts_maneuver", 0)) < cons.min_conflicts_maneuver:
        ok = False
    if int(metrics_dict.get("num_conflicts_signals", 0)) < cons.min_conflicts_signals:
        ok = False
    if int(metrics_dict.get("num_conflicts_domain", 0)) < cons.min_conflicts_domain:
        ok = False

    debug = {
        "kept_rules": tuple(kept_rules),
        "metrics": metrics_dict,
        "scene": scene.model_dump(),
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
    """
    Search-based sampler: tries up to max_tries to find a scene satisfying constraints.
    Returns (SceneSpec, debug_dict with kept_rules/metrics).
    """
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

        scene = SceneSpec(
            seed=seed + attempt,
            domain=pattern.domain,
            visibility=pattern.visibility,  # scene-level string; feature_builder will coerce to Enum
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
        f"Last metrics: {last_debug.get('metrics')}, last kept_rules: {last_debug.get('kept_rules')}"
    )


def preset_patterns() -> Dict[str, PatternSpec]:
    """
    Pattern library. restricted_multi defaults to force_conflict=True to make early-stage data generation feasible.
    """
    return {
        "restricted_multi": PatternSpec(
            name="restricted_multi",
            domain=Domain.RESTRICTED,
            encounter="mixed",
            visibility="restricted",
            in_sight=False,
            force_conflict=True,  # key change: not random; template-driven
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


# Backward-compatible API expected by colreg_scenegen.__init__.py
def sample_scene(seed: int = 0, domain: Domain = Domain.OPEN, n_targets: int = 2) -> SceneSpec:
    """
    Legacy wrapper: returns a scene quickly (low constraints), used by earlier MVP scripts.
    """
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