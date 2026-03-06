from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path

from colreg_kernel.aggregate import aggregate
from colreg_reasoner.multiagent import resolve_multi

from .spec import SceneSpec
from .sim_kinematics import VesselState
from .geometry import compute_pairwise_geometry
from .feature_builder import build_features_for_target


def _default_rules_path() -> str:
    """
    Robustly locate Part1 rule YAML via installed package path.
    Works for editable installs:
      .../colreg_kernel/src/colreg_kernel/__init__.py
    -> repo root at parents[2] = .../colreg_kernel
    -> rules/colregs_atomic.yaml
    """
    import colreg_kernel  # type: ignore

    pkg_file = Path(colreg_kernel.__file__).resolve()
    repo_root = pkg_file.parents[2]  # .../colreg_kernel
    rules_path = repo_root / "rules" / "colregs_atomic.yaml"
    if not rules_path.exists():
        raise FileNotFoundError(f"Cannot locate rules YAML at: {rules_path}")
    return str(rules_path)


def label_scene_multi(
    scene: SceneSpec,
    tracks: Dict[str, List[VesselState]],
    operational_params_path: Optional[str] = None,
    strict: bool = False,
    rules_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a JSON-serializable dict:
      - per_target: list of {target_id, features, triggered_rules}
      - global_resolution: labels/explanation/metrics/graph
    """
    rp = rules_path or _default_rules_path()

    feats_list = []
    per_target = []

    for tgt in scene.targets:
        geom = compute_pairwise_geometry(
            tracks,
            ownship_id=scene.ownship.vessel_id,
            target_id=tgt.vessel_id,
            t_index=-1,
            domain=scene.domain.value,
            tss_lane_heading_deg=scene.tss_lane_heading_deg,
        )
        feats = build_features_for_target(scene, tgt, geom, operational_params_path=operational_params_path)
        feats_list.append(feats)

        # IMPORTANT: pass rules_path explicitly (do not rely on CWD)
        agg = aggregate(feats, strict=False, rules_path=rp)
        per_target.append(
            {
                "target_id": tgt.vessel_id,
                "features": feats.model_dump(),
                "triggered_rules": [t.rule_id for t in agg.triggered_rules],
            }
        )

    # Global resolution across targets.
    # Pass rules_path explicitly so Part2 -> Part1 aggregate will not fall back to CWD.
    multi = resolve_multi(feats_list, strict=strict, rules_path=rp)

    def _dump(obj: Any) -> Any:
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: _dump(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        return obj

    return {
        "scene": scene.model_dump(),
        "per_target": per_target,
        "global_resolution": _dump(getattr(multi, "global_resolution", multi)),
        "rules_path": rp,
    }