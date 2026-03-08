from __future__ import annotations

import argparse
from pathlib import Path

# make rendering stable on headless servers
import matplotlib
matplotlib.use("Agg")

from colreg_scenegen.sampler import sample_scene_advanced, preset_patterns, ConflictConstraints
from colreg_scenegen.sim_kinematics import simulate_scene
from colreg_scenegen.labeler import label_scene_multi
from colreg_scenegen.render_topdown import render_topdown_grid, TopDownRenderConfig
from colreg_scenegen.render_radar import render_radar_ppi_grid, RadarRenderConfig
from colreg_scenegen.packager import DatasetLayout, save_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="dataset_out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pattern", type=str, default="restricted_multi")
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tries", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pats = preset_patterns()
    if args.pattern not in pats:
        raise ValueError(f"Unknown pattern: {args.pattern}. Available: {list(pats.keys())}")
    pat = pats[args.pattern]

    # constraints tuned for early-stage conflict generation
    if args.pattern == "restricted_multi":
        cons = ConflictConstraints(
            min_rules=8,
            min_conflicts=1,
            min_conflicts_maneuver=1,
            min_tensions=0,
            min_complexity_score=2.0,
            required_rule_prefix=("COLREG_R19", "COLREG_R35"),
            max_tries=args.max_tries,
        )
    elif args.pattern == "responsibility_mix":
        cons = ConflictConstraints(
            min_rules=6,
            min_conflicts=0,
            min_conflicts_maneuver=0,
            min_tensions=0,
            min_complexity_score=0.0,
            required_rule_prefix=("COLREG_R18",),
            required_target_classes=("nuc", "sailing"),
            max_tries=args.max_tries,
        )
    elif args.pattern == "tss_crossing":
        cons = ConflictConstraints(
            min_rules=6,
            min_conflicts=0,
            min_conflicts_maneuver=0,
            min_tensions=0,
            min_complexity_score=0.0,
            required_rule_prefix=("COLREG_R10",),
            max_tries=args.max_tries,
        )
    elif args.pattern == "channel_crossing_impede":
        cons = ConflictConstraints(
            min_rules=6,
            min_conflicts=0,
            min_conflicts_maneuver=0,
            min_tensions=0,
            min_complexity_score=0.0,
            required_rule_prefix=("COLREG_R9",),
            max_tries=args.max_tries,
        )
    else:
        cons = ConflictConstraints(
            min_rules=4,
            min_conflicts=0,
            min_conflicts_maneuver=0,
            min_tensions=0,
            min_complexity_score=0.0,
            max_tries=args.max_tries,
        )

    layout = DatasetLayout(root=out_dir)

    top_cfg = TopDownRenderConfig(display_scale=5.0)
    rad_cfg = RadarRenderConfig(seed=args.seed)

    made = 0
    seed = args.seed

    # clear manifest for clean run
    manifest = out_dir / layout.manifest_name
    if manifest.exists():
        manifest.unlink()

    while made < args.n:
        scene, dbg = sample_scene_advanced(seed=seed, pattern=pat, n_targets=args.targets, constraints=cons)
        tracks = simulate_scene(scene)
        label = label_scene_multi(scene, tracks, strict=False)

        # --- QA gate (MVP) ---
        # 1) risk gate: at least one target has risk_of_collision True
        risk_ok = False
        for pt in label.get("per_target", []):
            feats = pt.get("features", {})
            if feats.get("risk_of_collision") is True:
                risk_ok = True
                break
        if not risk_ok:
            seed += 1
            continue

        # 2) topdown visibility gate: all targets within topdown range at last frame
        range_m = top_cfg.range_m
        own_last = tracks[scene.ownship.vessel_id][-1]
        ok_in_view = True
        for t in scene.targets:
            st = tracks[t.vessel_id][-1]
            dx = st.x_m - own_last.x_m
            dy = st.y_m - own_last.y_m
            if abs(dx) > range_m or abs(dy) > range_m:
                ok_in_view = False
                break
        if not ok_in_view:
            seed += 1
            continue

        # 3) ppi gate: all targets within rmax and not extremely close
        rmax = rad_cfg.rmax_m
        ok_ppi = True
        for t in scene.targets:
            st = tracks[t.vessel_id][-1]
            dx = st.x_m - own_last.x_m
            dy = st.y_m - own_last.y_m
            r = (dx * dx + dy * dy) ** 0.5
            if r > rmax or r < 100.0:
                ok_ppi = False
                break
        if not ok_ppi:
            seed += 1
            continue
        # --- end QA gate ---

        # render
        top_img = render_topdown_grid(tracks, ownship_id=scene.ownship.vessel_id, cfg=top_cfg, scene=scene)
        rad_img = render_radar_ppi_grid(tracks, ownship_id=scene.ownship.vessel_id, cfg=rad_cfg)

        sample_id = f"{args.pattern}_{seed:06d}"

        difficulty = float(dbg.get("metrics", {}).get("complexity_score", 0.0))

        save_sample(
            layout=layout,
            sample_id=sample_id,
            topdown_rgb=top_img,
            radar_rgb=rad_img,
            label_json=label,
            spec_json=scene.model_dump(),
            extra_meta={
                "label_snapshot_s": 30.0,
                "pattern": args.pattern,
                "difficulty": difficulty,
                "sampled_metrics": dbg.get("metrics", {}),
                "kept_rules": list(dbg.get("kept_rules", ())),
                "render": {
                    "topdown": {
                        "size_px": top_cfg.size_px,
                        "dpi": top_cfg.dpi,
                        "frame_offsets": list(top_cfg.frame_offsets),
                        "range_m": top_cfg.range_m,
                        "trail_len_s": top_cfg.trail_len_s,
                    },
                    "radar": {
                        "size_px": rad_cfg.size_px,
                        "dpi": rad_cfg.dpi,
                        "frame_offsets": list(rad_cfg.frame_offsets),
                        "rmax_m": rad_cfg.rmax_m,
                        "range_sigma_m": rad_cfg.range_sigma_m,
                        "bearing_sigma_deg": rad_cfg.bearing_sigma_deg,
                        "clutter_points": rad_cfg.clutter_points,
                        "seed": rad_cfg.seed,
                        "grid_n": rad_cfg.grid_n,
                        "target_sigma_m": rad_cfg.target_sigma_m,
                        "clutter_sigma_m": rad_cfg.clutter_sigma_m,
                        "target_amp": rad_cfg.target_amp,
                        "clutter_amp": rad_cfg.clutter_amp,
                    },
                },
            },
        )

        print(f"[{made+1}/{args.n}] saved {sample_id} metrics={dbg.get('metrics')}")
        made += 1
        seed += 1


if __name__ == "__main__":
    main()