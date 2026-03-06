from colreg_scenegen.sampler import sample_scene_advanced, preset_patterns, ConflictConstraints
from colreg_scenegen.sim_kinematics import simulate_scene
from colreg_scenegen.labeler import label_scene_multi

pat = preset_patterns()["restricted_multi"]

constraints = ConflictConstraints(
    min_rules=8,
    min_conflicts=1,
    min_conflicts_maneuver=1,
    min_complexity_score=2.0,
    required_rule_prefix=("COLREG_R19", "COLREG_R35"),
    max_tries=300,
)

scene, dbg = sample_scene_advanced(seed=42, pattern=pat, n_targets=3, constraints=constraints)
tracks = simulate_scene(scene)
out = label_scene_multi(scene, tracks, strict=False)

print("Sampled metrics:", dbg["metrics"])
print("Kept rules:", dbg["kept_rules"])
print("Global metrics:", out["global_resolution"]["metrics"])