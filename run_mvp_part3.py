from colreg_scenegen.sampler import sample_scene_advanced, preset_patterns, ConflictConstraints
from colreg_scenegen.sim_kinematics import simulate_scene
from colreg_scenegen.labeler import label_scene_multi

# 1) choose a conflict-friendly pattern
pat = preset_patterns()["restricted_multi"]

# 2) constraints: require maneuver conflicts
constraints = ConflictConstraints(
    min_rules=8,
    min_conflicts=1,
    min_conflicts_maneuver=1,
    min_tensions=0,
    min_complexity_score=2.0,
    required_rule_prefix=("COLREG_R19", "COLREG_R35"),
    max_tries=300,
)

# 3) sample a multi-target scene
scene, dbg = sample_scene_advanced(seed=42, pattern=pat, n_targets=3, constraints=constraints)

# 4) simulate
tracks = simulate_scene(scene)

# 5) label (global resolution)
out = label_scene_multi(scene, tracks, strict=False)

global_res = out["global_resolution"]
labels = global_res.get("labels", {})

maneuver_allowed = labels.get("maneuver_allowed", {})
maneuver_forbidden = labels.get("maneuver_forbidden", {})

print("Sampled metrics:", dbg["metrics"])
print("Kept rules:", dbg["kept_rules"])
print("Global metrics:", global_res.get("metrics", {}))

print("\n--- Global maneuver_forbidden ---")
print(maneuver_forbidden)

print("\n--- Global maneuver_allowed ---")
print(maneuver_allowed)

# helper checks
def _has_key(d: dict, k: str) -> bool:
    return k in d and d[k] is not None and (d[k] != {})

print("\n--- Checks ---")
print("FORBIDDEN contains TURN_PORT:", _has_key(maneuver_forbidden, "TURN_PORT"))
print("FORBIDDEN contains TURN_STARBOARD:", _has_key(maneuver_forbidden, "TURN_STARBOARD"))
print("ALLOWED contains TURN_PORT:", _has_key(maneuver_allowed, "TURN_PORT"))
print("ALLOWED contains TURN_STARBOARD:", _has_key(maneuver_allowed, "TURN_STARBOARD"))