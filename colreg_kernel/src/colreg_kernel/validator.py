from __future__ import annotations

"""Scenario sanity validation and signals output consistency checks.

These checks are designed for:
  - rejecting inconsistent gold cases
  - rejecting inconsistent scenegen outputs (or forcing resampling)
  - ensuring signals head outputs are legally consistent (no contradictory requirements)

They are *not* a full COLREG resolver; they simply enforce data integrity and basic domain guards.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from .actions import LightShapePrimitive, SoundPrimitive
from .features import Features, FishingMode, NavStatus, VesselClass, Visibility


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def validate_feature_consistency(features: Features) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # --- visibility vs in_sight
    if features.visibility == Visibility.RESTRICTED and features.in_sight is True:
        issues.append(
            ValidationIssue(
                code="VIS_IN_SIGHT_CONFLICT",
                message="visibility=restricted usually implies in_sight=False for benchmark; set in_sight=None/False or justify.",
            )
        )

    # --- fishing_mode consistency
    for who, v in (("ownship", features.ownship), ("target", features.target)):
        if v.fishing_mode is not None and v.vessel_class != VesselClass.FISHING:
            issues.append(
                ValidationIssue(
                    code="FISHING_MODE_CLASS_MISMATCH",
                    message=f"{who}.fishing_mode is set but {who}.vessel_class != fishing.",
                )
            )

        if v.tow_length_m is not None and v.is_towing is not True:
            issues.append(
                ValidationIssue(
                    code="TOW_LENGTH_WITHOUT_TOW",
                    message=f"{who}.tow_length_m is set but {who}.is_towing is not True.",
                )
            )

        if v.is_pushing is True and v.is_towing is True:
            # Could happen in real ops, but keep benchmark simple.
            issues.append(
                ValidationIssue(
                    code="PUSH_AND_TOW",
                    message=f"{who} is both pushing and towing; benchmark simplification expects one mode.",
                )
            )

        # --- nav_status consistency
        if v.nav_status is not None:
            if v.nav_status == NavStatus.AGROUND and v.vessel_class != VesselClass.AGROUND:
                # Allow class aground or nav_status aground, but prefer consistency.
                issues.append(
                    ValidationIssue(
                        code="AGROUND_STATUS_CLASS_MISMATCH",
                        message=f"{who}.nav_status=aground but {who}.vessel_class != aground.",
                    )
                )

        # --- sailing engine flag consistency
        if v.sailing_using_engine is True and v.vessel_class != VesselClass.SAILING:
            issues.append(
                ValidationIssue(
                    code="SAILING_ENGINE_CLASS_MISMATCH",
                    message=f"{who}.sailing_using_engine is set but {who}.vessel_class != sailing.",
                )
            )

        # --- Rule 12 helper consistency
        if v.vessel_class == VesselClass.SAILING:
            # sailing_tack and is_windward may be None; that's OK. But if set, require in_sight True (benchmark simplification).
            if (v.sailing_tack is not None or v.is_windward is not None) and features.in_sight is False:
                issues.append(
                    ValidationIssue(
                        code="SAILING_FEATURES_WITHOUT_INSIGHT",
                        message=f"{who} sailing_tack/is_windward set while in_sight=False; ensure consistent scenario definition.",
                    )
                )

    # --- intent flags should only be used when in_sight (Rule 34)
    if features.in_sight is False:
        if features.intent_turn is not None and features.intent_turn.value != "none":
            issues.append(
                ValidationIssue(
                    code="INTENT_TURN_WITHOUT_INSIGHT",
                    message="intent_turn set while in_sight=False; Rule 34 maneuvering signals apply to vessels in sight.",
                )
            )
        if features.intent_astern_propulsion is True:
            issues.append(
                ValidationIssue(
                    code="ASTERN_WITHOUT_INSIGHT",
                    message="intent_astern_propulsion set while in_sight=False; Rule 34 typically requires in_sight.",
                )
            )

    return issues


def validate_signals_outputs(
    *,
    features: Features,
    lights_required: Sequence[LightShapePrimitive] = (),
    lights_forbidden: Sequence[LightShapePrimitive] = (),
    sounds_required: Sequence[SoundPrimitive] = (),
    sounds_forbidden: Sequence[SoundPrimitive] = (),
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    req_l = set(lights_required)
    forb_l = set(lights_forbidden)
    req_s = set(sounds_required)
    forb_s = set(sounds_forbidden)

    # No primitive can be both required and forbidden.
    both_l = req_l & forb_l
    if both_l:
        issues.append(
            ValidationIssue(
                code="LIGHT_REQUIRED_FORBIDDEN",
                message=f"Lights/shapes both required and forbidden: {sorted([x.value for x in both_l])}",
            )
        )

    both_s = req_s & forb_s
    if both_s:
        issues.append(
            ValidationIssue(
                code="SOUND_REQUIRED_FORBIDDEN",
                message=f"Sounds both required and forbidden: {sorted([x.value for x in both_s])}",
            )
        )

    # Domain guard: maneuvering signals (Rule 34 short blasts) should not appear in restricted visibility.
    if features.visibility == Visibility.RESTRICTED:
        short_blasts = {
            SoundPrimitive.SOUND_1_SHORT,
            SoundPrimitive.SOUND_2_SHORT,
            SoundPrimitive.SOUND_3_SHORT,
        }
        if req_s & short_blasts:
            issues.append(
                ValidationIssue(
                    code="SHORT_BLAST_IN_RESTRICTED",
                    message="Rule 34 short blasts required while visibility=restricted; use Rule 35 patterns instead.",
                )
            )

    # Domain guard: restricted-visibility patterns should not appear in clear, in-sight maneuvering context.
    if features.visibility == Visibility.CLEAR and features.in_sight is True:
        restricted_patterns = {
            SoundPrimitive.SOUND_PROLONGED_EVERY_2MIN,
            SoundPrimitive.SOUND_PROLONGED_PLUS_2_SHORT,
            SoundPrimitive.SOUND_BELL_RAPID,
            SoundPrimitive.SOUND_BELL_AND_GONG,
        }
        if req_s & restricted_patterns:
            issues.append(
                ValidationIssue(
                    code="RESTRICTED_PATTERN_IN_CLEAR_INSIGHT",
                    message="Restricted-visibility sound pattern required while visibility=clear and in_sight=True.",
                )
            )

    return issues
