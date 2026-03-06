from pathlib import Path

from colreg_kernel.derive_features import (
    NM_M,
    derive_action_substantial,
    derive_impedes,
    derive_risk_of_collision,
    derive_tss_crossing_ok,
    load_operational_params,
)
from colreg_kernel.features import AreaType


def test_risk_boundary_inclusive():
    params = load_operational_params(Path(__file__).resolve().parents[1] / "rules" / "operational_params.yaml")
    # Base threshold is max(0.5 nm, 6L). For L=100m => 6L=600m < 926m => threshold=926m
    L = 100.0
    thr = 0.5 * NM_M

    assert derive_risk_of_collision(cpa_m=thr, tcpa_s=900, own_length_m=L, params=params) is True
    assert derive_risk_of_collision(cpa_m=thr + 0.1, tcpa_s=900, own_length_m=L, params=params) is False
    assert derive_risk_of_collision(cpa_m=thr, tcpa_s=901, own_length_m=L, params=params) is False


def test_tss_crossing_angle_tolerance():
    params = load_operational_params(Path(__file__).resolve().parents[1] / "rules" / "operational_params.yaml")
    assert derive_tss_crossing_ok(tss_crossing_angle_deg=70.0, params=params) is True
    assert derive_tss_crossing_ok(tss_crossing_angle_deg=110.0, params=params) is True
    assert derive_tss_crossing_ok(tss_crossing_angle_deg=69.9, params=params) is False
    assert derive_tss_crossing_ok(tss_crossing_angle_deg=110.1, params=params) is False


def test_action_substantial_boundary():
    params = load_operational_params(Path(__file__).resolve().parents[1] / "rules" / "operational_params.yaml")
    assert derive_action_substantial(delta_cog_deg=15.0, params=params) is True
    assert derive_action_substantial(delta_cog_deg=14.9, params=params) is False
    assert derive_action_substantial(speed_drop_ratio=0.2, params=params) is True
    assert derive_action_substantial(speed_drop_ratio=0.199, params=params) is False


def test_impede_domain_specific_thresholds():
    params = load_operational_params(Path(__file__).resolve().parents[1] / "rules" / "operational_params.yaml")
    L = 50.0
    # open: max(0.3nm=555.6m, 4L=200m) => 555.6m
    assert derive_impedes(cpa_m=0.3 * NM_M, tcpa_s=720, own_length_m=L, area_type=AreaType.OPEN, params=params) is True
    assert derive_impedes(cpa_m=0.3 * NM_M + 0.1, tcpa_s=720, own_length_m=L, area_type=AreaType.OPEN, params=params) is False
    # narrow_channel stricter distance: 0.4nm
    assert derive_impedes(cpa_m=0.4 * NM_M, tcpa_s=720, own_length_m=L, area_type=AreaType.NARROW_CHANNEL, params=params) is True
    assert derive_impedes(cpa_m=0.4 * NM_M + 0.1, tcpa_s=720, own_length_m=L, area_type=AreaType.NARROW_CHANNEL, params=params) is False
