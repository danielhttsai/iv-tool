"""Tests for the Negative Control & Proximal Causal Inference (NC) method."""
from __future__ import annotations

import nc_gen
import nc_core
import nc_assumptions


def test_proximal_recovers_truth_and_naive_is_biased():
    df = nc_gen.generate()
    r = nc_core.full_nc(df, n_boot=0)
    truth = r["true_tau"]
    # proximal P2SLS lands near the truth
    assert abs(r["proximal"] - truth) < 0.15
    # the naive estimate is clearly more biased (unmeasured confounding)
    assert abs(r["naive"] - truth) > abs(r["proximal"] - truth) + 0.5


def test_detection_signal_fires():
    # A→W should be ~0 with no confounding, and clearly nonzero with confounding
    r1 = nc_core.full_nc(nc_gen.generate(conf=0.0), n_boot=0)
    r2 = nc_core.full_nc(nc_gen.generate(conf=1.0), n_boot=0)
    assert abs(r1["detect"]) < 0.1                          # no confounding → no signal
    assert abs(r2["detect"]) > 0.5                          # confounding → strong signal
    assert abs(r2["detect_z"]) > 3                          # many SEs from 0


def test_interactive_grid_monotone():
    g = nc_core._GRID
    assert g["naive"][-1] > g["naive"][0] + 1.0             # naive drifts up with confounding
    assert g["detect"][-1] > g["detect"][0] + 0.8           # detection signal grows
    assert all(abs(p - 1.0) < 0.1 for p in g["proximal"])   # proximal holds the truth


def test_interactive_endpoint():
    lo = nc_core.nc_interactive(0.0)
    hi = nc_core.nc_interactive(1.5)
    assert abs(lo["proximal"] - lo["true_tau"]) < 0.1
    assert abs(hi["proximal"] - hi["true_tau"]) < 0.1
    assert hi["naive"] > lo["naive"] + 1.0
    assert hi["detect"] > lo["detect"] + 0.8


def test_dashboard_shape_and_statuses():
    df = nc_gen.generate()
    dash = nc_assumptions.run_dashboard(df)
    checks = dash["checks"]
    assert [c["id"] for c in checks] == ["C1", "C2", "C3", "C4", "C5"]
    for c in checks:
        assert c["status"] in ("green", "amber", "red", "info")
        assert c["headline"] and c["plain"]


def test_bilingual_interpretation():
    df = nc_gen.generate()
    zh = nc_core.full_nc(df, n_boot=0, lang="zh")
    en = nc_core.full_nc(df, n_boot=0, lang="en")
    assert zh["interpretation"] != en["interpretation"]


def test_calibration_makes_pvalue_more_honest():
    import nc_ml
    r = nc_ml.calibration_demo()
    # negative controls' false-positive rate is too high naively, ~5% after calibration
    assert r["type1_naive"] > 0.3
    assert r["type1_cal"] <= 0.15
    # the main estimate's calibrated p-value is larger (more honest) than the naive one
    assert r["p_cal"] > r["p_naive"]
