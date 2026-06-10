"""Tests for the Propensity Score (PS) method.

Story to lock in: the crude vaccinated-vs-unvaccinated difference is biased upward by
confounding by indication; PS methods (regression adjustment, IPTW, overlap weighting,
PS matching) recover the truth; balance (SMD) improves dramatically after weighting; the
interactive grid is monotone in confounding strength; the ML-PS ⑤ demo beats the linear PS.
"""
from __future__ import annotations

import numpy as np
import pytest

import ps_gen
import ps_core
import ps_assumptions


TRUTH = ps_gen.TRUE_ATE


def _demo():
    return ps_gen.generate()


def test_crude_is_biased_high():
    df = _demo()
    r = ps_core.full_ps(df, n_boot=0)
    # confounding by indication inflates the crude estimate well above the truth
    assert r["crude"] > TRUTH + 0.4
    assert abs(r["crude"] - TRUTH) > 0.4


@pytest.mark.parametrize("key", ["iptw", "overlap", "att", "adjust"])
def test_ps_estimators_recover_truth(key):
    df = _demo()
    r = ps_core.full_ps(df, n_boot=0)
    assert abs(r[key] - TRUTH) < 0.2, f"{key}={r[key]} not near truth {TRUTH}"


def test_design_beats_naive():
    df = _demo()
    r = ps_core.full_ps(df, n_boot=0)
    assert abs(r["iptw"] - TRUTH) + 0.3 < abs(r["crude"] - TRUTH)


def test_balance_improves_after_weighting():
    df = _demo()
    r = ps_core.full_ps(df, n_boot=0)
    assert r["smd_before"] > 0.3            # strong imbalance before
    assert r["smd_after"] < 0.1             # balanced after IPTW
    assert r["smd_after"] < r["smd_before"]


def test_bootstrap_ci_covers_truth():
    df = _demo()
    r = ps_core.full_ps(df, n_boot=200)
    lo, hi = r["ci_iptw"]
    assert lo is not None and hi is not None
    assert lo < TRUTH < hi


def test_interactive_grid_monotone():
    g = ps_core._GRID
    crude = g["crude"]
    # crude estimate drifts ever further up as confounding strengthens (monotone non-decreasing)
    assert all(crude[i] <= crude[i + 1] + 1e-6 for i in range(len(crude) - 1))
    assert crude[-1] > crude[0] + 1.0
    # IPTW stays put on the truth across the whole grid
    for v in g["iptw"]:
        assert abs(v - g["true_ate"]) < 0.1


def test_interactive_reading_bilingual():
    zh = ps_core.ps_interactive(1.0, lang="zh")
    en = ps_core.ps_interactive(1.0, lang="en")
    assert zh["reading"] != en["reading"]
    assert zh["crude"] == en["crude"]


def test_dashboard_shape():
    df = _demo()
    d = ps_assumptions.run_dashboard(df, lang="zh")
    ids = [c["id"] for c in d["checks"]]
    assert ids == ["C1", "C2", "C3", "C4", "C5"]
    for c in d["checks"]:
        assert c["status"] in ("green", "amber", "red", "info")
        assert c["title"] and c["headline"] and c["plain"]
    # good overlap demo → positivity green; balance green
    by = {c["id"]: c for c in d["checks"]}
    assert by["C2"]["status"] == "green"
    assert by["C3"]["status"] == "green"


def test_interpretation_bilingual():
    df = _demo()
    zh = ps_core.full_ps(df, n_boot=0, lang="zh")["interpretation"]
    en = ps_core.full_ps(df, n_boot=0, lang="en")["interpretation"]
    assert zh != en and len(zh) > 40 and len(en) > 40


def test_ml_ps_beats_linear():
    pytest.importorskip("sklearn")
    out = ps_core_ml_demo()
    # non-linear PS surface: ML PS is closer to the truth than the main-effects logistic PS
    assert abs(out["ate_gb"] - out["ate_true"]) < abs(out["ate_lin"] - out["ate_true"])
    # and better balanced
    assert out["smd_gb"] < out["smd_lin"]


def ps_core_ml_demo():
    import ps_ml
    return ps_ml.ml_ps_demo()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
