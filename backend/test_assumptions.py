"""Tests for the assumption checks, focusing on the 反證法 (falsification)
instrumental-inequalities test ported from Homayra et al. 2024.

The valid synthetic demo (a real IV) must SURVIVE the falsification; a
deliberately broken instrument (one that violates the inequalities) must be
REFUTED (red).
"""
import numpy as np
import pandas as pd

import assumptions
from gen_data import generate, COVARIATES

Y, A, Z = "health_score_change", "vaccinated", "vaccine_reminder"


def _df():
    return generate()


def test_valid_instrument_survives_falsification():
    df = _df()
    res = assumptions.check_falsification_inequalities(df, Y, A, Z)
    assert res["id"] == "FALS"
    assert res["status"] in ("green", "amber")   # not refuted


def test_broken_instrument_is_refuted():
    # Construct a binary Y/A/Z where the instrumental inequality is violated.
    # Make Z perfectly determine (Y=1, A=0): P(1,0|1) = 1, and for Z=0 force
    # (Y=0, A=0): P(0,0|0)=1. Then for a=0: max_z P(0,0|z)+max_z P(1,0|z)=1+1=2>1.
    n = 2000
    z = np.r_[np.ones(n), np.zeros(n)].astype(int)
    a = np.zeros(2 * n, dtype=int)
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    df = pd.DataFrame({"Z": z, "A": a, "Y": y})
    res = assumptions.check_falsification_inequalities(df, "Y", "A", "Z")
    assert res["status"] == "red"


def test_continuous_outcome_is_binarized():
    df = _df()
    res = assumptions.check_falsification_inequalities(df, Y, A, Z)
    notes = " ".join(m["note"] for m in res["metrics"])
    assert "中位數" in notes   # continuous Y was split at its median


def test_non_binary_instrument_returns_info():
    df = _df().copy()
    df["Zc"] = np.linspace(0, 1, len(df))   # continuous "instrument"
    res = assumptions.check_falsification_inequalities(df, Y, A, "Zc")
    assert res["status"] == "info"


def test_check_all_includes_falsification():
    df = _df()
    out = assumptions.check_all(df, Y, A, Z, COVARIATES)
    ids = [c["id"] for c in out["checks"]]
    assert ids == ["A1", "A2", "FALS", "A3", "A4a", "A4b"]
    assert out["overall_status"] in ("green", "amber", "red", "info")


def test_a3_reports_bias_amplification():
    df = _df()
    res = assumptions.check_a3_independence(df, Z, COVARIATES, A)
    names = " ".join(str(m["name"]) for m in res["metrics"])
    assert "偏誤放大" in names
