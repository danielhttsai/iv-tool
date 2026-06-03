"""Tests for the Prior Event Rate Ratio core and assumptions."""
import numpy as np

import perr_gen
import perr_core
import perr_assumptions


def test_recovers_true_rr_and_naive_is_biased():
    df = perr_gen.generate()
    out = perr_core.full_perr(df)
    # PERR recovers the true protective RR (0.70) far better than the naive post ratio
    assert abs(out["perr"] - perr_gen.TRUE_RR) < 0.15
    assert out["naive_rr"] > out["perr"] + 0.15          # naive hides the benefit
    assert out["rr_prior"] > 1.1                          # confounding fingerprint
    assert out["ci"][0] < out["perr"] < out["ci"][1]


def test_time_varying_confounding_biases_perr():
    clean = perr_core.full_perr(perr_gen.generate(drift=0.0))
    drifted = perr_core.full_perr(perr_gen.generate(drift=0.8))
    # a strong drift pushes PERR away from the truth
    assert abs(drifted["perr"] - perr_gen.TRUE_RR) > abs(clean["perr"] - perr_gen.TRUE_RR)


def test_scale_demo():
    s = perr_core.scale_demo(seed=7)
    # multiplicative world: PERR hits the true RR
    assert abs(s["multiplicative"]["perr"] - s["multiplicative"]["true_rr"]) < 0.15
    # additive world: PERD hits the true rate difference
    assert abs(s["additive"]["perd"] - s["additive"]["true_diff"]) < 0.02


def test_dashboard_shape_and_statuses():
    df = perr_gen.generate()
    dash = perr_assumptions.run_dashboard(df)
    ids = [c["id"] for c in dash["checks"]]
    assert ids == ["P1", "P2", "P3", "P4", "P5"]
    for c in dash["checks"]:
        assert c["status"] in ("green", "amber", "red", "info")
        assert c["title"] and c["headline"] and c["plain"] and c["term"]
    assert dash["checks"][0]["status"] == "info"          # P1 untestable
    assert dash["checks"][4]["status"] == "green"         # P5 enough prior events


def test_bilingual_interpretation():
    df = perr_gen.generate()
    zh = perr_core.full_perr(df, lang="zh")
    en = perr_core.full_perr(df, lang="en")
    assert zh["interpretation"] != en["interpretation"]
    assert "PERR" in en["interpretation"]
