"""Tests for the Test-Negative Design (TND).

Story: a naive case-control with population controls is biased toward the null by care-seeking;
the TND (test-negative controls) recovers the true VE. Interactive grid monotone in care-seeking;
dashboard shape; bilingual; causal-TND ML (IPW) beats the unadjusted/logistic under a non-linear
measured confounder.
"""
from __future__ import annotations

import numpy as np
import pytest

import tnd_gen
import tnd_core
import tnd_assumptions


TRUTH = tnd_gen.VE_TRUE


def _demo():
    return tnd_gen.generate()


def test_tnd_recovers_ve():
    r = tnd_core.full_tnd(_demo())
    assert abs(r["ve_tnd"] - TRUTH) < 0.06, f"TND VE {r['ve_tnd']} not near truth {TRUTH}"


def test_naive_is_biased_toward_null():
    r = tnd_core.full_tnd(_demo())
    # care-seeking confounding pulls the naive VE well below the truth
    assert r["ve_naive"] < TRUTH - 0.15
    assert abs(r["ve_tnd"] - TRUTH) + 0.1 < abs(r["ve_naive"] - TRUTH)


def test_ci_covers_truth():
    r = tnd_core.full_tnd(_demo())
    lo, hi = r["ci_tnd"]
    assert lo < TRUTH < hi


def test_interactive_grid_monotone():
    g = tnd_core._GRID
    # naive VE sinks (more biased) as care-seeking strengthens
    assert all(g["naive"][i] >= g["naive"][i + 1] - 1e-6 for i in range(len(g["naive"]) - 1))
    assert g["naive"][0] - g["naive"][-1] > 0.2
    # TND holds at the truth across the grid
    for v in g["tnd"]:
        assert abs(v - g["true"]) < 0.04


def test_grid_matches_recompute():
    g = tnd_core._recompute_grid(n=120000)
    for k in ("naive", "tnd"):
        for a, b in zip(g[k], tnd_core._GRID[k]):
            assert abs(a - b) < 0.05, f"grid drift in {k}: {a} vs {b}"


def test_interactive_reading_bilingual():
    zh = tnd_core.tnd_interactive(1.0, lang="zh")
    en = tnd_core.tnd_interactive(1.0, lang="en")
    assert zh["reading"] != en["reading"]
    assert zh["tnd"] == en["tnd"]


def test_dashboard_shape():
    d = tnd_assumptions.run_dashboard(_demo(), lang="zh")
    ids = [c["id"] for c in d["checks"]]
    assert ids == ["C1", "C2", "C3", "C4", "C5"]
    for c in d["checks"]:
        assert c["status"] in ("green", "amber", "red", "info")
        assert c["title"] and c["headline"] and c["plain"]
    by = {c["id"]: c for c in d["checks"]}
    assert by["C5"]["status"] in ("green", "amber")     # demo has enough cases/controls


def test_interpretation_bilingual():
    zh = tnd_core.full_tnd(_demo(), lang="zh")["interpretation"]
    en = tnd_core.full_tnd(_demo(), lang="en")["interpretation"]
    assert zh != en and len(zh) > 40 and len(en) > 40


def test_causal_tnd_ml_beats_unadjusted():
    pytest.importorskip("sklearn")
    import tnd_ml
    o = tnd_ml.ml_tnd_demo()
    # ML-IPW is closer to the truth than the unadjusted TND and than logistic IPW
    assert abs(o["ve_gb"] - o["ve_true"]) < abs(o["ve_crude"] - o["ve_true"])
    assert abs(o["ve_gb"] - o["ve_true"]) < abs(o["ve_lin"] - o["ve_true"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
