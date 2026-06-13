"""Tests for the external-control core (naive biased, adjustments recover TAU)."""
import extctrl_core as ec
from extctrl_gen import TAU


def test_adjustments_recover_truth():
    out = ec.full_extctrl(lang="en")
    assert abs(out["standardize"] - TAU) < 0.25
    assert abs(out["ipsw"] - TAU) < 0.25


def test_naive_is_biased_under_imbalance():
    out = ec.full_extctrl(mu_ext=-1.0, lang="en")
    naive_err = abs(out["naive"] - TAU)
    std_err = abs(out["standardize"] - TAU)
    assert naive_err > std_err + 0.3


def test_no_imbalance_collapses_to_naive():
    """When the external controls share the trial's covariate mean, naive ≈ truth."""
    out = ec.full_extctrl(mu_ext=0.5, lang="en")   # mu_ext == MU_TRIAL
    assert abs(out["naive"] - TAU) < 0.25


def test_bias_monotone_in_shift():
    small = abs(ec.full_extctrl(mu_ext=0.2, lang="en")["naive"] - TAU)
    big = abs(ec.full_extctrl(mu_ext=-1.2, lang="en")["naive"] - TAU)
    assert big > small


def test_bilingual_reading_differs():
    zh = ec.full_extctrl(lang="zh")["reading"]
    en = ec.full_extctrl(lang="en")["reading"]
    assert zh != en and len(zh) > 0 and len(en) > 0


def test_dashboard_shape():
    import extctrl_assumptions as ea
    out = ea.run_dashboard(lang="en")
    assert out["overall_status"] in ("red", "amber", "info", "green")
    assert len(out["checks"]) == 5


def test_interactive():
    out = ec.extctrl_interactive(-0.5, lang="en")
    for k in ("truth", "naive", "standardize", "ipsw"):
        assert k in out
