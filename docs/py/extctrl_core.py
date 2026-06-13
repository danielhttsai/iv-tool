"""External-control core — estimate a single-arm treatment effect against a
borrowed external control arm.

Story (the toolbox's "naive biased -> method recovers truth"): comparing the
single-arm trial's mean outcome with the external controls' mean outcome is biased
when the two groups have different covariate (X) distributions. Two adjustments
recover the true effect TAU:
- standardisation / g-formula: model the untreated outcome on the external controls
  and predict it at the TRIAL patients' covariates → a counterfactual untreated mean
  for the trial population;
- IPSW (inverse-probability-of-selection weighting): weight the external controls so
  their covariate distribution matches the trial's, then take the weighted mean.

Refs: ICH E10; Pocock (1976); Schmidli et al. (2014); Burcu et al. (2020); Jahanshahi
et al. (2021); Hatswell et al. (2016).
"""
import numpy as np
from extctrl_gen import generate, TAU, MU_TRIAL

try:
    from i18n import t
except Exception:  # pragma: no cover
    def t(lang, zh, en):
        return zh if lang == "zh" else en


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _naive(yt, ye):
    """Raw single-arm vs external mean difference (confounded by covariate imbalance)."""
    return float(yt.mean() - ye.mean())


def _standardize(xt, yt, xe, ye):
    """Fit untreated outcome on the external controls, predict at the trial's X."""
    Xe = np.column_stack([np.ones_like(xe), xe])
    b = _ols(ye, Xe)                                  # E[Y(0)|X] from external controls
    y0_hat = b[0] + b[1] * xt                         # counterfactual untreated for the trial
    return float(yt.mean() - y0_hat.mean())


def _ipsw(xt, yt, xe, ye):
    """Weight external controls to the trial's covariate distribution, then compare."""
    x = np.concatenate([xt, xe])
    s = np.concatenate([np.ones_like(xt), np.zeros_like(xe)])   # 1 = trial, 0 = external
    Xd = np.column_stack([np.ones_like(x), x, x * x])
    beta = np.zeros(Xd.shape[1])
    for _ in range(25):                                          # logistic S ~ X via IRLS
        p = _sig(Xd @ beta)
        W = p * (1 - p) + 1e-6
        z = Xd @ beta + (s - p) / W
        beta = np.linalg.lstsq(Xd * W[:, None], z * W, rcond=None)[0]
    ps_ext = _sig(np.column_stack([np.ones_like(xe), xe, xe * xe]) @ beta)
    w = ps_ext / np.clip(1 - ps_ext, 1e-3, 1 - 1e-3)            # odds of being in the trial
    y0_w = np.average(ye, weights=w)
    return float(yt.mean() - y0_w)


def full_extctrl(mu_ext=None, lang="zh", seed=9):
    d = generate(mu_ext=(-0.5 if mu_ext is None else float(mu_ext)), seed=seed)
    xt, yt, xe, ye = d["trial_X"], d["trial_Y"], d["ext_X"], d["ext_Y"]
    truth = d["true_tau"]
    naive = _naive(yt, ye)
    std = _standardize(xt, yt, xe, ye)
    ipsw = _ipsw(xt, yt, xe, ye)
    gap = naive - truth
    reading = t(
        lang,
        f"真實治療效果＝<b>{truth:.2f}</b>。把單臂試驗直接和外部對照相比（<b>{naive:.2f}</b>），"
        f"因兩組共變項分布不同而偏了 {gap:+.2f}；<b>標準化</b>＝{std:.2f}、<b>選樣反機率加權（IPSW）</b>＝{ipsw:.2f}，"
        f"兩者都把外部對照校到試驗族群、救回真值。",
        f"The true treatment effect = <b>{truth:.2f}</b>. Comparing the single-arm trial directly with the external "
        f"controls (<b>{naive:.2f}</b>) is off by {gap:+.2f} because the two groups have different covariate "
        f"distributions; <b>standardisation</b> = {std:.2f} and <b>inverse-probability-of-selection weighting (IPSW)</b> "
        f"= {ipsw:.2f} both align the external controls to the trial population and recover the truth.",
    )
    interp = t(
        lang,
        "外部對照靠『把外部對照的共變項分布，校正／加權成試驗族群的分布』。前提：扣掉已測共變項後，外部對照與試驗對照可交換（無未測混淆、無時代／量測差異），且共變項有重疊（正性）。",
        "External-control analysis aligns the external controls' covariate distribution to the trial's. It assumes that, "
        "given the measured covariates, the external controls are exchangeable with the trial's would-be controls (no "
        "unmeasured confounding, no era/measurement drift), with covariate overlap (positivity).",
    )
    return {
        "mu_trial": d["mu_trial"], "mu_ext": d["mu_ext"],
        "truth": truth, "naive": naive, "standardize": std, "ipsw": ipsw,
        "reading": reading, "interpretation": interp,
    }


def extctrl_interactive(mu_ext, lang="zh"):
    return full_extctrl(mu_ext=mu_ext, lang=lang)
