"""Negative Control & Proximal Causal Inference (NC) core — pure numpy.

白話：要估「疫苗 A → 結果 Y」的因果效應，但有個<b>沒測到的混淆 U</b>（健康／就醫傾向），天真比較會被它偏掉。
我們用兩個<b>陰性對照</b>：NCO（W，疫苗不可能影響、但與 U 相關）、NCE（Z，不可能影響 Y、但與 U 相關）。

  ① 偵測（Lipsitch 2010）：天真估 <b>A → W</b> 本應為 0；若 ≠0，就是<b>殘留／未測混淆</b>的警訊。
  ② ③ 校正（雙陰性對照／近端因果，P2SLS — Miao et al. 2018；regression-based PCI 2025）：
       第一階段： W ~ A + Z + X        → Ŵ        （用 NCE Z 把 W 中「U 的部分」抓出來）
       第二階段： Y ~ A + Ŵ + X        → A 的係數＝因果效應（在有 U 下仍還原真值）
     對照：天真 Y ~ A + X 被 U 偏掉。

NOTE — faithful-but-simplified teaching reconstruction (Lipsitch, Tchetgen Tchetgen & Cohen
2010, Epidemiology; Miao, Geng & Tchetgen Tchetgen 2018, Biometrika; Tchetgen Tchetgen et al.
proximal causal learning; regression-based PCI, Epidemiology 2025). Synthetic data only.
"""
from __future__ import annotations

import numpy as np

from i18n import t


def _ols(cols, y):
    """OLS with intercept. cols = list of 1-D arrays. Returns (beta, se) incl. intercept."""
    X = np.column_stack([np.ones(len(y))] + list(cols))
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX + 1e-10 * np.eye(X.shape[1]))
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(XtXi) * sigma2)
    return beta, se


def _p2sls(A, Z, X, W, Y):
    """Proximal two-stage least squares. Stage 1: W ~ A+Z+X. Stage 2: Y ~ A+Ŵ+X.
    Returns the coefficient on A in stage 2 (the causal effect)."""
    b1, _ = _ols([A, Z, X], W)
    What = b1[0] + b1[1] * A + b1[2] * Z + b1[3] * X
    b2, se2 = _ols([A, What, X], Y)
    return float(b2[1]), What


def full_nc(df, treat="A", outcome="Y", cov="X", nco="W", nce="Z",
            true_tau=None, n_boot=300, lang="zh"):
    import nc_gen
    if true_tau is None:
        true_tau = nc_gen.TRUE_TAU
    A = np.asarray(df[treat], dtype=float); Y = np.asarray(df[outcome], dtype=float)
    X = np.asarray(df[cov], dtype=float); W = np.asarray(df[nco], dtype=float)
    Z = np.asarray(df[nce], dtype=float)

    # naive (confounded by U): Y ~ A + X
    bn, sen = _ols([A, X], Y)
    naive = float(bn[1]); ci_naive = [naive - 1.96 * sen[1], naive + 1.96 * sen[1]]

    # ① detection: A → W (should be ~0; nonzero flags unmeasured confounding)
    bd, sed = _ols([A, X], W)
    detect = float(bd[1]); ci_detect = [detect - 1.96 * sed[1], detect + 1.96 * sed[1]]
    detect_z = float(detect / sed[1]) if sed[1] > 0 else 0.0

    # proximal P2SLS (double negative control): recovers the truth under U
    prox, _ = _p2sls(A, Z, X, W, Y)
    # bootstrap CI for the proximal estimate (light: lstsq only)
    lo = hi = None
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(20240608); reps = []
        nN = len(Y)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, nN, nN)
            try:
                p, _ = _p2sls(A[idx], Z[idx], X[idx], W[idx], Y[idx]); reps.append(p)
            except Exception:
                pass
        if reps:
            lo = float(np.percentile(reps, 2.5)); hi = float(np.percentile(reps, 97.5))
    ci_prox = [lo, hi]

    interp = t(
        lang,
        f"天真地估『A → Y』得 ≈ <b>{naive:.2f}</b>（95% CI {ci_naive[0]:.2f}～{ci_naive[1]:.2f}），被未測混淆 U 偏掉"
        f"（真值 {true_tau:.2f}）。<b>偵測</b>：疫苗不可能影響陰性對照結果 W，但天真估『A → W』≈ <b>{detect:+.2f}</b>"
        f"（離 0 達 {abs(detect_z):.1f} 個標準誤）——這就是<b>未測混淆的警訊</b>。<b>校正</b>：用陰性對照暴露 Z 與結果 W 當 U 的代理，"
        f"做<b>近端因果 P2SLS</b>（W~A+Z+X→Ŵ；Y~A+Ŵ+X），得 ≈ <b>{prox:.2f}</b>"
        + (f"（95% 自助 CI {lo:.2f}～{hi:.2f}）" if lo is not None else "") + f"——還原真值 {true_tau:.2f}。",
        f"Naively estimating 'A → Y' gives ≈ <b>{naive:.2f}</b> (95% CI {ci_naive[0]:.2f}–{ci_naive[1]:.2f}), biased by the "
        f"unmeasured confounder U (truth {true_tau:.2f}). <b>Detection</b>: the vaccine cannot affect the negative-control "
        f"outcome W, yet the naive 'A → W' ≈ <b>{detect:+.2f}</b> ({abs(detect_z):.1f} SEs from 0) — the <b>signal of unmeasured "
        f"confounding</b>. <b>Correction</b>: using the negative-control exposure Z and outcome W as proxies for U, "
        f"<b>proximal P2SLS</b> (W~A+Z+X→Ŵ; Y~A+Ŵ+X) gives ≈ <b>{prox:.2f}</b>"
        + (f" (95% bootstrap CI {lo:.2f}–{hi:.2f})" if lo is not None else "") + f" — recovering the truth {true_tau:.2f}.",
    )
    return {
        "naive": naive, "ci_naive": ci_naive,
        "detect": detect, "ci_detect": ci_detect, "detect_z": detect_z,
        "proximal": prox, "ci_proximal": ci_prox, "true_tau": float(true_tau),
        "n": int(len(Y)),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# ② interactive — unmeasured-confounding strength knob. As U drives A and Y more
# strongly, the naive estimate AND the A→W detection signal grow, while proximal
# P2SLS stays on the truth. Offline-precomputed grid (via _recompute_grid()).
# ---------------------------------------------------------------------------
_CONF_X = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def _recompute_grid(n=40000):
    import nc_gen
    out = {"conf": _CONF_X, "naive": [], "detect": [], "proximal": [], "true_tau": nc_gen.TRUE_TAU}
    for c in out["conf"]:
        df = nc_gen.generate(seed=101, n=n, conf=c)
        r = full_nc(df, n_boot=0)
        out["naive"].append(round(r["naive"], 3))
        out["detect"].append(round(r["detect"], 3))
        out["proximal"].append(round(r["proximal"], 3))
    return out


# precomputed grid via _recompute_grid(): naive drifts up with confounding; the A→W
# detection signal grows in lockstep; proximal P2SLS holds the truth.
_GRID = {
    "conf": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "naive": [1.004, 1.112, 1.367, 1.736, 2.194, 2.699, 3.233],
    "detect": [0.038, 0.373, 0.664, 0.908, 1.105, 1.249, 1.366],
    "proximal": [1.004, 1.010, 1.004, 0.989, 0.982, 0.990, 0.988],
    "true_tau": 1.0,
}


def nc_interactive(conf=1.0, lang="zh"):
    g = _GRID
    xc = float(np.clip(conf, g["conf"][0], g["conf"][-1]))
    naive = float(np.interp(xc, g["conf"], g["naive"]))
    detect = float(np.interp(xc, g["conf"], g["detect"]))
    prox = float(np.interp(xc, g["conf"], g["proximal"]))
    reading = t(
        lang,
        f"未測混淆強度 {xc:.2f}：天真『A → Y』≈ {naive:.2f} 被 U 越推越偏；同時陰性對照『A → W』≈ {detect:+.2f}"
        f"（本應為 0）也跟著放大——這就是<b>偵測訊號</b>，越偏代表混淆越強。<b>近端因果 P2SLS ≈ {prox:.2f}</b>，"
        f"不管混淆多強都穩在真值 {g['true_tau']:.1f}。偵測（A→W≠0）與校正（P2SLS）用的是同一對陰性對照。",
        f"Unmeasured-confounding strength {xc:.2f}: the naive 'A → Y' ≈ {naive:.2f} is pushed ever further off by U; in lockstep "
        f"the negative control 'A → W' ≈ {detect:+.2f} (which should be 0) grows too — that is the <b>detection signal</b>, larger "
        f"= stronger confounding. <b>Proximal P2SLS ≈ {prox:.2f}</b> stays on the truth {g['true_tau']:.1f} however strong the "
        f"confounding. Detection (A→W≠0) and correction (P2SLS) use the same pair of negative controls.",
    )
    return {"conf": xc, "naive": naive, "detect": detect, "proximal": prox,
            "true_tau": g["true_tau"], "grid": g, "reading": reading}
