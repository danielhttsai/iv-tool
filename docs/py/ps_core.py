"""Propensity Score (PS) core — pure numpy.

白話：要估「疫苗 A → 結果 Y」的平均因果效應（ATE），但<b>適應症混淆</b>讓病重者（X 大）較易接種、也較易出事，
直接比較會偏。傾向分數 PS＝P(A=1|X)。把 X 平衡掉，就能還原 ATE：

  ・<b>粗估</b>：E[Y|A=1] − E[Y|A=0]（被 X 偏）。
  ・<b>迴歸校正</b>：Y ~ A + X 的 A 係數（靠結果模型）。
  ・<b>PS 配對</b>：每個接種者配一個 PS 最近的未接種者 → ATT。
  ・<b>IPTW 反機率加權</b>：權重 w = A/PS + (1−A)/(1−PS)，加權後比較 → ATE（Rosenbaum & Rubin 1983）。
  ・<b>重疊權重 ATO</b>：w = A·(1−PS) + (1−A)·PS，給「兩組都可能出現」的人最大權重，最穩定（Li, Morgan & Zaslavsky 2018）。
  平衡用<b>標準化差異（SMD）</b>檢查：加權前大、加權後接近 0。

NOTE — faithful-but-simplified teaching reconstruction (Rosenbaum & Rubin 1983; Austin 2011;
Brookhart et al. 2006; Li, Morgan & Zaslavsky 2018; Desai & Franklin 2019). Synthetic data only.
"""
from __future__ import annotations

import numpy as np

from i18n import t


def _ols(cols, y):
    X = np.column_stack([np.ones(len(y))] + list(cols))
    XtXi = np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1]))
    beta = XtXi @ (X.T @ y)
    return beta


def _logit_fit(X, y, iters=25):
    """Logistic regression by Newton–Raphson. X includes the intercept column."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p) + 1e-9
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _ps(A, X):
    """Estimated propensity score P(A=1|X) from a logistic model."""
    Xm = np.column_stack([np.ones(len(A)), X])
    beta = _logit_fit(Xm, A)
    p = 1.0 / (1.0 + np.exp(-(Xm @ beta)))
    return np.clip(p, 1e-4, 1 - 1e-4)


def _smd(x, A, w=None):
    """Standardized mean difference of covariate x between arms (optionally weighted)."""
    t1, t0 = A == 1, A == 0
    if w is None:
        m1, m0 = x[t1].mean(), x[t0].mean()
        s = np.sqrt(0.5 * (x[t1].var() + x[t0].var())) + 1e-12
    else:
        def wmean(mask):
            return np.sum(w[mask] * x[mask]) / np.sum(w[mask])
        m1, m0 = wmean(t1), wmean(t0)
        s = np.sqrt(0.5 * (x[t1].var() + x[t0].var())) + 1e-12   # unweighted pooled SD (Austin)
    return float((m1 - m0) / s)


def _match_att(ps, A, Y):
    """1:1 nearest-neighbour PS matching (treated → nearest control); returns ATT."""
    treated = np.where(A == 1)[0]
    control = np.where(A == 0)[0]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    cps = ps[control]
    order = np.argsort(cps)
    cps_sorted = cps[order]
    diffs = []
    for i in treated:
        j = np.searchsorted(cps_sorted, ps[i])
        cands = [k for k in (j - 1, j) if 0 <= k < len(cps_sorted)]
        best = min(cands, key=lambda k: abs(cps_sorted[k] - ps[i]))
        ci = control[order[best]]
        diffs.append(Y[i] - Y[ci])
    return float(np.mean(diffs))


def _weighted_diff(Y, A, w):
    return float(np.sum(w * A * Y) / np.sum(w * A) - np.sum(w * (1 - A) * Y) / np.sum(w * (1 - A)))


def full_ps(df, treat="A", outcome="Y", cov="X", true_ate=None, n_boot=300, lang="zh"):
    import ps_gen
    if true_ate is None:
        true_ate = ps_gen.TRUE_ATE
    A = np.asarray(df[treat], dtype=float); Y = np.asarray(df[outcome], dtype=float)
    X = np.asarray(df[cov], dtype=float)

    crude = float(Y[A == 1].mean() - Y[A == 0].mean())
    adjust = float(_ols([A, X], Y)[1])                       # regression adjustment Y~A+X
    ps = _ps(A, X)
    w_iptw = np.where(A == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    iptw = _weighted_diff(Y, A, w_iptw)
    w_over = np.where(A == 1, 1.0 - ps, ps)                  # overlap weights (ATO)
    overlap = _weighted_diff(Y, A, w_over)
    att = _match_att(ps, A, Y)

    smd_before = abs(_smd(X, A))
    smd_after = abs(_smd(X, A, w=w_iptw))

    # bootstrap CI for the IPTW estimate
    lo = hi = None
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(20260609); reps = []; nN = len(Y)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, nN, nN)
            try:
                pb = _ps(A[idx], X[idx]); wb = np.where(A[idx] == 1, 1.0 / pb, 1.0 / (1.0 - pb))
                reps.append(_weighted_diff(Y[idx], A[idx], wb))
            except Exception:
                pass
        if reps:
            lo = float(np.percentile(reps, 2.5)); hi = float(np.percentile(reps, 97.5))

    interp = t(
        lang,
        f"直接比較接種 vs 未接種，粗估 ATE ≈ <b>{crude:.2f}</b>，被適應症混淆（病重者較易接種、也較易出事）<b>高估</b>了"
        f"（真值 {true_ate:.2f}）。把傾向分數平衡掉後：<b>IPTW</b> ≈ <b>{iptw:.2f}</b>"
        + (f"（95% 自助 CI {lo:.2f}～{hi:.2f}）" if lo is not None else "")
        + f"、<b>重疊權重</b> ≈ {overlap:.2f}、<b>PS 配對(ATT)</b> ≈ {att:.2f}、迴歸校正 ≈ {adjust:.2f}——都貼回真值 {true_ate:.2f}。"
        f"平衡證據：共變項 X 的標準化差異 從 <b>{smd_before:.2f}</b>（加權前）降到 <b>{smd_after:.2f}</b>（加權後，應 &lt; 0.1）。",
        f"Comparing vaccinated vs unvaccinated directly, the crude ATE ≈ <b>{crude:.2f}</b> is <b>inflated</b> by confounding by "
        f"indication (sicker people are both more likely to be vaccinated and to have the outcome; truth {true_ate:.2f}). After "
        f"balancing on the propensity score: <b>IPTW</b> ≈ <b>{iptw:.2f}</b>"
        + (f" (95% bootstrap CI {lo:.2f}–{hi:.2f})" if lo is not None else "")
        + f", <b>overlap weighting</b> ≈ {overlap:.2f}, <b>PS matching (ATT)</b> ≈ {att:.2f}, regression adjustment ≈ {adjust:.2f} — "
        f"all back near the truth {true_ate:.2f}. Balance evidence: the standardized mean difference of X falls from "
        f"<b>{smd_before:.2f}</b> (before) to <b>{smd_after:.2f}</b> (after weighting; should be &lt; 0.1).",
    )
    return {
        "crude": crude, "adjust": adjust, "iptw": iptw, "overlap": overlap, "att": att,
        "ci_iptw": [lo, hi], "smd_before": smd_before, "smd_after": smd_after,
        "true_ate": float(true_ate), "n": int(len(Y)),
        "ps_treated": [round(float(v), 4) for v in ps[A == 1][:400]],
        "ps_control": [round(float(v), 4) for v in ps[A == 0][:400]],
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# ② interactive — confounding-strength knob. As severity X drives treatment more
# strongly, the crude estimate drifts up while PS methods hold the truth.
# Offline-precomputed grid (via _recompute_grid()).
# ---------------------------------------------------------------------------
_CONF_X = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def _recompute_grid(n=40000):
    import ps_gen
    out = {"conf": _CONF_X, "crude": [], "iptw": [], "overlap": [], "true_ate": ps_gen.TRUE_ATE}
    for c in out["conf"]:
        df = ps_gen.generate(seed=303, n=n, conf=c)
        r = full_ps(df, n_boot=0)
        out["crude"].append(round(r["crude"], 3))
        out["iptw"].append(round(r["iptw"], 3))
        out["overlap"].append(round(r["overlap"], 3))
    return out


# precomputed via _recompute_grid(): crude drifts up with confounding; IPTW / overlap hold.
_GRID = {
    "conf": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "crude": [2.011, 2.086, 2.311, 2.65, 3.074, 3.561, 4.072],
    "iptw": [2.011, 2.01, 2.014, 2.01, 2.015, 2.031, 2.015],
    "overlap": [2.011, 2.01, 2.015, 2.01, 2.005, 2.003, 2.008],
    "true_ate": 2.0,
}


def ps_interactive(conf=1.0, lang="zh"):
    g = _GRID
    xc = float(np.clip(conf, g["conf"][0], g["conf"][-1]))
    crude = float(np.interp(xc, g["conf"], g["crude"]))
    iptw = float(np.interp(xc, g["conf"], g["iptw"]))
    overlap = float(np.interp(xc, g["conf"], g["overlap"]))
    reading = t(
        lang,
        f"適應症混淆強度 {xc:.2f}：<b>粗估 ≈ {crude:.2f}</b> 被嚴重度 X 越推越偏；而 <b>IPTW ≈ {iptw:.2f}</b>、"
        f"<b>重疊權重 ≈ {overlap:.2f}</b> 不管混淆多強都穩在真值 {g['true_ate']:.1f}。0 時沒有混淆、粗估就準。",
        f"Confounding-by-indication strength {xc:.2f}: the <b>crude estimate ≈ {crude:.2f}</b> is pushed ever further off by severity "
        f"X; while <b>IPTW ≈ {iptw:.2f}</b> and <b>overlap weighting ≈ {overlap:.2f}</b> stay on the truth {g['true_ate']:.1f} however "
        f"strong the confounding. At 0 there is no confounding and the crude estimate is already correct.",
    )
    return {"conf": xc, "crude": crude, "iptw": iptw, "overlap": overlap,
            "true_ate": g["true_ate"], "grid": g, "reading": reading}
