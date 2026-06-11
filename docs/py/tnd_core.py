"""Test-Negative Design (TND) core — pure numpy.

白話：想估疫苗對<b>目標病原</b>的效力 VE。麻煩是<b>就醫傾向</b>是未測混淆（愛就醫的人較常接種、也較常被檢驗）。

  ・<b>天真病例對照（一般族群對照）</b>：case＝檢驗陽性、control＝一般族群未感染者。被就醫傾向偏掉。
  ・<b>TND</b>：只在「有來檢驗」的人裡比——case＝目標病原陽性、control＝檢驗陰性（其他病原）。
    兩組都來檢驗過，就醫傾向被「條件在有檢驗」抵銷 → 接種勝算比 OR、<b>VE ＝ 1 − OR</b>。

識別：檢驗敏感／特異度高、疫苗不影響非目標病原、檢驗者中接種與就醫無差別關聯、症狀相似（Jackson &
Nelson 2013；Sullivan, Tchetgen Tchetgen & Cowling 2016；Vandenbroucke & Pearce 2019）。Synthetic data only.
"""
from __future__ import annotations

import numpy as np

from i18n import t


def _logit_fit(X, y, iters=40):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = p * (1 - p) + 1e-9
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _logor(case, vacc):
    """log OR of vaccination (case vs control) + its delta-method SE via a 2x2 table."""
    a = float(np.sum((case == 1) & (vacc == 1))); b = float(np.sum((case == 1) & (vacc == 0)))
    c = float(np.sum((case == 0) & (vacc == 1))); d = float(np.sum((case == 0) & (vacc == 0)))
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    logor = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return logor, se, (a, b, c, d)


def full_tnd(df, ve=None, lang="zh"):
    import tnd_gen
    if ve is None:
        ve = tnd_gen.VE_TRUE
    V = np.asarray(df["vaccinated"], float)
    tested = np.asarray(df["tested"], float)
    case = np.asarray(df["case"], float)
    infected = np.asarray(df["infected"], float)

    # --- TND: among the tested, case (target+) vs control (test-negative) ---
    m = tested == 1
    Vt, ct = V[m], case[m]
    logor_t, se_t, tab = _logor(ct, Vt)
    or_t = float(np.exp(logor_t))
    ve_tnd = 1.0 - or_t
    lo_t, hi_t = 1.0 - float(np.exp(logor_t + 1.96 * se_t)), 1.0 - float(np.exp(logor_t - 1.96 * se_t))

    # --- naive case-control with POPULATION controls (not conditioned on testing) ---
    case_idx = (tested == 1) & (case == 1)              # detected target cases
    ctrl_pool = infected == 0                           # population non-cases (any care-seeking)
    cc_case = np.concatenate([V[case_idx], V[ctrl_pool]])
    cc_y = np.concatenate([np.ones(int(case_idx.sum())), np.zeros(int(ctrl_pool.sum()))])
    logor_n, se_n, _ = _logor(cc_y, cc_case)
    ve_naive = 1.0 - float(np.exp(logor_n))

    interp = t(
        lang,
        f"真實疫苗效力 VE ＝ {ve*100:.0f}%。用「一般族群」當對照的<b>天真病例對照 VE ≈ {ve_naive*100:.0f}%</b>，"
        f"被<b>就醫傾向</b>嚴重偏掉（愛就醫者較常接種、其感染也較常被檢驗發現）。<b>陰性檢驗設計</b>只在來檢驗的人裡比："
        f"接種勝算比 OR ≈ {or_t:.2f} → <b>VE ＝ 1 − OR ≈ {ve_tnd*100:.0f}%</b>（95% CI {lo_t*100:.0f}%～{hi_t*100:.0f}%）——"
        f"貼回真值 {ve*100:.0f}%。因為 case 與 control 都<b>來檢驗過</b>，就醫傾向被條件掉了。",
        f"True vaccine effectiveness VE = {ve*100:.0f}%. A <b>naive case-control with population controls VE ≈ {ve_naive*100:.0f}%</b> "
        f"is badly biased by <b>care-seeking</b> (care-seekers vaccinate more and have their infections detected more). The "
        f"<b>test-negative design</b> compares only within those tested: the vaccination odds ratio OR ≈ {or_t:.2f} → "
        f"<b>VE = 1 − OR ≈ {ve_tnd*100:.0f}%</b> (95% CI {lo_t*100:.0f}%–{hi_t*100:.0f}%) — back on the truth {ve*100:.0f}%. "
        f"Because cases and controls were <b>all tested</b>, care-seeking is conditioned out.",
    )
    return {
        "ve_naive": ve_naive, "ve_tnd": ve_tnd, "or_tnd": or_t, "ci_tnd": [lo_t, hi_t],
        "ve_true": float(ve), "n_tested": int(m.sum()), "n_case": int(np.sum(ct == 1)),
        "n_control": int(np.sum(ct == 0)),
        "table": {"vax_case": tab[0] - 0.5, "vax_ctrl": tab[2] - 0.5,
                  "unvax_case": tab[1] - 0.5, "unvax_ctrl": tab[3] - 0.5},
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# ② interactive — care-seeking-confounding strength knob. As care-seeking drives
# vaccination & testing more, the naive VE drifts off while the TND VE holds.
# Offline-precomputed grid (via _recompute_grid()).
# ---------------------------------------------------------------------------
_CS_X = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def _recompute_grid(n=200000):
    import tnd_gen
    out = {"cs": _CS_X, "naive": [], "tnd": [], "true": tnd_gen.VE_TRUE}
    for cs in out["cs"]:
        df = tnd_gen.generate(seed=303, n=n, cseek=cs)
        r = full_tnd(df)
        out["naive"].append(round(r["ve_naive"], 3))
        out["tnd"].append(round(r["ve_tnd"], 3))
    return out


# precomputed via _recompute_grid(): naive VE sinks as care-seeking strengthens; TND holds at truth.
_GRID = {
    "cs": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "naive": [0.606, 0.537, 0.478, 0.41, 0.344, 0.299, 0.259],
    "tnd": [0.597, 0.593, 0.596, 0.595, 0.594, 0.596, 0.596],
    "true": 0.6,
}


def tnd_interactive(cseek=1.0, lang="zh"):
    g = _GRID
    x = float(np.clip(cseek, g["cs"][0], g["cs"][-1]))
    naive = float(np.interp(x, g["cs"], g["naive"]))
    tnd = float(np.interp(x, g["cs"], g["tnd"]))
    reading = t(
        lang,
        f"就醫傾向混淆強度 {x:.2f}：<b>天真病例對照 VE ≈ {naive*100:.0f}%</b> 隨就醫傾向越偏越低（低估保護）；"
        f"而 <b>TND VE ≈ {tnd*100:.0f}%</b> 不管就醫傾向多強都穩在真值 {g['true']*100:.0f}%。為 0 時兩者一致。",
        f"Care-seeking confounding strength {x:.2f}: the <b>naive case-control VE ≈ {naive*100:.0f}%</b> sinks further as "
        f"care-seeking grows (under-stating protection); the <b>TND VE ≈ {tnd*100:.0f}%</b> stays on the truth "
        f"{g['true']*100:.0f}% however strong it is. At 0 the two agree.",
    )
    return {"cseek": x, "naive": naive, "tnd": tnd, "true": g["true"], "grid": g, "reading": reading}
