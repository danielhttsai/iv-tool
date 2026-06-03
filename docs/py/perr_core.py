"""Prior Event Rate Ratio (PERR) core — pure numpy, Pyodide-safe.

白話：比較處置組 vs 對照組的事件率比，在「事前期」（兩組都還沒用藥）與「事後期」各算
一次，相除：PERR = RR_後 / RR_前。時間不變、乘法尺度的混淆會在比值上相消，露出真效果。

  RR = (處置事件/處置人時) ÷ (對照事件/對照人時)
  PERR = RR_post / RR_prior         （= PERR-ALT 的組內框架：兩組各自前→後率比再相除）
  PERD = (rate_T,post − rate_C,post) − (rate_T,prior − rate_C,prior)   （加法尺度版本）
log 尺度 delta method：Var(log PERR) = 1/E_Tpost+1/E_Cpost+1/E_Tprior+1/E_Cprior。
"""
from __future__ import annotations

import numpy as np

from i18n import t


def _rate(events, pt):
    e = float(np.sum(events)); p = float(np.sum(pt))
    return (e / p if p > 0 else float("nan")), e, p


def _cells(df, group, ev_prior, pt_prior, ev_post, pt_post):
    g = np.asarray(df[group])
    T, C = g == 1, g == 0
    rTp, eTp, _ = _rate(np.asarray(df[ev_prior])[T], np.asarray(df[pt_prior])[T])
    rCp, eCp, _ = _rate(np.asarray(df[ev_prior])[C], np.asarray(df[pt_prior])[C])
    rTs, eTs, _ = _rate(np.asarray(df[ev_post])[T], np.asarray(df[pt_post])[T])
    rCs, eCs, _ = _rate(np.asarray(df[ev_post])[C], np.asarray(df[pt_post])[C])
    return {"rTp": rTp, "rCp": rCp, "rTs": rTs, "rCs": rCs,
            "eTp": eTp, "eCp": eCp, "eTs": eTs, "eCs": eCs}


def full_perr(df, group="group", events_prior="events_prior", pt_prior="pt_prior",
              events_post="events_post", pt_post="pt_post", true_rr=0.70, lang="zh"):
    c = _cells(df, group, events_prior, pt_prior, events_post, pt_post)
    rr_prior = c["rTp"] / c["rCp"] if c["rCp"] else float("nan")
    rr_post = c["rTs"] / c["rCs"] if c["rCs"] else float("nan")
    perr = rr_post / rr_prior if rr_prior else float("nan")
    # delta-method CI on the log scale (Poisson event-count variances)
    var = sum(1.0 / max(e, 0.5) for e in (c["eTs"], c["eCs"], c["eTp"], c["eCp"]))
    se = float(np.sqrt(var))
    lo = float(perr * np.exp(-1.96 * se)); hi = float(perr * np.exp(1.96 * se))
    # additive-scale companion (PERD): difference-in-differences of rates
    perd = (c["rTs"] - c["rCs"]) - (c["rTp"] - c["rCp"])

    interp = t(
        lang,
        f"事前事件率比（PERR）估計因果率比 RR ≈ {perr:.2f}（95% 信賴區間 {lo:.2f}～{hi:.2f}）。"
        f"對照：只看事後期的天真率比約 {rr_post:.2f}——被適應症混淆（體弱者較易用藥也較易發病）"
        f"嚴重拉偏，幾乎看不出藥效。事前期率比 {rr_prior:.2f} 正是這個混淆的「指紋」；"
        f"把它從事後期率比中除掉，就還原出真效果（真值 RR={true_rr:.2f}）。",
        f"The prior event rate ratio (PERR) estimates a causal rate ratio of RR ≈ {perr:.2f} "
        f"(95% CI {lo:.2f}–{hi:.2f}). Contrast: the naive post-period rate ratio is about {rr_post:.2f} — badly "
        f"pulled off by confounding by indication (frail patients are both more likely to be treated and to have the "
        f"event), so the drug's benefit is hidden. The prior-period rate ratio {rr_prior:.2f} is the fingerprint of "
        f"that confounding; dividing it out of the post-period ratio recovers the true effect (true RR={true_rr:.2f}).",
    )
    return {
        "perr": float(perr), "ci": [lo, hi], "log_se": se,
        "rr_prior": float(rr_prior), "rr_post": float(rr_post),
        "naive_rr": float(rr_post), "perd": float(perd),
        "rates": {"treated_prior": c["rTp"], "control_prior": c["rCp"],
                  "treated_post": c["rTs"], "control_post": c["rCs"]},
        "events": {"treated_prior": c["eTp"], "control_prior": c["eCp"],
                   "treated_post": c["eTs"], "control_post": c["eCs"]},
        "true_rr": true_rr, "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# ⑤ scale-sensitivity demo (a documented refinement, NOT machine learning):
# PERR (multiplicative) vs PERD (additive). Each shines under its matching scale.
# ---------------------------------------------------------------------------
def scale_demo(seed=7, lang="zh"):
    import perr_gen
    rng = np.random.default_rng(seed)

    # (A) MULTIPLICATIVE confounding (the default generator): PERR matches the true RR.
    dfm = perr_gen.generate(seed=seed)
    m = full_perr(dfm)
    a = {"perr": m["perr"], "perd": m["perd"], "true_rr": perr_gen.TRUE_RR}

    # (B) ADDITIVE confounding: frailty ADDS a constant rate; the drug REMOVES a
    # constant rate. Here PERD (difference scale) recovers the additive effect,
    # while PERR (ratio scale) is biased.
    n = 8000
    frail = rng.binomial(1, 0.4, n)
    group = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.2 + 1.6 * frail))))
    base, add_frail, true_diff, PT = 0.05, 0.10, -0.04, 3.0
    lam_prior = base + add_frail * frail
    lam_post = base + add_frail * frail + np.where(group == 1, true_diff, 0.0)
    lam_post = np.clip(lam_post, 1e-4, None)
    dfa = (__import__("pandas").DataFrame({
        "pid": np.arange(n), "group": group,
        "events_prior": rng.poisson(lam_prior * PT), "pt_prior": np.full(n, PT),
        "events_post": rng.poisson(lam_post * PT), "pt_post": np.full(n, PT),
    }))
    aa = full_perr(dfa)
    b = {"perr": aa["perr"], "perd": aa["perd"], "true_diff": true_diff}

    return {
        "multiplicative": a, "additive": b,
        "plain": t(
            lang,
            "情境（這是文獻中真有的精修，<b>不是</b> AI／機器學習）：混淆可能作用在<b>乘法</b>尺度"
            "（讓率變成幾倍）或<b>加法</b>尺度（讓率加減一個固定值）。PERR 假設乘法、在比值上相消；"
            "PERD（事前事件率<b>差</b>）假設加法、在差值上相消。兩者不可能同時對——資料長得像哪種尺度，"
            "就該用對應的方法，並把另一個當敏感度分析。",
            "Scenario (a genuine refinement from the literature — <b>not</b> AI/ML): confounding may act on a "
            "<b>multiplicative</b> scale (multiplying the rate) or an <b>additive</b> scale (adding a fixed amount to the "
            "rate). PERR assumes multiplicative and cancels on the ratio; PERD (prior event rate <b>difference</b>) "
            "assumes additive and cancels on the difference. They cannot both be right — match the method to the scale "
            "the data look like, and report the other as a sensitivity analysis.",
        ),
        "reading": t(
            lang,
            f"乘法混淆下：PERR ≈ {a['perr']:.2f} 命中真值 RR {a['true_rr']:.2f}（PERD 在此尺度無對應意義）。"
            f"加法混淆下：PERD ≈ {b['perd']:.3f} 命中真值率差 {b['true_diff']:.2f}（此時 PERR ≈ {b['perr']:.2f} 偏掉）。",
            f"Under multiplicative confounding: PERR ≈ {a['perr']:.2f} hits the true RR {a['true_rr']:.2f} (PERD is not "
            f"the right scale here). Under additive confounding: PERD ≈ {b['perd']:.3f} hits the true rate difference "
            f"{b['true_diff']:.2f} (there PERR ≈ {b['perr']:.2f} is biased).",
        ),
    }
