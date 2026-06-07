"""NC ⑤『用 AI 強化』—— empirical calibration with a panel of negative controls.

對應文獻：Schuemie et al. —— <b>實證校準（empirical calibration）</b>：用一<b>大批已知為虛無</b>的陰性對照
（真效應＝0）去估計分析的<b>系統誤差分布</b>，再用它把觀察到的 p 值／信賴區間<b>校準</b>回誠實的水準
（Stat Med 2014 校 p 值；PNAS 2018 校信賴區間；Stat Med 2023 加序列監測）。

教學重點：天真分析假設「只有抽樣誤差」，於是連<b>真效應為 0 的陰性對照</b>都被判為「顯著」——這證明 p 值被
<b>系統誤差</b>汙染。把這批陰性對照的估計擬合成經驗虛無 Normal(μ,σ)，校準後：陰性對照回到該有的 ~5% 偽陽性，
主結果的 p 值／CI 也誠實了。

誠實聲明：這是<b>資料驅動的校準</b>，不是機器學習本身；ML 前沿另有 DANCE（自動搜尋／驗證陰性對照，Kummerfeld
2024）與用 ML 估計近端因果的橋函數——本頁不實作，僅列為延伸。
"""
from __future__ import annotations

import numpy as np


def _norm_cdf(x):
    # standard normal CDF via erf (avoid importing scipy at module load)
    from math import erf, sqrt
    if np.isscalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in np.asarray(x, dtype=float)])


def calibration_demo(seed=67, lang="zh"):
    """Real, runnable empirical calibration (Schuemie). A panel of K negative controls (true
    effect = 0) all carry the SAME systematic error from unmeasured confounding, so their naive
    p-values are falsely 'significant'. Fitting an empirical null Normal(mu, sigma) to them and
    recalibrating restores honest p-values/CIs — for the negative controls AND the main estimate."""
    from i18n import t
    rng = np.random.default_rng(seed)
    K = 50
    # systematic error: a shared bias (mean) + extra between-control variation (on log-RR scale)
    sys_mean, sys_sd = 0.45, 0.20
    se_nc = rng.uniform(0.08, 0.16, K)                       # each NC's sampling SE
    sys_err = rng.normal(sys_mean, sys_sd, K)               # the systematic error per control
    nc_est = sys_err + rng.normal(0, 1, K) * se_nc          # observed NC estimates (true effect 0)

    # empirical null: fit Normal(mu, sigma) to the NC estimates (mean + extra systematic SD)
    mu = float(np.mean(nc_est))
    extra_var = max(float(np.var(nc_est, ddof=1) - np.mean(se_nc ** 2)), 1e-6)
    sigma = float(np.sqrt(extra_var))                       # systematic-error SD beyond sampling

    def p_naive(est, se):
        return float(2 * (1 - _norm_cdf(abs(est) / se)))
    def p_cal(est, se):
        return float(2 * (1 - _norm_cdf(abs(est - mu) / np.sqrt(sigma ** 2 + se ** 2))))

    # type-I error among the negative controls (true effect = 0): naive vs calibrated
    t1_naive = float(np.mean([p_naive(nc_est[i], se_nc[i]) < 0.05 for i in range(K)]))
    t1_cal = float(np.mean([p_cal(nc_est[i], se_nc[i]) < 0.05 for i in range(K)]))

    # the MAIN estimate: a genuine but modest effect, also hit by the same systematic error
    main_true = 0.30
    main_se = 0.12
    main_est = main_true + sys_mean + rng.normal(0, main_se)
    pmain_naive = p_naive(main_est, main_se)
    pmain_cal = p_cal(main_est, main_se)
    ci_naive = [main_est - 1.96 * main_se, main_est + 1.96 * main_se]
    ci_cal = [main_est - mu - 1.96 * np.sqrt(sigma ** 2 + main_se ** 2),
              main_est - mu + 1.96 * np.sqrt(sigma ** 2 + main_se ** 2)]

    return {
        "key": "nc_calibration",
        "title": t(lang, "實證校準：用一批陰性對照修正被系統誤差汙染的 p 值／CI",
                   "Empirical calibration: fix p-values / CIs distorted by systematic error, using a panel of negative controls"),
        "k": K, "null_mean": round(mu, 3), "null_sd": round(sigma, 3),
        "type1_naive": round(t1_naive, 3), "type1_cal": round(t1_cal, 3),
        "nc_est": [round(float(v), 3) for v in nc_est], "nc_se": [round(float(v), 3) for v in se_nc],
        "main_est": round(float(main_est), 3), "main_se": round(float(main_se), 3),
        "p_naive": pmain_naive, "p_cal": pmain_cal,
        "ci_naive": [round(float(ci_naive[0]), 3), round(float(ci_naive[1]), 3)],
        "ci_cal": [round(float(ci_cal[0]), 3), round(float(ci_cal[1]), 3)],
        "bars": {"labels": [t(lang, "天真（假設只有抽樣誤差）", "naive (sampling error only)"),
                            t(lang, "實證校準後", "after empirical calibration")],
                 "values": [round(t1_naive * 100, 1), round(t1_cal * 100, 1)]},
        "plain": t(
            lang,
            "天真的 p 值／信賴區間只假設<b>抽樣誤差</b>。但在觀察性資料裡還有<b>系統誤差</b>（未測混淆等），於是連<b>真效應為 0 的"
            "陰性對照</b>都會被一堆判為「顯著」——上圖左：天真下，這 50 個本該虛無的陰性對照有<b>很高比例</b> p<0.05，遠超過該有的 5%。"
            "<b>實證校準</b>（Schuemie）把這批陰性對照的估計擬合成<b>經驗虛無 Normal(μ,σ)</b>，量出系統誤差，再用它重算 p 值／CI："
            "校準後陰性對照回到 ~5%（右），主結果的 p 值／CI 也誠實了。這是<b>資料驅動的校準</b>，不是 ML 本身（ML 前沿見下方註）。",
            "Naive p-values / CIs assume <b>sampling error only</b>. But observational data also carry <b>systematic error</b> "
            "(unmeasured confounding, etc.), so even <b>negative controls with a true effect of 0</b> are flagged 'significant' en masse — "
            "left bar: under naive analysis a <b>large fraction</b> of these 50 truly-null controls have p<0.05, far above the nominal 5%. "
            "<b>Empirical calibration</b> (Schuemie) fits an <b>empirical null Normal(μ,σ)</b> to those control estimates, measures the "
            "systematic error, and recomputes p-values / CIs: afterwards the controls return to ~5% (right), and the main result's p / CI "
            "become honest too. This is <b>data-driven calibration</b>, not ML itself (ML frontier noted below).",
        ),
        "reading": t(
            lang,
            f"經驗虛無 ≈ Normal(μ={mu:.2f}, σ={sigma:.2f})。陰性對照偽陽性率：天真 {t1_naive*100:.0f}% → 校準後 {t1_cal*100:.0f}%（應 ~5%）。"
            f"主結果：估計 {main_est:.2f}，天真 p={pmain_naive:.3f}（假顯著）→ 校準後 p={pmain_cal:.3f}；"
            f"CI {ci_naive[0]:.2f}～{ci_naive[1]:.2f} → 校準後 {ci_cal[0]:.2f}～{ci_cal[1]:.2f}。",
            f"Empirical null ≈ Normal(μ={mu:.2f}, σ={sigma:.2f}). Negative-control false-positive rate: naive {t1_naive*100:.0f}% → "
            f"calibrated {t1_cal*100:.0f}% (should be ~5%). Main result: estimate {main_est:.2f}, naive p={pmain_naive:.3f} (falsely "
            f"significant) → calibrated p={pmain_cal:.3f}; CI {ci_naive[0]:.2f}–{ci_naive[1]:.2f} → calibrated {ci_cal[0]:.2f}–{ci_cal[1]:.2f}.",
        ),
    }
