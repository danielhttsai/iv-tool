"""TND ⑤『用 AI 強化』—— a GENUINE machine-learning demo (real scikit-learn).

對應文獻：因果／調整式陰性檢驗設計（Schnitzer 2022；Rowley 等 2025 模擬）。TND 去掉的是<b>就醫傾向</b>，
但<b>不</b>會自動去掉其他<b>已測混淆</b>（如年齡、共病）。當這些已測混淆對接種與感染是<b>非線性</b>影響時，
<b>未調整的 TND OR 仍偏</b>；用<b>傾向分數 IPW</b>調整、且 PS 用<b>梯度提升</b>學非線性，VE 才回到真值。

教學重點：TND 處理就醫傾向；其餘已測混淆要靠調整（IPW／g-comp）；ML 讓 PS 更彈性。識別仍需 TND 的假設。
誠實聲明：合成資料、簡化教學重建。
"""
from __future__ import annotations

import numpy as np

from i18n import t


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _weighted_or(case, V, w):
    a = np.sum(w * (case == 1) * (V == 1)); b = np.sum(w * (case == 1) * (V == 0))
    c = np.sum(w * (case == 0) * (V == 1)); d = np.sum(w * (case == 0) * (V == 0))
    return float((a * d) / (b * c + 1e-9))


def ml_tnd_demo(seed=9, lang="zh"):
    """Real sklearn. A measured confounder X (age/comorbidity) affects vaccination and the target
    non-linearly. The crude TND OR is still confounded by X; an IPW-adjusted (causal) TND with a
    LOGISTIC PS is only partly fixed, while a GRADIENT-BOOSTING PS recovers the true VE (0.60)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(seed)
    ve = 0.60
    n = 45000
    X = rng.normal(0, 1, n)
    V = (rng.random(n) < _sig(0.7 * X + 0.8 * (X ** 2 - 1.0))).astype(float)        # non-linear PS
    T = (rng.random(n) < _sig(-1.0 + np.log(1 - ve) * V + 0.9 * X + 0.7 * (X ** 2 - 1.0))).astype(float)
    Npath = (rng.random(n) < _sig(-1.0 + 0.9 * X + 0.7 * (X ** 2 - 1.0))).astype(float)
    keep = (T == 1) | ((Npath == 1) & (T == 0))
    Xt, Vt = X[keep].reshape(-1, 1), V[keep]
    case = (T[keep] == 1).astype(float)

    ve_crude = 1.0 - _weighted_or(case, Vt, np.ones(len(case)))

    lin = LogisticRegression(max_iter=300).fit(Xt, Vt)
    ps_lin = np.clip(lin.predict_proba(Xt)[:, 1], 1e-3, 1 - 1e-3)
    w_lin = np.where(Vt == 1, 1 / ps_lin, 1 / (1 - ps_lin))
    ve_lin = 1.0 - _weighted_or(case, Vt, w_lin)

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=seed).fit(Xt, Vt)
    ps_gb = np.clip(gb.predict_proba(Xt)[:, 1], 1e-3, 1 - 1e-3)
    w_gb = np.where(Vt == 1, 1 / ps_gb, 1 / (1 - ps_gb))
    ve_gb = 1.0 - _weighted_or(case, Vt, w_gb)

    return {
        "key": "tnd_ml",
        "title": t(lang, "因果 TND：未調整 vs IPW（邏輯斯）vs IPW（梯度提升）（真的跑 ML）",
                   "Causal TND: unadjusted vs IPW (logistic) vs IPW (gradient boosting) — real ML"),
        "ve_crude": round(ve_crude, 3), "ve_lin": round(ve_lin, 3), "ve_gb": round(ve_gb, 3), "ve_true": ve,
        "bars": {"labels": [t(lang, "未調整 TND", "unadjusted TND"),
                            t(lang, "IPW（邏輯斯）", "IPW (logistic)"),
                            t(lang, "IPW（梯度提升 ML）", "IPW (gradient boosting, ML)"),
                            t(lang, "真值", "truth")],
                 "values": [round(ve_crude, 3), round(ve_lin, 3), round(ve_gb, 3), ve]},
        "plain": t(
            lang,
            "情境：除了就醫傾向，還有一個<b>已測混淆 X</b>（年齡／共病）對接種與感染都是<b>非線性</b>影響。TND 去掉了就醫傾向，"
            "但<b>沒</b>去掉 X——所以<b>未調整的 TND VE 仍偏</b>。用<b>傾向分數 IPW</b> 調整 X：若 PS 用<b>邏輯斯</b>（主效應）會"
            "設定錯、只修一半；改用<b>梯度提升</b> 學會非線性的 PS，VE 才<b>回到真值</b>。重點：TND 管就醫傾向，其餘已測混淆"
            "要靠調整，ML 讓調整更準。",
            "Beyond care-seeking there is a <b>measured confounder X</b> (age / comorbidity) that affects vaccination and infection "
            "<b>non-linearly</b>. TND removes care-seeking but <b>not</b> X — so the <b>unadjusted TND VE is still biased</b>. Adjusting "
            "for X with a <b>propensity-score IPW</b>: a main-effects <b>logistic</b> PS mis-specifies and only half-fixes it, while a "
            "<b>gradient-boosting</b> PS learns the non-linearity and the VE <b>returns to the truth</b>. The point: TND handles "
            "care-seeking; other measured confounders need adjustment, and ML makes that adjustment more accurate.",
        ),
        "reading": t(
            lang,
            f"VE：未調整 TND ≈ {ve_crude*100:.0f}%、IPW 邏輯斯 ≈ {ve_lin*100:.0f}%、IPW 梯度提升 ≈ {ve_gb*100:.0f}%、"
            f"真值 {ve*100:.0f}%。已測混淆 X 非線性時，ML 傾向分數把調整做對、VE 貼回真值。",
            f"VE: unadjusted TND ≈ {ve_crude*100:.0f}%, IPW logistic ≈ {ve_lin*100:.0f}%, IPW gradient boosting ≈ {ve_gb*100:.0f}%, "
            f"truth {ve*100:.0f}%. With a non-linear measured confounder X, the ML propensity score gets the adjustment right and VE "
            f"returns to the truth.",
        ),
    }
