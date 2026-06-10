"""PS ⑤『用 AI 強化』—— a GENUINE machine-learning demo (real scikit-learn).

對應文獻：高維／機器學習傾向分數（Schneeweiss et al. 2009 的 hdPS 精神；以 ML 估計 PS）。當「會不會接種」
和共變項的關係是<b>非線性／有交互</b>時，只放主效應的<b>邏輯斯 PS</b> 設定錯誤、加權後仍<b>殘留失衡與偏誤</b>；
改用<b>梯度提升</b>估 PS，能把非線性學對，<b>平衡更好、ATE 回到真值</b>。

教學重點：PS 的好壞看<b>平衡</b>不看 AUC。ML 在這裡的角色是把<b>傾向分數模型</b>配得更彈性，識別仍靠
「無未測混淆＋正性」這些因果假設。誠實聲明：合成資料、簡化教學重建。
"""
from __future__ import annotations

import numpy as np

from i18n import t


def _gen_nonlinear(rng, n):
    """True DGP with a NON-LINEAR / interactive propensity surface."""
    X1 = rng.normal(0, 1, n)                                 # severity
    X2 = rng.normal(0, 1, n)                                 # age
    logit = 1.6 * (X1 > 0.5) + 1.3 * X1 * X2 - 0.9 * (X2 ** 2 - 1.0)
    A = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    Y = 2.0 * A + 1.0 * X1 + 0.8 * X2 + 0.7 * X1 * X2 + rng.normal(0, 1.0, n)
    return X1, X2, A, Y


def _iptw(Y, A, ps):
    ps = np.clip(ps, 1e-3, 1 - 1e-3)
    w = np.where(A == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    return float(np.sum(w * A * Y) / np.sum(w * A) - np.sum(w * (1 - A) * Y) / np.sum(w * (1 - A))), w


def _max_smd(cols, A, w):
    out = 0.0
    for x in cols:
        m1 = np.sum(w[A == 1] * x[A == 1]) / np.sum(w[A == 1])
        m0 = np.sum(w[A == 0] * x[A == 0]) / np.sum(w[A == 0])
        s = np.sqrt(0.5 * (x[A == 1].var() + x[A == 0].var())) + 1e-12
        out = max(out, abs((m1 - m0) / s))
    return float(out)


def ml_ps_demo(seed=41, lang="zh"):
    """Real sklearn. A non-linear / interactive propensity surface: a main-effects LOGISTIC
    PS stays imbalanced and biased after IPTW; a GRADIENT-BOOSTING PS balances the covariates
    and recovers the true ATE (2.0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X1, X2, A, Y = _gen_nonlinear(rng, 4000)
    feat = np.column_stack([X1, X2])

    # balance is judged on the terms that actually carry the confounding here — the
    # main effects AND the interaction / quadratic terms a main-effects model ignores.
    bal_cols = [X1, X2, X1 * X2, X2 ** 2 - 1.0]

    lin = LogisticRegression(max_iter=400)
    lin.fit(feat, A)
    ps_lin = lin.predict_proba(feat)[:, 1]
    ate_lin, w_lin = _iptw(Y, A, ps_lin)
    smd_lin = _max_smd(bal_cols, A, w_lin)

    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=seed)
    gb.fit(feat, A)
    ps_gb = gb.predict_proba(feat)[:, 1]
    ate_gb, w_gb = _iptw(Y, A, ps_gb)
    smd_gb = _max_smd(bal_cols, A, w_gb)

    truth = 2.0
    return {
        "key": "ps_ml",
        "title": t(lang, "高維／機器學習傾向分數：邏輯斯 vs 梯度提升（真的跑 ML）",
                   "High-dimensional / ML propensity score: logistic vs gradient boosting (real ML)"),
        "ate_lin": round(ate_lin, 3), "ate_gb": round(ate_gb, 3), "ate_true": truth,
        "smd_lin": round(smd_lin, 3), "smd_gb": round(smd_gb, 3),
        "bars": {"labels": [t(lang, "主效應邏輯斯 PS", "main-effects logistic PS"),
                            t(lang, "梯度提升 PS（ML）", "gradient-boosting PS (ML)"),
                            t(lang, "真值", "truth")],
                 "values": [round(ate_lin, 3), round(ate_gb, 3), truth]},
        "plain": t(
            lang,
            "情境：「會不會接種」和嚴重度、年齡的關係是<b>非線性、有交互</b>的。只放主效應的<b>邏輯斯 PS</b> 把這個結構配錯，"
            "IPTW 加權後共變項<b>仍然失衡</b>，ATE 也<b>偏</b>。改用<b>梯度提升</b>估 PS（高維 PS／hdPS 的精神），把非線性學對："
            "加權後<b>平衡變好</b>、ATE <b>回到真值</b>。重點：PS 的標準是<b>平衡</b>，不是預測準度；ML 讓 PS 模型更彈性，"
            "但識別仍靠「無未測混淆＋正性」。",
            "Here the propensity to be vaccinated depends on severity and age in a <b>non-linear, interactive</b> way. A main-effects "
            "<b>logistic PS</b> mis-specifies this, so after IPTW the covariates are <b>still imbalanced</b> and the ATE is <b>biased</b>. "
            "A <b>gradient-boosting</b> PS (the spirit of high-dimensional / hdPS) learns the non-linearity: after weighting the "
            "<b>balance improves</b> and the ATE <b>returns to the truth</b>. The point: a PS is judged by <b>balance</b>, not "
            "prediction accuracy; ML makes the PS model flexible, but identification still rests on no-unmeasured-confounding + positivity.",
        ),
        "reading": t(
            lang,
            f"IPTW 後的 ATE：邏輯斯 PS ≈ {ate_lin:.2f}、ML PS ≈ {ate_gb:.2f}、真值 {truth:.1f}。加權後最大 SMD："
            f"邏輯斯 {smd_lin:.2f} → ML {smd_gb:.2f}（越小越平衡）——ML 把線性模型留下的失衡清掉了。",
            f"ATE after IPTW: logistic PS ≈ {ate_lin:.2f}, ML PS ≈ {ate_gb:.2f}, truth {truth:.1f}. Max SMD after weighting: "
            f"logistic {smd_lin:.2f} → ML {smd_gb:.2f} (smaller = better balance) — ML cleared the imbalance the linear model left.",
        ),
    }
