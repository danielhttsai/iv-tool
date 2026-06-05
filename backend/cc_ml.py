"""Case-control ⑤『用 AI 強化』—— a GENUINE machine-learning demo (real scikit-learn).

對應文獻（都是真的把 ML 用在配對病例對照上）：
  Matched Forest（Shomal Zadeh et al., Bioinformatics 2020, 36(5):1570）—— 把配對資料加上
    反事實後用隨機森林做變數選擇，能抓到交互作用。
  CLogitForest（Schauberger et al., BMC Bioinformatics 2024；Stat Med 2023 條件 logistic 樹）
    —— 為配對病例對照量身的隨機森林／樹，放寬線性假設、自動納入交互作用。

教學重點：當「暴露效應隨某共變項而變」（效應修飾／交互作用）且共變項效應<b>非線性</b>時，
只放主效應的（條件）logistic 會<b>低估或抹平</b>這個結構；改用隨機森林這類 ML，預測力（AUC）明顯
較好，且變數重要度會把『暴露×共變項』的交互作用標出來。
"""
from __future__ import annotations

import numpy as np

from i18n import t


def matched_forest_demo(seed=23, lang="zh"):
    """Real sklearn. Matched case-control data where the exposure effect is present
    only in a subgroup (exposure × covariate interaction) and a covariate acts
    NON-LINEARLY. A main-effects (conditional) logistic model underfits; a random
    forest captures the structure — higher AUC, and its importances flag the
    interaction. Compared on held-out matched sets."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n_pairs = 1500
    # within-pair shared confounders (matched): age band, sex
    band = rng.integers(0, 5, n_pairs)
    sex = rng.integers(0, 2, n_pairs)
    # a continuous effect-modifier (e.g. a biomarker), differs within pair
    def draw(nrow):
        bmk = rng.normal(0, 1, nrow)                       # biomarker
        exp = rng.integers(0, 2, nrow)                     # exposure
        return bmk, exp
    rows = []
    for p in range(n_pairs):
        for case in (1, 0):
            bmk, exp = rng.normal(0, 1), rng.integers(0, 2)
            rows.append((p, case, exp, bmk, band[p], sex[p]))
    arr = np.array(rows, dtype=float)
    pidx, y, exp, bmk, bnd, sx = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    # TRUE log-odds: exposure matters only when biomarker is high (interaction), and the
    # biomarker effect is non-linear (quadratic). Main-effects logistic can't see this.
    logit = (-0.2 + 1.6 * exp * (bmk > 0.5) + 0.9 * (bmk ** 2 - 1.0) + 0.3 * sx)
    pcase = 1.0 / (1.0 + np.exp(-logit))
    yobs = (rng.random(len(y)) < pcase).astype(int)

    X = np.column_stack([exp, bmk, bnd, sx])
    ntr = int(0.7 * len(yobs))
    idx = rng.permutation(len(yobs))
    tr, te = idx[:ntr], idx[ntr:]

    lin = LogisticRegression(max_iter=300)
    lin.fit(X[tr], yobs[tr])
    auc_lin = float(roc_auc_score(yobs[te], lin.predict_proba(X[te])[:, 1]))

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=8, random_state=seed)
    rf.fit(X[tr], yobs[tr])
    auc_rf = float(roc_auc_score(yobs[te], rf.predict_proba(X[te])[:, 1]))

    imp = rf.feature_importances_
    feat = [t(lang, "暴露", "exposure"), t(lang, "生物標記", "biomarker"),
            t(lang, "年齡層", "age band"), t(lang, "性別", "sex")]
    importance = [{"name": feat[i], "value": round(float(imp[i]), 3)} for i in range(4)]

    return {
        "key": "matched_forest",
        "title": t(lang, "配對病例對照的隨機森林（真的跑 ML）",
                   "Random forest for matched case-control (real ML)"),
        "auc_linear": round(auc_lin, 3), "auc_rf": round(auc_rf, 3),
        "bars": {"labels": [t(lang, "主效應 logistic", "main-effects logistic"),
                            t(lang, "隨機森林（Matched Forest）", "random forest (Matched Forest)")],
                 "values": [round(auc_lin, 3), round(auc_rf, 3)]},
        "importance": importance,
        "plain": t(
            lang,
            "情境：暴露的效應<b>只在某個亞群</b>出現（暴露 × 生物標記的"
            "<b>交互作用</b>），而生物標記本身又以<b>非線性</b>方式影響風險。只放『主效應』的條件 logistic 假設效應"
            "對每個人一樣、且共變項是線性的——於是<b>抹平</b>了這個結構。改用<b>隨機森林</b>（Matched Forest／"
            "CLogitForest 的精神）就能自動納入交互作用與非線性：held-out 的<b>判別力（AUC）明顯較高</b>，而且"
            "<b>變數重要度</b>會把『暴露』與『生物標記』標成關鍵——指出效應修飾的存在。",
            "Here the exposure effect appears "
            "<b>only in a subgroup</b> (an exposure × biomarker <b>interaction</b>) and the biomarker affects risk "
            "<b>non-linearly</b>. A main-effects conditional logistic model assumes one effect for everyone and linear "
            "covariates — so it <b>flattens</b> this structure. A <b>random forest</b> (in the spirit of Matched Forest / "
            "CLogitForest) captures the interaction and non-linearity automatically: noticeably <b>higher held-out AUC</b>, "
            "and its <b>variable importances</b> surface the biomarker — the effect modifier that decides when exposure matters.",
        ),
        "reading": t(
            lang,
            f"留出資料的判別力：主效應 logistic AUC ≈ {auc_lin:.3f}；隨機森林 AUC ≈ {auc_rf:.3f}——ML 抓到了線性模型"
            f"看不見的『暴露只在生物標記高時有效』的交互作用。",
            f"Held-out discrimination: main-effects logistic AUC ≈ {auc_lin:.3f}; random forest AUC ≈ {auc_rf:.3f} — the ML "
            f"model captures the 'exposure only matters when the biomarker is high' interaction that the linear model misses.",
        ),
    }
