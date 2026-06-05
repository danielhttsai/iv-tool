"""SCCS ⑤『用 AI 強化』—— a GENUINE machine-learning demo (real scikit-learn).

對應文獻：Xu, Chen, Zeng & Wang (2022, Statistics in Medicine 41) —— <b>self-matched
learning</b>：用「同一個人前後比」的自我對照訊號＋機器學習，去找出<b>效應的異質性</b>、
建構<b>個人化的治療決策規則</b>。標準 SCCS 只給<b>一個平均 IRR</b>；當不同人的危險窗風險其實差很多
（效應修飾）時，平均值會把高風險亞群藏起來。把每個人的自我對照結果交給隨機森林，就能依基線特徵
把<b>高風險亞群</b>標出來
"""
from __future__ import annotations

import numpy as np

from i18n import t


def self_matched_demo(seed=31, lang="zh"):
    """Real sklearn. Each case contributes a within-person self-controlled label (did the
    event fall in the risk window?). The true risk-window effect is HETEROGENEOUS — much
    larger in one subgroup (an interaction of two baseline features). A single pooled IRR
    averages it away; a random forest on the self-matched labels recovers WHO is high-risk
    (higher held-out AUC + a clear subgroup IRR), in the spirit of Xu et al. 2022."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = 6000
    OBS, RISK = 365.0, 56.0
    age = rng.integers(18, 85, n).astype(float)
    male = rng.integers(0, 2, n).astype(float)
    biomarker = rng.normal(0, 1, n)
    vacc = rng.integers(30, OBS - 90, n).astype(float)
    e_risk = np.minimum(OBS, vacc + RISK) - (vacc + 1)
    e_base = OBS - e_risk
    # HETEROGENEOUS true IRR: high only for young males (an interaction), low otherwise
    high = ((age < 40) & (male == 1)).astype(float)
    irr_i = np.where(high == 1, 9.0, 1.2)
    p_risk = (e_risk * irr_i) / (e_base + e_risk * irr_i)
    in_risk = (rng.random(n) < p_risk).astype(int)            # the self-matched label

    X = np.column_stack([age, male, biomarker])
    ntr = int(0.7 * n); idx = rng.permutation(n); tr, te = idx[:ntr], idx[ntr:]
    # a random forest learns WHO tends to have the event in the risk window (self-matched
    # label), automatically finding the age×sex interaction a main-effects model misses
    rf = RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=12, random_state=seed).fit(X[tr], in_risk[tr])
    auc_rf = float(roc_auc_score(in_risk[te], rf.predict_proba(X[te])[:, 1]))
    # use the forest's predicted risk to flag the top-quintile high-risk group, then
    # estimate each group's own within-person IRR (crude, person-time weighted)
    score = rf.predict_proba(X)[:, 1]
    flag = score >= np.quantile(score, 0.80)
    def crude_irr(mask):
        er = e_risk[mask].sum(); eb = e_base[mask].sum()
        nr = in_risk[mask].sum(); nb = int(mask.sum()) - nr
        return (nr / er) / (nb / eb) if er and eb and nb else float("nan")
    irr_hi = crude_irr(flag); irr_lo = crude_irr(~flag); pooled = crude_irr(np.ones(n, bool))

    imp = rf.feature_importances_
    feat = [t(lang, "年齡", "age"), t(lang, "性別", "sex"), t(lang, "生物標記", "biomarker")]
    importance = [{"name": feat[i], "value": round(float(imp[i]), 3)} for i in range(3)]

    return {
        "key": "self_matched",
        "title": t(lang, "自我對照 × 機器學習：找出高風險亞群（真的跑 ML）",
                   "Self-matched learning: find the high-risk subgroup (real ML)"),
        "auc_rf": round(auc_rf, 3),
        "pooled_irr": round(pooled, 2), "irr_high": round(irr_hi, 2), "irr_low": round(irr_lo, 2),
        "bars": {"labels": [t(lang, "整體（單一 IRR）", "pooled (single IRR)"),
                            t(lang, "ML 標出的高風險亞群", "ML-flagged high-risk subgroup"),
                            t(lang, "其餘", "everyone else")],
                 "values": [round(pooled, 2), round(irr_hi, 2), round(irr_lo, 2)]},
        "importance": importance,
        "plain": t(
            lang,
            "標準 SCCS 只回一個<b>平均 IRR</b>；但若危險窗的風險其實"
            "<b>因人而異</b>（效應修飾——這裡只有「年輕男性」風險特別高），平均值就把高風險亞群<b>稀釋藏起來</b>。"
            "把每個人的<b>自我對照標籤</b>（事件有沒有落在危險窗）交給<b>隨機森林</b>（Xu 等 2022 的 self-matched "
            "learning 精神），它會自動抓到「年齡 × 性別」的交互作用、依基線特徵把<b>高風險亞群</b>標出來：再分別"
            "估每群自己的 IRR，整體的單一數字就被拆成天差地遠的兩群——這正是個人化決策規則的起點。",
            "Standard SCCS returns a single "
            "<b>pooled IRR</b>; but if the risk-window effect is actually <b>heterogeneous</b> (effect modification — here it "
            "is large only for young males), the average <b>hides the high-risk subgroup</b>. Feeding each case's "
            "<b>self-matched label</b> (did the event fall in the risk window?) to a <b>random forest</b> (the spirit of Xu "
            "et al. 2022's self-matched learning) automatically captures the age × sex interaction and flags the "
            "<b>high-risk subgroup</b> by baseline features; estimating each group's own IRR splits the single pooled number "
            "into two very different subgroups — the starting point for a personalized decision rule.",
        ),
        "reading": t(
            lang,
            f"整體只看到單一 <b>IRR ≈ {pooled:.2f}</b>；機器學習依基線特徵把它拆開——標出的高風險亞群 "
            f"<b>IRR ≈ {irr_hi:.2f}</b>、其餘僅 ≈ {irr_lo:.2f}（森林預測『事件落在危險窗』的留出 AUC ≈ {auc_rf:.3f}）。",
            f"The pooled analysis sees a single <b>IRR ≈ {pooled:.2f}</b>; ML splits it by baseline features — the flagged "
            f"high-risk subgroup has <b>IRR ≈ {irr_hi:.2f}</b> vs ≈ {irr_lo:.2f} for the rest (the forest's held-out AUC for "
            f"'event in the risk window' ≈ {auc_rf:.3f}).",
        ),
    }
