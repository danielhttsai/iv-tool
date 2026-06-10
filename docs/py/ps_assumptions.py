"""Propensity Score assumption checks C1–C5. PS methods only remove confounding from
MEASURED covariates; the key 'no unmeasured confounding' is untestable (blue card). The
testable supports: positivity/overlap of the PS, covariate balance after weighting (SMD),
and effective sample size. Covariate selection (which X to put in the PS) is design judgement.

Each check returns {id, title, status, headline, plain, term, metrics:[...]};
run_dashboard returns {"checks": [...]}.
"""
from __future__ import annotations

import numpy as np

import ps_core
from i18n import t


def run_dashboard(df, treat="A", outcome="Y", cov="X", lang="zh"):
    res = ps_core.full_ps(df, treat, outcome, cov, n_boot=0, lang=lang)
    A = np.asarray(df[treat], dtype=float); X = np.asarray(df[cov], dtype=float)
    ps = ps_core._ps(A, X)
    # positivity: how extreme are the PS values, and how much overlap
    frac_extreme = float(np.mean((ps < 0.05) | (ps > 0.95)))
    w = np.where(A == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    ess = float(np.sum(w) ** 2 / np.sum(w ** 2))            # effective sample size of IPTW
    checks = [
        _c1_unmeasured(lang),
        _c2_positivity(frac_extreme, lang),
        _c3_balance(res, lang),
        _c4_selection(lang),
        _c5_ess(ess, len(A), lang),
    ]
    return {"checks": checks}


def _c1_unmeasured(lang="zh"):
    return {
        "id": "C1",
        "title": t(lang, "無未測混淆（conditional exchangeability，最關鍵、不可檢驗）",
                   "No unmeasured confounding (conditional exchangeability — the key, untestable assumption)"),
        "status": "info",
        "headline": t(lang, "PS 只能平衡<b>有測到</b>的共變項；沒測到的混淆（如未記錄的疾病嚴重度）它救不了。",
                      "PS only balances the covariates you <b>measured</b>; an unmeasured confounder (e.g. unrecorded severity) it cannot fix."),
        "plain": t(
            lang,
            "傾向分數的全部威力都建立在「<b>在已測共變項 X 之下，接種與否可交換</b>」這條假設上——也就是<b>沒有未測到的混淆</b>。"
            "PS 配對／加權只把<b>量得到</b>的 X 平衡掉；任何沒被記錄、卻同時影響接種與結果的因子（虛弱、生活型態、未編碼的病史）"
            "仍會殘留偏誤，且<b>無法用資料證明它不存在</b>。實務上靠：豐富的共變項、主動對照／新使用者設計、陰性對照與敏感度分析。",
            "All of the propensity score's power rests on the assumption that, <b>given the measured covariates X, treatment is "
            "exchangeable</b> — i.e. <b>no unmeasured confounding</b>. PS matching/weighting only balances the X you can <b>measure</b>; "
            "any unrecorded factor that drives both vaccination and outcome (frailty, lifestyle, uncoded history) leaves residual bias, "
            "and you <b>cannot prove from data that it is absent</b>. Support it with rich covariates, active-comparator / new-user "
            "designs, negative controls, and sensitivity analysis.",
        ),
        "term": t(lang, "專有名詞：條件可交換性（conditional exchangeability）；無未測混淆；可忽略性（ignorability）。",
                  "Term: conditional exchangeability; no unmeasured confounding; ignorability."),
        "metrics": [],
    }


def _c2_positivity(frac_extreme, lang="zh"):
    if frac_extreme < 0.02:
        status, head = "green", t(lang, "重疊良好——幾乎沒有 PS 接近 0 或 1 的人，加權穩定。",
                                  "Good overlap — almost no one has a PS near 0 or 1; weighting is stable.")
    elif frac_extreme < 0.1:
        status, head = "amber", t(lang, "有些人 PS 很極端——IPTW 權重會很大，考慮截斷或改用重疊權重。",
                                  "Some people have extreme PS — IPTW weights blow up; consider trimming or overlap weights.")
    else:
        status, head = "red", t(lang, "正性受威脅——不少人幾乎一定／一定不接種，這些區域的效果其實估不出來。",
                                "Positivity threatened — many people almost surely (don't) get treated; the effect is not really estimable there.")
    return {
        "id": "C2",
        "title": t(lang, "正性／重疊：每種人都「有可能」接種也有可能不接種嗎？（可檢驗）",
                   "Positivity / overlap: can every kind of person plausibly be treated or untreated? (testable)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "<b>正性（positivity）</b>要求：在每個共變項組合下，接種與不接種的機率都<b>大於 0</b>。若某種人<b>幾乎一定</b>會接種"
            "（PS≈1）或<b>幾乎一定</b>不會（PS≈0），就沒有對照可比，IPTW 權重會爆大、估計很不穩。可檢的指標是<b>PS 分布的重疊</b>"
            "與極端 PS 的比例。修法：截斷極端權重、改用<b>重疊權重（ATO）</b>（自動把權重壓在共同支持區），或限制在重疊區。",
            "<b>Positivity</b> requires that, within every covariate pattern, the probability of being treated and untreated is "
            "<b>above 0</b>. If some people are <b>almost certain</b> to be treated (PS≈1) or untreated (PS≈0), there is no comparison, "
            "IPTW weights explode, and the estimate is unstable. The testable metric is the <b>overlap of the PS distributions</b> and "
            "the fraction with extreme PS. Fixes: trim extreme weights, use <b>overlap weights (ATO)</b> (which down-weight the tails "
            "automatically), or restrict to the region of common support.",
        ),
        "term": t(lang, "專有名詞：正性（positivity）；共同支持（common support）；極端權重；重疊權重（overlap weights）。",
                  "Term: positivity; common support; extreme weights; overlap weights."),
        "metrics": [
            {"name": t(lang, "PS 極端（&lt;0.05 或 &gt;0.95）比例", "fraction with extreme PS (&lt;0.05 or &gt;0.95)"),
             "value": f"{frac_extreme*100:.1f}%",
             "note": t(lang, "越低越好；高代表正性受威脅", "lower is better; high means positivity is threatened")},
        ],
    }


def _c3_balance(res, lang="zh"):
    smd = res.get("smd_after", 1.0)
    if smd < 0.1:
        status, head = "green", t(lang, "加權後共變項平衡良好（SMD &lt; 0.1）——PS 模型把可測混淆清掉了。",
                                  "Good covariate balance after weighting (SMD &lt; 0.1) — the PS model removed the measured confounding.")
    elif smd < 0.2:
        status, head = "amber", t(lang, "平衡尚可但不完美——可加交互／樣條重配 PS 模型，再檢查一次。",
                                  "Balance is acceptable but imperfect — refit the PS with interactions/splines and recheck.")
    else:
        status, head = "red", t(lang, "加權後仍不平衡——PS 模型設定錯誤，結果別信，先把平衡做好。",
                                "Still imbalanced after weighting — the PS model is mis-specified; don't trust the estimate until balance is fixed.")
    return {
        "id": "C3",
        "title": t(lang, "PS 模型設定正確嗎？用加權後的共變項平衡來檢查（可檢驗）",
                   "Is the PS model correct? check via covariate balance after weighting (testable)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "PS 的<b>檢驗標準不是預測得準不準，而是平衡好不好</b>。配完 PS、做完配對／加權後，把每個共變項在兩組的<b>標準化差異"
            "（SMD）</b>算出來：理想是<b>都 &lt; 0.1</b>。若還大，代表 PS 模型漏了某個結構（交互、非線性），就回去把它加進去重配——"
            "這是 PS 的迭代精神：<b>調模型 → 看平衡 → 再調</b>，直到平衡為止（不是看 AUC）。",
            "The <b>yardstick for a PS model is balance, not prediction accuracy</b>. After fitting the PS and matching/weighting, "
            "compute the <b>standardized mean difference (SMD)</b> of each covariate between arms: ideally <b>all &lt; 0.1</b>. If some "
            "remain large, the PS model missed structure (an interaction, a non-linearity) — add it and refit. This is the iterative "
            "spirit of PS: <b>tune the model → check balance → tune again</b> until balanced (not chase AUC).",
        ),
        "term": t(lang, "專有名詞：標準化差異（SMD）；平衡診斷；PS 模型設定。",
                  "Term: standardized mean difference (SMD); balance diagnostics; PS model specification."),
        "metrics": [
            {"name": t(lang, "加權後最大 SMD", "max SMD after weighting"), "value": f"{smd:.3f}",
             "note": t(lang, "目標 &lt; 0.1", "target &lt; 0.1")},
            {"name": t(lang, "加權前 SMD", "SMD before weighting"), "value": f"{res.get('smd_before', 0):.3f}",
             "note": t(lang, "加權前的失衡（對照）", "imbalance before weighting (for contrast)")},
        ],
    }


def _c4_selection(lang="zh"):
    return {
        "id": "C4",
        "title": t(lang, "共變項選對了嗎？放混淆與結果風險因子、別放工具變數（不可純檢驗）",
                   "Are the right covariates in the PS? include confounders & risk factors, exclude instruments (largely untestable)"),
        "status": "info",
        "headline": t(lang, "PS 該放的是「<b>影響結果</b>的變項」（混淆＋純結果風險因子）；放<b>只影響治療</b>的工具反而增加變異。",
                      "Put variables that <b>affect the outcome</b> in the PS (confounders + pure outcome risk factors); putting in <b>treatment-only</b> instruments hurts."),
        "plain": t(
            lang,
            "哪些 X 該進 PS 模型？指引（Brookhart 等 2006）：(1) <b>混淆因子</b>（同時影響治療與結果）一定要放；(2) <b>只影響結果</b>"
            "的風險因子<b>也該放</b>（提升效率、降變異）；(3) <b>只影響治療、不影響結果</b>的『工具變數型』變項<b>不要放</b>——放了會"
            "放大變異、甚至放大殘餘偏誤（Z-bias）。要納入<b>暴露前</b>測得的變項，<b>絕不可</b>放治療<b>之後</b>的中介或結果。"
            "這條主要靠領域知識與 DAG 判斷，無法純由資料決定。",
            "Which X belong in the PS? Guidance (Brookhart et al. 2006): (1) <b>confounders</b> (affect both treatment and outcome) "
            "must go in; (2) <b>pure outcome risk factors</b> (affect only the outcome) <b>should also go in</b> (more efficient, less "
            "variance); (3) <b>instrument-like</b> variables that affect <b>only treatment, not the outcome</b> should be <b>left out</b> "
            "— including them inflates variance and can amplify residual bias (Z-bias). Use only variables measured <b>before</b> "
            "exposure; <b>never</b> a post-treatment mediator or the outcome. This rests on domain knowledge and a DAG, not data alone.",
        ),
        "term": t(lang, "專有名詞：混淆因子 vs 工具變數；Z-bias；變數選擇（Brookhart 2006）。",
                  "Term: confounder vs instrument; Z-bias; variable selection (Brookhart 2006)."),
        "metrics": [],
    }


def _c5_ess(ess, n, lang="zh"):
    ratio = ess / max(n, 1)
    if ratio >= 0.5:
        status, head = "green", t(lang, "有效樣本充足——加權沒有把資訊耗掉太多。",
                                  "Ample effective sample — weighting hasn't thrown away much information.")
    elif ratio >= 0.25:
        status, head = "amber", t(lang, "有效樣本中等——少數大權重在主導，信賴區間偏寬。",
                                  "Moderate effective sample — a few large weights dominate; the CI is wider.")
    else:
        status, head = "red", t(lang, "有效樣本太小——權重極不均，估計被少數人主導、很不穩。",
                                "Effective sample too small — weights are very uneven; a few people dominate and the estimate is unstable.")
    return {
        "id": "C5",
        "title": t(lang, "加權後還剩多少『有效樣本』？（可檢驗）",
                   "How much effective sample is left after weighting? (testable)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "IPTW 給某些人很大的權重，等於讓<b>少數人主導</b>整個估計。<b>有效樣本數（ESS = (Σw)² / Σw²）</b>衡量加權後實際剩多少"
            "資訊；ESS 遠小於樣本數，代表權重很不均、信賴區間會變寬、結果不穩。修法：截斷／穩定化權重，或改用<b>重疊權重</b>"
            "（權重有界、ESS 較高）。",
            "IPTW gives some people very large weights, effectively letting <b>a few people dominate</b> the estimate. The <b>effective "
            "sample size (ESS = (Σw)² / Σw²)</b> measures how much information is really left after weighting; an ESS far below the "
            "sample size means very uneven weights, a wider CI, and an unstable result. Fixes: trim/stabilize the weights, or use "
            "<b>overlap weights</b> (bounded weights, higher ESS).",
        ),
        "term": t(lang, "專有名詞：有效樣本數（ESS）；權重穩定化／截斷；重疊權重。",
                  "Term: effective sample size (ESS); weight stabilization/trimming; overlap weights."),
        "metrics": [
            {"name": t(lang, "IPTW 有效樣本數 ESS", "IPTW effective sample size (ESS)"),
             "value": f"{ess:.0f} / {n}",
             "note": t(lang, "越接近總樣本越好", "the closer to the total sample, the better")},
        ],
    }
