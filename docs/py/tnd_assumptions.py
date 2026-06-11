"""Test-Negative Design assumption checks C1–C5. The TND removes confounding by health-care-
seeking, but it rests on conditions that are mostly design judgement (blue): an accurate test,
the vaccine not affecting the non-target illness, equal care-seeking given symptoms, and a
shared symptom profile. The testable support is having enough tested cases and controls.

Each check returns {id, title, status, headline, plain, term, metrics:[...]};
run_dashboard returns {"checks": [...]}.
"""
from __future__ import annotations

import numpy as np

from i18n import t


def run_dashboard(df, lang="zh"):
    tested = np.asarray(df["tested"], float)
    case = np.asarray(df["case"], float)
    m = tested == 1
    n_case = int(np.sum(case[m] == 1)); n_ctrl = int(np.sum(case[m] == 0))
    checks = [
        _c1_test(lang),
        _c2_nontarget(lang),
        _c3_careseeking(lang),
        _c4_symptoms(lang),
        _c5_counts(n_case, n_ctrl, lang),
    ]
    return {"checks": checks}


def _c1_test(lang="zh"):
    return {
        "id": "C1",
        "title": t(lang, "檢驗準確（高敏感度／特異度，否則目標與非目標病原會錯分）",
                   "Accurate test (high sensitivity / specificity, else target vs non-target are misclassified)"),
        "status": "info",
        "headline": t(lang, "TND 靠檢驗把 case／control 分開；檢驗不準會把效果<b>往虛無拉</b>。",
                      "TND relies on the test to split cases from controls; an inaccurate test <b>biases VE toward the null</b>."),
        "plain": t(
            lang,
            "TND 用<b>檢驗結果</b>定義 case（目標病原陽性）與 control（陰性）。若檢驗<b>敏感度／特異度不夠高</b>，會把真正的"
            "case 漏判為 control（或反之），造成<b>分類錯誤</b>，通常把 VE <b>往虛無（低估保護）</b>拉。實務：用高品質的"
            "PCR／分子檢驗、明確的檢驗適應症，並做不同檢驗定義的敏感度分析。",
            "TND uses the <b>test result</b> to define cases (target-positive) and controls (negative). If the test's <b>sensitivity / "
            "specificity is not high enough</b>, true cases are misclassified as controls (or vice versa), and this <b>misclassification</b> "
            "usually biases VE <b>toward the null</b>. In practice: use a high-quality PCR / molecular assay, clear testing indications, "
            "and run sensitivity analyses across test definitions.",
        ),
        "term": t(lang, "專有名詞：敏感度／特異度；分類錯誤；往虛無的偏誤。",
                  "Term: sensitivity / specificity; misclassification; bias toward the null."),
        "metrics": [],
    }


def _c2_nontarget(lang="zh"):
    return {
        "id": "C2",
        "title": t(lang, "疫苗不影響「非目標病原」（即檢驗陰性的對照疾病）——關鍵、不可檢驗",
                   "The vaccine does not affect the non-target illness (the test-negative control disease) — key, untestable"),
        "status": "info",
        "headline": t(lang, "若疫苗也改變了對照疾病的就醫／發生，<b>對照被污染</b>，VE 會偏。",
                      "If the vaccine also changes the control illness (its occurrence or care-seeking), the <b>controls are contaminated</b> and VE is biased."),
        "plain": t(
            lang,
            "TND 的對照是「<b>檢驗陰性</b>（得到其他病原）」的人。核心假設：<b>疫苗對這些非目標病原沒有影響</b>。若疫苗"
            "（或接種者的行為）也改變了非目標疾病的發生或就醫，對照組就被污染、OR 失準。例：流感疫苗若改變了 RSV 的"
            "就醫，TND 的 control 就不中性。這條<b>無法用資料證明</b>，要靠生物學與領域知識（Jackson & Nelson 2013）。",
            "TND controls are people who tested <b>negative</b> (a different pathogen). The core assumption is that the <b>vaccine has no "
            "effect on those non-target pathogens</b>. If the vaccine (or vaccinees' behaviour) also changes the occurrence or "
            "care-seeking for the control illness, the controls are contaminated and the OR is off. E.g. if an influenza vaccine changed "
            "RSV care-seeking, the TND control would not be neutral. This <b>cannot be proven from data</b> — it rests on biology and "
            "domain knowledge (Jackson & Nelson 2013).",
        ),
        "term": t(lang, "專有名詞：對照疾病中性；非目標病原；對照污染。",
                  "Term: neutral control disease; non-target pathogen; control contamination."),
        "metrics": [],
    }


def _c3_careseeking(lang="zh"):
    return {
        "id": "C3",
        "title": t(lang, "檢驗者中，接種與就醫無差別關聯（TND 的命門，不可純檢驗）",
                   "Among the tested, no differential association between vaccination and care-seeking (the crux, largely untestable)"),
        "status": "info",
        "headline": t(lang, "TND 之所以去掉就醫傾向，靠的是「case 與 control <b>同樣都來檢驗</b>」；若接種者就醫門檻不同，仍會偏。",
                      "TND removes care-seeking because cases and controls were <b>all tested</b>; if vaccinees have a different testing threshold, bias remains."),
        "plain": t(
            lang,
            "TND 把<b>就醫傾向</b>條件掉的關鍵：case 與 control 都是「<b>因症狀就醫並接受檢驗</b>」的人。前提是——<b>在有症狀</b>"
            "之下，接種與否不會改變「來不來檢驗」。若接種者（例如更注意健康）症狀輕也更會來驗、或門檻不同，這層條件就破"
            "了、殘留偏誤。這也是為什麼 TND 限定在「<b>有相同就醫管道</b>」的人。可用因果 TND（IPW／g-comp，見 ⑤）對<b>已測</b>"
            "的就醫相關變項做調整。",
            "The key to TND removing <b>care-seeking</b> is that cases and controls are both people who <b>sought care and were tested</b>. "
            "The premise is that, <b>given symptoms</b>, vaccination does not change whether you get tested. If vaccinees (say, more "
            "health-conscious) seek testing at a lower threshold, this breaks and bias remains. That's why TND restricts to people with "
            "the <b>same access to testing</b>. A causal TND (IPW / g-computation, see ⑤) can adjust for <b>measured</b> care-seeking "
            "correlates.",
        ),
        "term": t(lang, "專有名詞：就醫傾向混淆；檢驗門檻；條件在「有檢驗」上。",
                  "Term: care-seeking confounding; testing threshold; conditioning on being tested."),
        "metrics": [],
    }


def _c4_symptoms(lang="zh"):
    return {
        "id": "C4",
        "title": t(lang, "目標與非目標疾病的症狀／就醫驅力相似（不可純檢驗）",
                   "Target and non-target illnesses have similar symptoms / drivers of care-seeking (largely untestable)"),
        "status": "info",
        "headline": t(lang, "對照要和 case「<b>因為一樣的症狀</b>」來檢驗，就醫傾向才會在相除時抵銷。",
                      "Controls must be tested <b>for the same kind of symptoms</b> as cases, so care-seeking cancels in the ratio."),
        "plain": t(
            lang,
            "為了讓就醫傾向在 case／control 間抵銷，兩者最好是<b>因相似的臨床症候群</b>（如急性呼吸道感染）來就醫的。若對照"
            "疾病的症狀更輕／更重、就醫驅力不同，殘留偏誤就回來了。實務：用<b>共同的納入症狀定義</b>（例如 ILI／ARI），"
            "讓 case 與 control 來自同一就醫情境。",
            "For care-seeking to cancel between cases and controls, both should be tested <b>for a similar clinical syndrome</b> (e.g. "
            "acute respiratory infection). If the control illness is milder/severer with different care-seeking drivers, residual bias "
            "returns. In practice: use a <b>common eligibility symptom definition</b> (e.g. ILI / ARI) so cases and controls arise from "
            "the same care-seeking context.",
        ),
        "term": t(lang, "專有名詞：共同症狀定義（ILI／ARI）；就醫情境；可比性。",
                  "Term: common symptom definition (ILI / ARI); care-seeking context; comparability."),
        "metrics": [],
    }


def _c5_counts(n_case, n_ctrl, lang="zh"):
    m = min(n_case, n_ctrl)
    if m >= 200:
        status, head = "green", t(lang, "檢驗者中 case 與 control 數量充足，OR 估計穩定。",
                                  "Plenty of cases and controls among the tested; the OR is stable.")
    elif m >= 60:
        status, head = "amber", t(lang, "某一組人數偏少——VE 的信賴區間會偏寬。",
                                  "One arm is thin — the VE confidence interval will be wide.")
    else:
        status, head = "red", t(lang, "case 或 control 太少——OR 很不穩、VE 估不準。",
                                "Too few cases or controls — the OR is unstable and VE imprecise.")
    return {
        "id": "C5",
        "title": t(lang, "檢驗者中有足夠的 case 與 control 嗎？（可檢驗）",
                   "Enough cases and controls among the tested? (testable)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "TND 的精度全看「<b>有來檢驗</b>」的人裡有多少 case（目標陽性）與 control（陰性）。樣本太小，勝算比與 VE 都會"
            "很不穩、信賴區間很寬。這個指標看兩組的最小人數。",
            "TND precision depends entirely on how many cases (target-positive) and controls (negative) there are <b>among the tested</b>. "
            "Too few and the odds ratio and VE are unstable with wide intervals. This metric is the smaller of the two counts.",
        ),
        "term": t(lang, "專有名詞：檢驗者樣本；勝算比精度；信賴區間寬度。",
                  "Term: tested sample; odds-ratio precision; confidence-interval width."),
        "metrics": [
            {"name": t(lang, "目標陽性 case 數", "target-positive cases"), "value": str(n_case), "note": ""},
            {"name": t(lang, "檢驗陰性 control 數", "test-negative controls"), "value": str(n_ctrl), "note": ""},
        ],
    }
