"""PERR assumption checks P1-P5 — following Yu et al. 2012, van Aalst et al. 2021,
Cheung et al. 2024. Most of PERR's assumptions are untestable from data alone, so we
are honest (blue 'info' cards) and point to the checks that the literature DOES offer
(self-controlled case series, dynamic random-intercept modelling, scale sensitivity,
enough prior events).

Each check returns {id, title, status, headline, plain, term, metrics:[...]};
run_dashboard returns {"checks": [...]}.
"""
from __future__ import annotations

import numpy as np

import perr_core
from i18n import t


def run_dashboard(df, group="group", events_prior="events_prior", pt_prior="pt_prior",
                  events_post="events_post", pt_post="pt_post", lang="zh"):
    res = perr_core.full_perr(df, group, events_prior, pt_prior, events_post, pt_post, lang=lang)
    ev = res["events"]
    checks = [
        _p1_time_invariant(res, lang),
        _p2_no_anticipation(lang),
        _p3_event_dependent_treatment(lang),
        _p4_event_dependent_dropout(lang),
        _p5_enough_prior_events(ev, lang),
    ]
    return {"checks": checks}


def _p1_time_invariant(res, lang="zh"):
    rr_prior = res["rr_prior"]; perr = res["perr"]; naive = res["naive_rr"]
    metrics = [
        {"name": t(lang, "事前期率比（混淆的指紋）", "Prior-period rate ratio (the confounding fingerprint)"),
         "value": round(rr_prior, 2),
         "note": t(lang, "偏離 1 越多，代表混淆越強、PERR 校正越多",
                   "the further from 1, the stronger the confounding PERR removes")},
        {"name": t(lang, "PERR 與天真事後率比的差距", "Gap between PERR and the naive post ratio"),
         "value": f"{perr:.2f} vs {naive:.2f}",
         "note": t(lang, "差距大代表校正幅度大——也代表更依賴這條假設",
                   "a large gap means a large correction — and heavier reliance on this assumption")},
    ]
    return {
        "id": "P1",
        "title": t(lang, "混淆是「時間不變且乘法」的嗎？（關鍵、不可直接檢驗）",
                   "Is the confounding time-invariant and multiplicative? (key, not directly testable)"),
        "status": "info",
        "headline": t(lang, "這是 PERR 能成立的核心假設，無法用資料證明，要靠領域知識＋敏感度分析。",
                      "This is PERR's core assumption; it cannot be proven from data — use domain knowledge + sensitivity analysis."),
        "plain": t(
            lang,
            "PERR 之所以能消掉看不見的混淆，靠的是『混淆對事件率的相對影響，在事前期與事後期一樣（時間不變），"
            "而且作用在乘法尺度上』。只要這成立，事前期率比就完整捕捉了混淆，除掉它就還原真效果。但如果混淆"
            "<b>隨時間改變</b>（例如體弱程度在用藥後自然好轉），或其實作用在<b>加法尺度</b>，PERR 就會有偏。"
            "資料無法直接證明它，但你可以：①用 ⑤ 的 PERR vs PERD 做尺度敏感度；②檢查可測的時間變動代理變項；"
            "③與隨機試驗（若有）對照。",
            "PERR removes unseen confounding by assuming the confounder's RELATIVE effect on the event rate is the SAME "
            "in the prior and post periods (time-invariant) and acts on a MULTIPLICATIVE scale. When that holds, the "
            "prior-period rate ratio fully captures the confounding and dividing it out recovers the truth. But if the "
            "confounding <b>changes over time</b> (e.g. frailty resolves after treatment) or acts on an <b>additive</b> "
            "scale, PERR is biased. The data cannot prove it, but you can: ① run the PERR-vs-PERD scale sensitivity in "
            "⑤; ② inspect measured time-varying proxies; ③ benchmark against a randomized trial where one exists.",
        ),
        "term": t(lang, "專有名詞：時間不變、乘法尺度的混淆（time-invariant, multiplicative confounding）；尺度敏感度（PERR vs PERD）。",
                  "Term: time-invariant, multiplicative confounding; scale sensitivity (PERR vs PERD)."),
        "metrics": metrics,
    }


def _p2_no_anticipation(lang="zh"):
    return {
        "id": "P2",
        "title": t(lang, "事前期真的「乾淨」嗎？（無預期、尚未用藥）",
                   "Is the prior period truly clean? (no anticipation, not yet treated)"),
        "status": "info",
        "headline": t(lang, "事前期必須在用藥前、且未受未來用藥影響，靠研究設計與領域知識保證。",
                      "The prior period must precede treatment and be unaffected by future treatment — ensured by design and domain knowledge."),
        "plain": t(
            lang,
            "PERR 用事前期率比當『純混淆』的基準，前提是事前期裡<b>兩組都還沒用藥</b>，而且大家沒有因為"
            "『知道之後會用藥』而提前改變行為。如果事前期其實已混進部分用藥、或有預期效應，事前期率比就"
            "不再是乾淨的混淆指紋，PERR 會被汙染。做法：明確定義一段確實在用藥之前的窗、排除過渡期。",
            "PERR uses the prior-period rate ratio as the 'pure confounding' benchmark, which requires that <b>neither "
            "group is treated yet</b> in the prior period and that people did not change behaviour in anticipation of "
            "future treatment. If the prior period is contaminated by some treatment or anticipation effects, the "
            "prior-period ratio is no longer a clean confounding fingerprint and PERR is contaminated. Remedy: define a "
            "prior window strictly before treatment and exclude any transition period.",
        ),
        "term": t(lang, "專有名詞：無預期假設（no-anticipation）；乾淨的事前期（clean prior window）。",
                  "Term: no-anticipation assumption; clean prior window."),
        "metrics": [],
    }


def _p3_event_dependent_treatment(lang="zh"):
    return {
        "id": "P3",
        "title": t(lang, "事前發生事件，會不會改變之後是否用藥？（可用 SCCS 檢查）",
                   "Does having a prior event change later treatment? (checkable by SCCS)"),
        "status": "info",
        "headline": t(lang, "若『事前事件』本身改變了用藥機率，會破壞 PERR；可用自我對照病例系列（SCCS）評估。",
                      "If a prior event itself changes the treatment probability it breaks PERR; assess with a self-controlled case series (SCCS)."),
        "plain": t(
            lang,
            "PERR 假設『有沒有在事前期發生事件』不會（在配對/分層之外）改變一個人之後會不會用藥。"
            "但現實中，剛發生過事件的人可能更容易被開藥（或更不容易）。Cheung 等人（2024）提出用"
            "<b>自我對照病例系列（SCCS）</b>把角色反過來——看『事前事件』是否預測之後的用藥——來實際檢查這一條。",
            "PERR assumes that whether an event occurred in the prior period does not (beyond matching/stratification) "
            "change whether a person is later treated. In reality, someone who just had an event may be more (or less) "
            "likely to be put on the drug. Cheung et al. (2024) propose using a <b>self-controlled case series (SCCS)</b> "
            "— reversing the roles to see whether prior events predict later treatment — to actually check this.",
        ),
        "term": t(lang, "專有名詞：事件相依的處置分派（event-dependent treatment）；自我對照病例系列（SCCS）。",
                  "Term: event-dependent treatment assignment; self-controlled case series (SCCS)."),
        "metrics": [],
    }


def _p4_event_dependent_dropout(lang="zh"):
    return {
        "id": "P4",
        "title": t(lang, "事前事件會不會造成之後退出／死亡？（可用 DRIM 檢查；改用 PERR-Comp）",
                   "Do prior events cause later dropout/death? (checkable by DRIM; use PERR-Comp)"),
        "status": "info",
        "headline": t(lang, "若事前事件直接導致事後期退出或死亡，PERR 會有偏；可用動態隨機截距模型檢查。",
                      "If prior events directly cause post-period dropout/death, PERR is biased; check with a dynamic random-intercept model."),
        "plain": t(
            lang,
            "如果在事前期發生過事件的人，更容易在事後期退出或死亡（選擇性流失），高風險的人被選掉，"
            "事後期率比就被扭曲。Cheung & Ma（2025）指出：只要退出不是<b>直接由事前事件造成</b>，用"
            "『完成者』一致定義的 <b>PERR-Comp</b> 仍可近似無偏；並可用<b>動態隨機截距模型（DRIM）</b>"
            "檢查『早期事件是否預測晚期事件/流失』。",
            "If people who had a prior event are more likely to drop out or die in the post period (selective attrition), "
            "high-risk people are removed and the post-period ratio is distorted. Cheung & Ma (2025) show that as long as "
            "dropout is not <b>directly caused by the prior event</b>, a completers-consistent <b>PERR-Comp</b> stays "
            "approximately unbiased; a <b>dynamic random-intercept model (DRIM)</b> can check whether earlier events "
            "predict later events/attrition.",
        ),
        "term": t(lang, "專有名詞：事件相依的流失（event-dependent dropout）；PERR-Comp（完成者版）；DRIM。",
                  "Term: event-dependent dropout; PERR-Comp (completers version); DRIM."),
        "metrics": [],
    }


def _p5_enough_prior_events(ev, lang="zh"):
    nmin = int(min(ev["treated_prior"], ev["control_prior"]))
    metrics = [
        {"name": t(lang, "事前期事件數（兩組取較少者）", "Prior-period events (the smaller group)"),
         "value": nmin,
         "note": t(lang, "太少會讓事前期率比很不穩、PERR 區間爆寬",
                   "too few makes the prior ratio unstable and the PERR interval blow up")},
    ]
    if nmin >= 30:
        status = "green"
        head = t(lang, "事前期事件數足夠，事前期率比估得穩，PERR 區間可信。",
                 "Plenty of prior-period events — the prior ratio is stable and the PERR interval is trustworthy.")
    elif nmin >= 10:
        status = "amber"
        head = t(lang, "事前期事件偏少，事前期率比較不穩，PERR 區間會偏寬。",
                 "Few prior-period events — the prior ratio is unstable and the PERR interval is wide.")
    else:
        status = "red"
        head = t(lang, "事前期事件太少，事前期率比極不穩，PERR 不可靠。",
                 "Too few prior-period events — the prior ratio is very unstable and PERR is unreliable.")
    return {
        "id": "P5",
        "title": t(lang, "事前期的事件數夠多嗎？", "Are there enough prior-period events?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "PERR 要除以『事前期率比』。如果事前期事件很少（例如某組只有幾件），這個分母本身估得很不穩，"
            "一除下去，整個 PERR 的不確定性就被放大、信賴區間變得很寬。延長事前觀察窗、或選事件較常見的"
            "結果，都能改善。",
            "PERR divides by the prior-period rate ratio. If prior events are few (say only a handful in one group), that "
            "denominator is itself very unstable, and dividing inflates PERR's uncertainty — the confidence interval "
            "blows up. A longer prior observation window, or a more common outcome, both help.",
        ),
        "term": t(lang, "專有名詞：事前期事件數／估計穩定性（prior-event count, stability）。",
                  "Term: prior-event count / estimate stability."),
        "metrics": metrics,
    }
