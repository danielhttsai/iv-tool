"""IV assumption checks A1-A4b — port of the framework in Homayra et al. 2024.

Design principle for the text: **白話文優先**。每一項先用日常語言說明我們在檢查什麼、
結果代表什麼,專有名詞一律放到最後的 `term` 欄位,並附上白話的專業解釋。

Each check returns:
    {
        "id": "A1",
        "title":   "<白話標題>",
        "status":  "green" | "amber" | "red" | "info",
        "headline":"<一句白話結論>",
        "plain":   "<我們在檢查什麼、為什麼重要,全用白話>",
        "term":    "<最後才出現的專有名詞 + 白話解釋>",
        "metrics": [{"name": "<白話名稱>", "value": ..., "note": "<可含術語>"}],
    }
"""
from __future__ import annotations

import numpy as np

import iv_core
from i18n import t


def _binary(series) -> bool:
    vals = set(np.unique(np.asarray(series)))
    return vals.issubset({0, 1})


# ---------------------------------------------------------------------------
# A1 — Instrument strength (relevance)
# ---------------------------------------------------------------------------
def check_a1_strength(df, A, Z, lang="zh"):
    fs = iv_core.first_stage(df, A, Z)
    f = fs["f_stat"]
    pct = fs["coef"] * 100
    metrics = [
        {"name": t(lang, "有多少人被這個外力推動", "How many people the nudge moves"),
         "value": t(lang, f"約 {pct:.1f}%", f"~{pct:.1f}%"),
         "note": t(lang, "被推動而改變行為的人(complier)的比例",
                   "share of people (compliers) nudged into changing behaviour")},
        {"name": t(lang, "外力夠不夠力的分數", "Strength score of the nudge"),
         "value": round(f, 1),
         "note": t(lang, "統計上叫「第一階段 F 統計量」,大於 10 算夠力",
                   "the first-stage F-statistic; above 10 counts as strong")},
    ]
    if _binary(df[A]):
        mcf = _mcfadden_r2(df, A, Z)
        metrics.append({"name": t(lang, "外力對「要不要做」的解釋力",
                                  "Nudge's explanatory power for the choice"),
                        "value": round(mcf, 4),
                        "note": t(lang, "統計上叫 McFadden 偽 R²", "McFadden pseudo-R²")})
    if f >= 10:
        status = "green"
        head = t(lang, "這個外力夠力,有推動足夠多人改變,後面的分析可以放心做。",
                 "The nudge is strong: it moves enough people, so the rest of the "
                 "analysis is on solid footing.")
    elif f >= 5:
        status = "amber"
        head = t(lang, "這個外力有點弱,推動的人不夠多,結果要保守看待。",
                 "The nudge is a bit weak: it moves too few people, so read the "
                 "results cautiously.")
    else:
        status = "red"
        head = t(lang, "這個外力太弱了,幾乎沒推動什麼人,算出來的結果很可能不可靠。",
                 "The nudge is too weak: it barely moves anyone, so the resulting "
                 "estimates are likely unreliable.")
    return {
        "id": "A1",
        "title": t(lang, "A1・這個「外力」夠不夠力?", "A1 · Is the nudge strong enough?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            ("工具變數法靠一個「外力」(這裡是有沒有收到衛生單位的免費接種提醒)去推動人們改變行為。"
             "如果這個外力其實幾乎沒推動任何人,那我們等於拿一個沒作用的東西在做推論,"
             "算出來的因果效應就會非常不穩定、忽大忽小。所以第一步要確認:"
             "這個外力真的有讓夠多人去接種嗎?"),
            ("The instrumental-variable method relies on a nudge (here, whether you "
             "received the health authority's free vaccination reminder) to push "
             "people into changing behaviour. If that nudge barely moves anyone, we "
             "are drawing conclusions from something with no real effect, and the "
             "estimated causal effect becomes very unstable. So the first step is to "
             "confirm: did this nudge really get enough people vaccinated?")),
        "term": t(
            lang,
            ("📖 專有名詞:這一步在檢查工具的「相關性 / 強度」。常用「第一階段 F 統計量」當分數,"
             "經驗法則是大於 10 就算夠強、低於 10 稱為「弱工具(weak instrument)」。"
             "當處置是「有/沒有」這種二元時,也可看 McFadden 偽 R²,代表外力對行為的解釋力。"),
            ("📖 Terminology: this step checks the instrument's relevance / strength. "
             "The usual score is the first-stage F-statistic; a rule of thumb is that "
             "above 10 is strong, while below 10 is a \"weak instrument\". When the "
             "treatment is binary you can also look at the McFadden pseudo-R², which "
             "captures how much the nudge explains the behaviour.")),
        "metrics": metrics,
    }


def _mcfadden_r2(df, A, Z):
    y = np.asarray(df[A], dtype=float)
    X = np.column_stack([np.ones(len(df)), np.asarray(df[Z], dtype=float)])
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        grad = X.T @ (y - p)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ beta
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    pbar = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    ll0 = np.sum(y * np.log(pbar) + (1 - y) * np.log(1 - pbar))
    return float(1 - ll / ll0) if ll0 != 0 else float("nan")


# ---------------------------------------------------------------------------
# A2 — Exclusion restriction (not statistically testable)
# ---------------------------------------------------------------------------
def check_a2_exclusion(df, Y, A, Z, lang="zh"):
    return {
        "id": "A2",
        "title": t(lang, "A2・這個外力是不是只走「一條路」?",
                   "A2 · Does the nudge travel only one path?"),
        "status": "info",
        "headline": t(
            lang,
            "這題資料沒辦法正面證明,要靠你對這個領域的了解來判斷;但下面那道「抓包測試」會試著從資料戳破它。",
            "Data cannot prove this one directly — it takes your domain knowledge to "
            "judge; but the \"gotcha test\" below tries to break it using the data."),
        "plain": t(
            lang,
            ("我們需要這個外力(免費接種提醒)『只透過』讓人去接種疫苗,來影響健康;"
             "不能還偷偷走別條路。舉例來說,如果收到接種提醒的社區剛好同時加發了健康補助、"
             "或醫療資源也比較多,那健康變好就不全是疫苗的功勞——外力走了別條路,結論就會被汙染。"
             "麻煩的是:這件事沒辦法用手上的資料直接驗證,必須靠常識和專業知識去論證,"
             "確認這個外力真的沒有別的影響途徑。"),
            ("We need the nudge (the free vaccination reminder) to affect health "
             "*only through* getting people vaccinated — it must not secretly take "
             "another path. For example, if the reminded communities also happened to "
             "receive extra health subsidies or more medical resources, then better "
             "health is not purely the vaccine's doing: the nudge took another path "
             "and the conclusion is contaminated. The catch is that this cannot be "
             "verified directly from the data on hand; it has to be argued from common "
             "sense and domain knowledge that the nudge truly has no other route.")),
        "term": t(
            lang,
            ("📖 專有名詞:這個假設叫「排除限制(exclusion restriction)」。"
             "因為無法用統計檢定,實務上的輔助做法是:找一個理論上『不該被外力影響』的結果當"
             "「負對照(negative control)」來檢查;或做「敏感度分析」,評估萬一假設稍微被違反、結論會不會翻盤。"),
            ("📖 Terminology: this assumption is the \"exclusion restriction\". Because "
             "it cannot be tested statistically, common supporting tactics are to use "
             "an outcome that should *not* be affected by the nudge as a \"negative "
             "control\", or to run a \"sensitivity analysis\" estimating whether the "
             "conclusion would flip if the assumption were slightly violated.")),
        "metrics": [
            {"name": t(lang, "能不能用資料證明?", "Provable from data?"),
             "value": t(lang, "不能", "No"),
             "note": t(lang, "要靠專業判斷", "requires domain judgement")},
            {"name": t(lang, "建議的輔助做法", "Suggested supporting checks"),
             "value": t(lang, "負對照 / 敏感度分析", "negative control / sensitivity analysis"),
             "note": t(lang, "間接佐證", "indirect evidence")},
        ],
    }


# ---------------------------------------------------------------------------
# A3 — Independence / exchangeability (instrument vs covariates balance)
# ---------------------------------------------------------------------------
def check_a3_independence(df, Z, covariates, A=None, lang="zh"):
    rows = []
    max_smd = 0.0
    worst_cov = None
    for c in covariates:
        x = np.asarray(df[c], dtype=float)
        g1 = x[np.asarray(df[Z]) == 1]
        g0 = x[np.asarray(df[Z]) == 0]
        if len(g1) == 0 or len(g0) == 0:
            continue
        pooled_sd = np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2)
        smd = abs(g1.mean() - g0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        if smd > max_smd:
            max_smd = smd
            worst_cov = c
        rows.append({
            "name": c,
            "value": round(smd, 3),
            "note": t(lang,
                      f"有外力組均值 {g1.mean():.2f} / 沒有組 {g0.mean():.2f}（差距越小越好）",
                      f"nudged-group mean {g1.mean():.2f} / un-nudged {g0.mean():.2f} "
                      "(smaller gap is better)"),
        })

    # 偏誤放大(bias amplification):一個小小的不平衡,會被「外力推動的力道」倒數放大。
    # 放大倍數 = 1 / |處置在有/沒有外力兩組的盛行率差| = 1 / |第一階段係數|。
    amp_note = None
    if A is not None and _binary(df.get(A, [])) and covariates:
        a = np.asarray(df[A], dtype=float)
        z = np.asarray(df[Z], dtype=float)
        prev_diff = abs(a[z == 1].mean() - a[z == 0].mean()) if (z == 1).any() and (z == 0).any() else 0.0
        if prev_diff > 1e-6:
            amp = 1.0 / prev_diff
            rows.append({
                "name": t(lang,
                          f"⚠ 偏誤放大倍數（最大不平衡：{worst_cov or '—'}）",
                          f"⚠ Bias-amplification factor (largest imbalance: {worst_cov or '—'})"),
                "value": t(lang, f"約 {amp:.0f} 倍", f"~{amp:.0f}×"),
                "note": t(lang,
                          (f"最大不平衡 {max_smd:.3f} 被「外力力道」倒數放大後 ≈ {max_smd*amp:.2f}；"
                           "外力越弱(推動的人越少),同樣的小不平衡造成的偏誤越大"),
                          (f"the largest imbalance {max_smd:.3f}, divided by the nudge's "
                           f"strength, blows up to ≈ {max_smd*amp:.2f}; the weaker the "
                           "nudge (the fewer people moved), the more a small imbalance "
                           "distorts the result")),
            })
            amp_note = max_smd * amp

    if max_smd < 0.1:
        status = "green"
        head = t(lang, "有外力和沒外力的兩群人,其他條件都很接近,看起來像隨機分配,很好。",
                 "The nudged and un-nudged groups look alike on the other "
                 "characteristics — consistent with random assignment. Good.")
    elif max_smd < 0.2:
        status = "amber"
        head = t(lang, "兩群人有些條件不太一樣,要留意這個外力可能不是完全隨機。",
                 "The two groups differ on some characteristics — watch out, the "
                 "nudge may not be fully random.")
    else:
        status = "red"
        head = t(lang, "兩群人有明顯差異,這個外力恐怕不像隨機分配,結果要小心。",
                 "The two groups differ markedly — the nudge probably is not like "
                 "random assignment, so treat the results with care.")
    return {
        "id": "A3",
        "title": t(lang, "A3・有外力 vs 沒外力的兩群人,長得像嗎?",
                   "A3 · Do the nudged and un-nudged groups look alike?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            ("一個好的外力,應該像『抽籤』一樣決定誰收到接種提醒、誰沒有。如果真的像抽籤,"
             "那收到提醒和沒收到提醒的兩群人,除了那個提醒之外,其他條件(年齡、BMI、收入…)"
             "應該都差不多。我們就把這兩群人的各項特徵拿來比一比:如果處處都很接近,"
             "代表這個外力比較可信;如果某些特徵差很多,就要懷疑它其實不是隨機的。"),
            ("A good nudge should decide who gets the reminder and who does not as if "
             "by a *lottery*. If it truly is like a lottery, then apart from the "
             "reminder itself, the two groups should look about the same on everything "
             "else (age, BMI, income…). So we compare the groups characteristic by "
             "characteristic: if they match closely everywhere, the nudge is more "
             "credible; if some characteristics differ a lot, we should suspect it was "
             "not really random.")),
        "term": t(
            lang,
            ("📖 專有名詞:這一步在檢查工具的「獨立性 / 可交換性(independence / exchangeability)」。"
             "比較兩群人差異的數字叫「標準化均值差(SMD)」,通常小於 0.1 就算夠接近。"
             "另外列出的「偏誤放大(bias amplification)」是 Homayra 等人特別提醒的陷阱:"
             "IV 會把共變項的不平衡『除以外力的力道』來放大,所以外力越弱、一點點不平衡就可能造成很大的偏誤。"
             "提醒:這只能比『有量到』的特徵,沒量到的干擾因子無法保證也平衡。"),
            ("📖 Terminology: this step checks the instrument's independence / "
             "exchangeability. The number comparing the two groups is the "
             "\"standardised mean difference (SMD)\"; below 0.1 usually counts as close "
             "enough. The separately listed \"bias amplification\" is a trap "
             "highlighted by Homayra et al.: IV divides any covariate imbalance by the "
             "nudge's strength, so the weaker the nudge, the more a tiny imbalance can "
             "blow up into a large bias. Note: this only compares *measured* "
             "characteristics — unmeasured confounders are not guaranteed to be "
             "balanced.")),
        "metrics": rows or [{"name": t(lang, "(沒有選共變項)", "(no covariates selected)"),
                             "value": "-",
                             "note": t(lang, "請在分析頁選一些共變項",
                                       "pick some covariates on the analysis tab")}],
    }


# ---------------------------------------------------------------------------
# A4a — Monotonicity (no defiers)
# ---------------------------------------------------------------------------
def check_a4a_monotonicity(df, A, Z, lang="zh"):
    fs = iv_core.first_stage(df, A, Z)
    share = fs["coef"]
    if not _binary(df[A]) or not _binary(df[Z]):
        status = "info"
        head = t(lang, "處置或外力不是「有/沒有」這種二元,這題只能粗略近似,請保守看待。",
                 "The treatment or the nudge is not a binary yes/no, so this check is "
                 "only a rough approximation — read it cautiously.")
    elif share > 0:
        status = "green"
        head = t(lang, "大家被推動的方向一致(有外力的人確實更會去做),沒看到唱反調的跡象,很好。",
                 "Everyone is nudged in the same direction (the nudged really are more "
                 "likely to act), with no sign of contrarians. Good.")
    else:
        status = "red"
        head = t(lang, "整體方向是反的,可能有人「唱反調」,這會讓結果不可靠。",
                 "The overall direction is reversed — there may be \"contrarians\", "
                 "which makes the results unreliable.")
    return {
        "id": "A4a",
        "title": t(lang, "A4a・有沒有人「唱反調」?", "A4a · Is anyone a contrarian?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            ("我們假設這個外力對每個人的推力方向都一樣——沒有人會『因為收到了接種提醒,"
             "反而更不去接種』。這種專門唱反調的人如果存在,就會把結果攪亂。"
             "問題是這種人沒辦法直接認出來,所以我們退而求其次:看整體的推動方向是不是一致地"
             "『有外力→更會做這件事』,沒有反過來。"),
            ("We assume the nudge pushes everyone in the same direction — nobody "
             "becomes *less* likely to vaccinate *because* they got the reminder. If "
             "such contrarians exist, they scramble the results. The trouble is they "
             "cannot be identified directly, so we settle for the next best thing: "
             "checking that the overall push is consistently \"nudge → more likely to "
             "act\", never the reverse.")),
        "term": t(
            lang,
            ("📖 專有名詞:這個假設叫「單調性(monotonicity)」,唱反調的人叫「defier」。"
             "因為 defier 看不到,只能用第一階段的方向是否一致為正來間接判斷,"
             "並在不同次群組中確認方向不會反轉。"),
            ("📖 Terminology: this assumption is \"monotonicity\", and the contrarians "
             "are \"defiers\". Because defiers are unobservable, we judge indirectly by "
             "whether the first stage is consistently positive, and confirm the "
             "direction does not flip across subgroups.")),
        "metrics": [
            {"name": t(lang, "被推動的人佔比", "Share of people moved"),
             "value": t(lang, f"約 {share*100:.1f}%", f"~{share*100:.1f}%"),
             "note": t(lang, "第一階段係數", "first-stage coefficient")},
            {"name": t(lang, "整體推動方向", "Overall push direction"),
             "value": t(lang, "一致(正向)", "consistent (positive)") if share > 0
                      else t(lang, "相反(負向)", "reversed (negative)"),
             "note": t(lang, "應該要一致正向", "should be consistently positive")},
        ],
    }


# ---------------------------------------------------------------------------
# A4b — Homogeneity (effect stability — change-in-estimate)
# ---------------------------------------------------------------------------
def check_a4b_homogeneity(df, Y, A, Z, covariates, lang="zh"):
    iv0 = iv_core.iv_2sls(df, Y, A, Z)["estimate"]
    if covariates:
        iv1 = iv_core.iv_2sls(df, Y, A, Z, covariates)["estimate"]
        denom = abs(iv0) if abs(iv0) > 1e-9 else 1.0
        change = abs(iv1 - iv0) / denom
        if change < 0.10:
            status = "green"
            head = t(lang, "第二次把背景條件一起放進去算,答案幾乎沒變,效果看起來蠻穩定的。",
                     "The second pass, with background conditions added, barely changes "
                     "the answer — the effect looks fairly stable.")
        elif change < 0.25:
            status = "amber"
            head = t(lang, "第二次把背景條件一起放進去算,答案有點變動,效果在不同人身上可能不太一樣。",
                     "The second pass, with background conditions added, shifts the "
                     "answer somewhat — the effect may differ across people.")
        else:
            status = "red"
            head = t(lang, "第二次把背景條件一起放進去算,答案變很多,效果在不同人身上差異可能很大。",
                     "The second pass, with background conditions added, changes the "
                     "answer a lot — the effect likely varies widely across people.")
        metrics = [
            {"name": t(lang, "第一次:只算工具變數", "Pass 1: instrument only"),
             "value": round(iv0, 3),
             "note": t(lang, "沒放年齡、BMI、收入等背景條件",
                       "without age, BMI, income and other background conditions")},
            {"name": t(lang, "第二次:再加上背景條件", "Pass 2: add background conditions"),
             "value": round(iv1, 3),
             "note": t(lang, "把年齡、BMI、收入等一起放進同一個模型重算",
                       "re-estimated with age, BMI, income, etc. in the same model")},
            {"name": t(lang, "兩次答案差多少", "How much the two differ"),
             "value": f"{change*100:.1f}%",
             "note": t(lang, "差越小越穩定;術語叫 change-in-estimate",
                       "smaller is more stable; called the change-in-estimate")},
        ]
    else:
        status = "info"
        head = t(lang, "你還沒挑任何背景條件(共變項),所以沒辦法做「加進去前後」的兩次比較。",
                 "You have not picked any background conditions (covariates) yet, so "
                 "the before-vs-after two-pass comparison cannot be run.")
        metrics = [{"name": t(lang, "目前的估計", "Current estimate"), "value": round(iv0, 3),
                    "note": t(lang, "只用了工具變數,沒加背景條件",
                              "instrument only, no background conditions")}]
    return {
        "id": "A4b",
        "title": t(lang, "A4b・這個效果對「每個人」都差不多嗎?",
                   "A4b · Is the effect roughly the same for everyone?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            ("接種疫苗對健康的好處,是不是對每個人都差不多?還是有些人受益很多、有些人幾乎沒用?"
             "如果差很多,那我們算出來的數字其實只代表『被外力推動的那群人』,不能隨便套到所有人身上。\n"
             "怎麼檢查?做兩次估計、比一比:\n"
             "・第一次——只用工具變數,單純算出一個效果。\n"
             "・第二次——把每個人的『背景條件』(也就是年齡、BMI、收入這些『共變項』)"
             "一起放進同一個模型,再算一次。\n"
             "如果效果真的對每個人都差不多,那不管有沒有把這些背景條件放進去,"
             "兩次算出來的數字應該都差不多;反過來,如果第二次答案跳很多,"
             "就表示效果其實會隨著背景條件不同而不同,沒那麼一致。"),
            ("Is the vaccine's health benefit roughly the same for everyone? Or do "
             "some people gain a lot while others gain almost nothing? If it varies a "
             "lot, then our estimate really only describes \"the people the nudge "
             "moved\" and cannot be casually applied to everyone.\n"
             "How do we check? Estimate twice and compare:\n"
             "• Pass 1 — use the instrument only, and get one effect.\n"
             "• Pass 2 — add each person's \"background conditions\" (age, BMI, income "
             "— the \"covariates\") into the same model and estimate again.\n"
             "If the effect really is about the same for everyone, then whether or not "
             "those background conditions are included, the two numbers should come out "
             "similar. Conversely, if the second answer jumps a lot, the effect must "
             "actually differ with background, so it is not that uniform.")),
        "term": t(
            lang,
            ("📖 專有名詞:這個假設叫「效果同質性(homogeneity)」。「背景條件」在統計裡叫"
             "「共變項(covariates)」。IV 算出來的數字只代表順從者那群人的效果,"
             "術語叫「LATE(局部平均處置效果)」。這種「加進共變項看答案變多少」的檢查法,"
             "叫「change-in-estimate」。"),
            ("📖 Terminology: this assumption is effect \"homogeneity\". The "
             "\"background conditions\" are called \"covariates\". The number IV "
             "produces only describes the compliers' effect, known as the \"LATE "
             "(local average treatment effect)\". This \"add covariates and see how "
             "much the answer moves\" check is the \"change-in-estimate\".")),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# 反證法 — Instrumental inequalities (Balke–Pearl / Pearl) falsification test
#   核心精神:IV 假設沒辦法正面「證明」,但可以設計檢驗去「試著推翻」。
#   資料如果違反了這組不等式,代表 A2(只走一條路)或 A3(像抽籤)其中至少一個
#   一定被打破了 —— 等於資料直接「反證」了這個外力的資格。
#   推不翻不代表合格,只代表「通過了這一關」。
# ---------------------------------------------------------------------------
def check_falsification_inequalities(df, Y, A, Z, lang="zh"):
    if not (_binary(df[A]) and _binary(df[Z])):
        return {
            "id": "FALS",
            "title": t(lang, "抓包測試・能不能用資料當場戳破這個外力?",
                       "Gotcha test · Can the data catch this nudge red-handed?"),
            "status": "info",
            "headline": t(
                lang,
                "這個抓包測試需要外力和處置都是「有/沒有」的二元,目前資料不符合,先略過。",
                "This gotcha test needs both the nudge and the treatment to be binary "
                "yes/no; the current data is not, so it is skipped."),
            "plain": t(
                lang,
                ("這是上一題(A2)的好搭檔。A2 那種「外力只走一條路」沒辦法用資料正面證明,"
                 "所以我們換個玩法:不去證明它對,而是『反過來挑它毛病、想辦法當場戳破它』。"
                 "如果真的戳破了,就確定這個外力不合格;戳不破,至少先過一關。"
                 "不過這招需要外力(Z)和處置(A)都是二元的『有/沒有』才能算。"),
                ("This is the companion to the previous item (A2). A2's \"the nudge "
                 "travels only one path\" cannot be proven directly from data, so we "
                 "flip the game: instead of proving it right, we try to *catch it out* "
                 "— pick holes and break it on the spot. If we do break it, the nudge "
                 "is definitely disqualified; if we cannot, it at least clears this "
                 "hurdle. But this trick requires both the nudge (Z) and the treatment "
                 "(A) to be binary yes/no.")),
            "term": t(
                lang,
                ("📖 專有名詞:這套檢查叫「工具不等式(instrumental inequalities)」,"
                 "屬於「反證 / 否證檢定(falsification test)」——不證明假設成立,而是試著推翻它。"),
                ("📖 Terminology: this check is the \"instrumental inequalities\", a kind "
                 "of \"falsification test\" — it does not prove the assumption holds, it "
                 "tries to refute it.")),
            "metrics": [{"name": t(lang, "目前能不能做這個抓包?", "Can the gotcha run now?"),
                         "value": t(lang, "不能", "No"),
                         "note": t(lang, "需要 Z 與 A 都是二元", "needs both Z and A binary")}],
        }

    a = np.asarray(df[A], dtype=int)
    z = np.asarray(df[Z], dtype=int)
    yraw = np.asarray(df[Y], dtype=float)
    if _binary(df[Y]):
        y = yraw.astype(int)
        ybin_note = t(lang, "結果本來就是二元,直接使用",
                      "the outcome is already binary, used as is")
    else:
        thr = float(np.median(yraw))
        y = (yraw > thr).astype(int)
        ybin_note = t(lang,
                      f"結果是連續的,先以中位數 {thr:.2f} 切成「高/低」兩組再檢查",
                      f"the outcome is continuous, so it is split at its median "
                      f"{thr:.2f} into high/low groups first")

    def P(yy, aa, zz):
        denom = int(np.sum(z == zz))
        if denom == 0:
            return 0.0
        return float(np.sum((y == yy) & (a == aa) & (z == zz))) / denom

    # Pearl 工具不等式:對每一個處置值 a,Σ_y max_z P(y,a|z) ≤ 1
    lhs = []
    for aa in (0, 1):
        s = max(P(0, aa, 0), P(0, aa, 1)) + max(P(1, aa, 0), P(1, aa, 1))
        lhs.append(s)
    worst = max(lhs)
    margin = worst - 1.0

    metrics = [
        {"name": t(lang, "把結果怎麼分組", "How the outcome is grouped"),
         "value": "—", "note": ybin_note},
        {"name": t(lang, "資料算出來的最大值", "Largest value from the data"),
         "value": round(worst, 3),
         "note": t(lang, "合格的外力不會超過 1.000,超過就是被戳破",
                   "a valid nudge never exceeds 1.000; over that means it is broken")},
        {"name": t(lang, "離上限還有多少", "Slack to the ceiling"),
         "value": (t(lang, f"{-margin:+.3f}", f"{-margin:+.3f}") if margin <= 0
                   else t(lang, f"超出 {margin:.3f}", f"over by {margin:.3f}")),
         "note": t(lang, "有餘裕(負)=目前挑不出毛病;正=當場被抓包",
                   "slack (negative) = no fault found yet; positive = caught red-handed")},
    ]
    if worst > 1.0 + 1e-9:
        status = "red"
        head = t(lang,
                 "當場被抓包了!這個外力違反了資料該守的上限,代表它一定不合格——必須換工具或重想設計。",
                 "Caught red-handed! This nudge breaks the ceiling the data must "
                 "respect, so it is definitely invalid — pick another instrument or "
                 "rethink the design.")
    elif worst > 0.99:
        status = "amber"
        head = t(lang,
                 "數字非常貼近上限,可能只是抽樣誤差,但很接近被戳破的邊緣,要謹慎。",
                 "The value sits very close to the ceiling — possibly just sampling "
                 "noise, but near the breaking edge, so be careful.")
    else:
        status = "green"
        head = t(lang,
                 "戳不破,這個外力通過了這道抓包測試。注意:通過不等於「保證合格」,只是「目前抓不到毛病」。",
                 "Unbroken — the nudge passes this gotcha test. Note: passing does not "
                 "mean \"guaranteed valid\", only \"no fault found so far\".")
    return {
        "id": "FALS",
        "title": t(lang, "抓包測試・能不能用資料當場戳破這個外力?",
                   "Gotcha test · Can the data catch this nudge red-handed?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            ("上一題(A2『外力只走一條路』)沒辦法用資料正面證明。這裡換個思路:"
             "與其證明它對,不如『反過來挑它毛病、試著戳破它』。如果一個外力真的合格(只透過處置影響結果、"
             "又像抽籤一樣分配),那資料裡某些比例的組合就『不可能』超過一個上限。"
             "我們把這些上限算出來:只要有任何一個被超過,就等於資料當場說『這個外力不合格』,"
             "直接出局;如果通通沒超過,代表它被你挑不出毛病——但這只是『目前沒被抓到』,"
             "不能當成『已經證明清白』。"),
            ("The previous item (A2, \"the nudge travels only one path\") cannot be "
             "proven directly from data. Here we take a different tack: rather than "
             "prove it right, we *pick holes and try to break it*. If a nudge really "
             "is valid (affecting the outcome only through the treatment, and assigned "
             "as if by lottery), then certain combinations of proportions in the data "
             "*cannot* exceed a ceiling. We compute those ceilings: if any one is "
             "exceeded, the data is saying outright \"this nudge is invalid\" and it is "
             "out; if none is exceeded, you could not find a fault — but that only "
             "means \"not caught so far\", not \"proven innocent\".")),
        "term": t(
            lang,
            ("📖 專有名詞:這套上限叫「工具不等式(instrumental inequalities,Balke–Pearl / Pearl)」,"
             "是一種「反證 / 否證檢定(falsification test)」。它能否證的是「排除限制(A2)」與"
             "「獨立性(A3)」這組假設:違反 → IV 一定不成立;沒違反 → 通過這關,但無法反推它一定成立。"),
            ("📖 Terminology: these ceilings are the \"instrumental inequalities "
             "(Balke–Pearl / Pearl)\", a form of \"falsification test\". What they can "
             "refute is the pair of assumptions exclusion restriction (A2) and "
             "independence (A3): violated → IV definitely fails; not violated → it "
             "clears this hurdle, but that does not prove it must hold.")),
        "metrics": metrics,
    }


def check_all(df, Y, A, Z, covariates=(), lang="zh"):
    covariates = list(covariates)
    checks = [
        check_a1_strength(df, A, Z, lang=lang),
        check_a2_exclusion(df, Y, A, Z, lang=lang),
        check_falsification_inequalities(df, Y, A, Z, lang=lang),
        check_a3_independence(df, Z, covariates, A, lang=lang),
        check_a4a_monotonicity(df, A, Z, lang=lang),
        check_a4b_homogeneity(df, Y, A, Z, covariates, lang=lang),
    ]
    order = {"red": 0, "amber": 1, "info": 2, "green": 3}
    worst = min((c["status"] for c in checks), key=lambda s: order[s])
    summary = {
        "green": t(lang,
                   "整體來說:能用資料檢查、以及試著「抓包戳破」的項目都過關了。剩下「外力有沒有走別條路」這題,要再靠你的專業判斷。",
                   "Overall: the items that can be checked from data, including the "
                   "\"gotcha\" attempt to break it, all pass. The remaining question — "
                   "whether the nudge takes another path — still needs your domain "
                   "judgement."),
        "amber": t(lang,
                   "整體來說:有項目亮了黃燈,IV 的結論請保守一點看。",
                   "Overall: some items are amber, so read the IV conclusion a bit "
                   "more cautiously."),
        "red": t(lang,
                 "整體來說:有項目亮了紅燈(甚至被資料當場抓包戳破),IV 的結論可能不可靠,要先處理問題。",
                 "Overall: some items are red (one was even caught and broken by the "
                 "data), so the IV conclusion may be unreliable — fix the problems "
                 "first."),
        "info": t(lang,
                  "整體來說:有些項目資料算不出來,需要你的專業判斷才能下結論。",
                  "Overall: some items cannot be computed from data and need your "
                  "domain judgement to conclude."),
    }[worst]
    return {"checks": checks, "overall_status": worst, "overall_headline": summary}
