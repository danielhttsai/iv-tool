"""RDD assumption checks R1-R5 — the discontinuity-design analogue of the IV A1-A4b.

白話文優先：每一項先用日常語言說明在檢查什麼、結果代表什麼，專有名詞放到 `term`。

RDD relies on DIFFERENT assumptions from IV. The key one (R1, continuity) is not
directly testable, so we are honest about that and provide the supporting checks
that ARE testable (no manipulation, covariate balance, first-stage jump,
bandwidth sensitivity).

Each check returns the same shape as the IV dashboard:
    {
        "id": "R1",
        "title":   "<白話標題>",
        "status":  "green" | "amber" | "red" | "info",
        "headline":"<一句白話結論>",
        "plain":   "<我們在檢查什麼、為什麼重要，全用白話>",
        "term":    "<最後才出現的專有名詞 + 白話解釋>",
        "metrics": [{"name": ..., "value": ..., "note": ...}],
    }
"""
from __future__ import annotations

import numpy as np

import rdd_core
from i18n import t


# ---------------------------------------------------------------------------
# R1 — Continuity of potential outcomes at the cutoff (the key, NOT testable)
# ---------------------------------------------------------------------------
def check_r1_continuity(lang="zh"):
    return {
        "id": "R1",
        "title": t(lang, "斷點兩側的人本來就「差不多」嗎？", "Are people just below vs. just above the cutoff basically alike?"),
        "status": "info",
        "headline": t(
            lang,
            "這是 RDD 最關鍵的前提，無法用資料直接證明，要靠領域知識判斷。",
            "This is RDD's key premise; it cannot be proven from data alone and rests on domain knowledge.",
        ),
        "plain": t(
            lang,
            "RDD 的核心想法是：剛好不到 65 歲與剛滿 65 歲的人，除了「有沒有資格」之外，"
            "其他條件（包含我們看不見的健康意識）幾乎一樣。所以資格門檻附近的結果差異，"
            "可以歸因於資格本身。這個「兩側連續」的假設沒辦法直接檢驗，但下面 R2、R3 提供間接佐證。",
            "RDD's core idea is that people just under 65 and just over 65 are almost identical "
            "apart from eligibility — including the unobserved health-consciousness confounder. "
            "So a jump in outcomes at the threshold can be attributed to eligibility itself. "
            "This continuity assumption is not directly testable, but R2 and R3 below give indirect support.",
        ),
        "term": t(
            lang,
            "專有名詞：潛在結果的連續性（continuity of potential outcomes）。意思是，如果沒有資格門檻，"
            "結果會是年齡的平滑函數、不會在 65 歲忽然跳一下。另一種等價講法是「局部隨機化"
            "（local randomization）」：把斷點附近一個很窄的窗，當成像隨機分組的小實驗——⑤ 用 AI 強化"
            "的「藥方一」正是用這個想法，讓估計對視窗（頻寬）更穩。",
            "Term: continuity of potential outcomes — absent the threshold, the outcome would be a "
            "smooth function of age with no sudden jump exactly at 65. An equivalent alternative is "
            "'local randomization': treat a very narrow window around the cutoff as a tiny as-if-randomized "
            "experiment — Recipe ① in '⑤ Boost with AI' uses exactly this idea to make the estimate more "
            "robust to the window (bandwidth).",
        ),
        "metrics": [],
    }


# ---------------------------------------------------------------------------
# R2 — No manipulation / sorting around the cutoff (McCrary density test)
# ---------------------------------------------------------------------------
def check_r2_manipulation(x, c, lang="zh"):
    dt = rdd_core.density_test(np.asarray(x, dtype=float), c)
    p = dt["p"]
    ratio = dt["ratio"]
    metrics = [
        {"name": t(lang, "斷點兩側人數密度的比值", "Density ratio just above vs. below the cutoff"),
         "value": round(ratio, 3) if np.isfinite(ratio) else None,
         "note": t(lang, "接近 1 代表沒有人為堆積", "close to 1 means no pile-up on one side")},
        {"name": t(lang, "密度有沒有跳一下的檢定 p 值", "p-value for a jump in density"),
         "value": round(p, 3) if np.isfinite(p) else None,
         "note": t(lang, "p 大（>0.05）代表沒有明顯操弄跡象",
                   "a large p (>0.05) means no clear sign of manipulation")},
    ]
    if not np.isfinite(p):
        status, head = "info", t(lang, "資料量不足以判斷密度是否連續。",
                                 "Not enough data to judge whether the density is continuous.")
    elif p >= 0.05:
        status = "green"
        head = t(lang, "斷點兩側人數沒有不正常堆積，看不出有人刻意挑邊。",
                 "No abnormal pile-up on either side — no sign that people sorted themselves across the cutoff.")
    elif p >= 0.01:
        status = "amber"
        head = t(lang, "斷點附近人數分布有點不平均，要留意是否有挑邊行為。",
                 "The density looks somewhat uneven near the cutoff; watch for possible sorting.")
    else:
        status = "red"
        head = t(lang, "斷點兩側人數明顯不對稱，可能有人為操弄，RDD 結果要保守看。",
                 "A clear asymmetry in density at the cutoff suggests manipulation; interpret the RDD with caution.")
    return {
        "id": "R2",
        "title": t(lang, "有人「挑邊」讓自己剛好符合資格嗎？", "Did anyone game which side of the cutoff they land on?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "如果大家能控制自己的跑分變數（這裡是年齡，沒辦法操弄），就可能把自己擠到有資格那一側，"
            "破壞「兩側相似」。我們數一數斷點兩側的人數密度，看有沒有不正常的堆積。年齡無法造假，"
            "所以這裡預期會通過——這正是用年齡當斷點的好處。"
            "那如果這項沒過（密度在斷點不對稱、像有人挑邊），整個 RDD 就完全不能用嗎？不一定——"
            "可以改用「甜甜圈 RDD」把緊貼斷點的人先拿掉、或改報一個區間範圍（界限）、"
            "或縮到一個更窄、更像隨機的窗重估；但結論要更保守，把它當「參考」而非「定論」。",
            "breaking the 'both sides alike' premise. We count the density of subjects on each side and look for "
            "an abnormal pile-up. Age cannot be faked, so we expect this to pass — that is a strength of using age as the cutoff. "
            "And if this check fails (an asymmetric density at the cutoff, as if people sorted themselves), is the whole RDD "
            "useless? Not necessarily — you can switch to a 'donut-hole RDD' that drops points hugging the cutoff, report "
            "bounds instead of a single point, or re-estimate inside a narrower, more random-like window; but treat the result "
            "as suggestive rather than definitive.",
        ),
        "term": t(lang, "專有名詞：McCrary 密度檢定（操弄/分類排序檢定）。",
                  "Term: McCrary density test (manipulation / sorting test)."),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# R3 — Covariate continuity / balance at the cutoff
# ---------------------------------------------------------------------------
def check_r3_balance(df, x_name, cov_names, c, h, lang="zh"):
    cc = rdd_core.covariate_continuity(df, x_name, cov_names, c, h, lang=lang)
    n_bad = cc["n_imbalanced"]
    metrics = [
        {"name": t(lang, f"在斷點跳一下的背景變項：{r['name']}",
                   f"Background variable jumping at the cutoff: {r['name']}"),
         "value": round(r["jump"], 3),
         "note": t(lang,
                   ("平衡（信賴區間含 0）" if r["balanced"] else "不平衡（信賴區間不含 0）"),
                   ("balanced (CI covers 0)" if r["balanced"] else "imbalanced (CI excludes 0)"))}
        for r in cc["covariates"]
    ]
    if n_bad == 0:
        status = "green"
        head = t(lang, "所有背景條件在斷點都沒有跳動，兩側的人確實很像，支持 RDD。",
                 "No background variable jumps at the cutoff — the two sides really do look alike, supporting the RDD.")
    elif n_bad == 1:
        status = "amber"
        head = t(lang, "有一項背景條件在斷點附近跳了一下，要檢查是否影響結論。",
                 "One background variable jumps at the cutoff; check whether it affects the conclusion.")
    else:
        status = "red"
        head = t(lang, f"有 {n_bad} 項背景條件在斷點跳動，兩側可能不夠相似，結果要保守看。",
                 f"{n_bad} background variables jump at the cutoff — the two sides may not be comparable; interpret with caution.")
    return {
        "id": "R3",
        "title": t(lang, "斷點兩側的背景條件平衡嗎？", "Are background characteristics balanced across the cutoff?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "如果「兩側相似」成立，那些不該被資格門檻影響的背景條件（性別、BMI、慢性病、收入），"
            "在斷點兩側應該也是平滑、不跳動的。我們對每個背景條件做一次 RD，看它在 65 歲有沒有跳。"
            "全部沒跳，就是兩側相似的有力佐證。",
            "If 'both sides alike' holds, background variables that the threshold should NOT affect "
            "(sex, BMI, chronic conditions, income) should also be smooth across the cutoff. We run an RD on each "
            "and check for a jump at 65. No jumps is strong support for comparability.",
        ),
        "term": t(lang, "專有名詞：共變項連續性／平衡檢定（covariate continuity / balance）。",
                  "Term: covariate continuity / balance test."),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# R4 — First-stage jump in treatment probability (fuzzy RDD only)
# ---------------------------------------------------------------------------
def check_r4_firststage(x, d, c, h, lang="zh"):
    res = rdd_core.sharp_rd(np.asarray(x, dtype=float), np.asarray(d, dtype=float), c, h)
    jump = res["estimate"]
    p = res["p"]
    metrics = [
        {"name": t(lang, "接種率在斷點跳升的幅度", "Jump in vaccination rate at the cutoff"),
         "value": t(lang, f"約 {jump*100:.0f} 個百分點", f"~{jump*100:.0f} percentage points"),
         "note": t(lang, "跳幅越大，工具越強", "the larger the jump, the stronger the instrument")},
        {"name": t(lang, "跳升顯著性 p 值", "Significance p-value of the jump"),
         "value": round(p, 4) if np.isfinite(p) else None,
         "note": t(lang, "p 小代表跳升真實存在", "a small p means the jump is real")},
    ]
    if jump >= 0.20 and np.isfinite(p) and p < 0.05:
        status = "green"
        head = t(lang, "資格門檻讓接種率明顯跳升，模糊 RDD 的「第一階段」夠力。",
                 "Eligibility produces a clear jump in vaccination — the fuzzy-RDD first stage is strong.")
    elif jump >= 0.10:
        status = "amber"
        head = t(lang, "接種率在斷點有跳，但幅度偏小，模糊 RDD 估計會比較不穩。",
                 "There is a jump in uptake but it is modest; the fuzzy-RDD estimate will be less stable.")
    else:
        status = "red"
        head = t(lang, "接種率在斷點幾乎沒跳，模糊 RDD 缺乏第一階段，不建議使用。",
                 "Almost no jump in uptake at the cutoff — the fuzzy RDD lacks a first stage and is not advisable.")
    return {
        "id": "R4",
        "title": t(lang, "資格門檻真的改變了接種行為嗎？（模糊 RDD 專用）",
                   "Does crossing the threshold actually change vaccination? (fuzzy RDD only)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "模糊 RDD 的前提是：跨過 65 歲資格門檻會讓「實際接種率」跳升（不必到 100%）。"
            "如果接種率在斷點沒有明顯跳動，就沒有可放大的訊號，模糊 RDD 會失效。"
            "這對應 IV 的工具強度（A1）。",
            "Fuzzy RDD requires that crossing the age-65 threshold makes the actual vaccination rate jump "
            "(it need not reach 100%). If uptake does not jump at the cutoff there is no signal to scale up and "
            "the fuzzy RDD breaks down. This mirrors IV instrument strength (A1).",
        ),
        "term": t(lang, "專有名詞：第一階段跳躍（first-stage discontinuity），即模糊 RDD 的工具強度。",
                  "Term: first-stage discontinuity — the fuzzy-RDD analogue of instrument strength."),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# R5 — Bandwidth sensitivity (is the estimate stable as the window changes?)
# ---------------------------------------------------------------------------
def check_r5_bandwidth(x, y, c, d=None, lang="zh"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    curve = rdd_core.bandwidth_curve(x, y, c, d=d, hmin=3.0, hmax=12.0, steps=10)
    ests = np.array(curve["estimate"], dtype=float)
    finite = ests[np.isfinite(ests)]
    spread = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else np.nan
    center = float(np.nanmedian(finite)) if finite.size else np.nan
    rel = abs(spread / center) if center else np.nan
    metrics = [
        {"name": t(lang, "不同視窗下估計的最小～最大", "Estimate range across windows (min–max)"),
         "value": (f"{np.nanmin(finite):.2f} ~ {np.nanmax(finite):.2f}" if finite.size else None),
         "note": t(lang, "視窗(bandwidth)從 3 變到 12 年", "bandwidth varied from 3 to 12 years")},
        {"name": t(lang, "相對波動幅度", "Relative variation"),
         "value": (f"{rel*100:.0f}%" if np.isfinite(rel) else None),
         "note": t(lang, "越小代表結論越不挑視窗", "smaller means the conclusion is less sensitive to the window")},
    ]
    if np.isfinite(rel) and rel <= 0.25:
        status = "green"
        head = t(lang, "不論視窗取大取小，估計都差不多，結論很穩。",
                 "The estimate barely changes whether the window is wide or narrow — a robust conclusion.")
    elif np.isfinite(rel) and rel <= 0.5:
        status = "amber"
        head = t(lang, "估計會隨視窗變動一些，請參考互動分頁的敏感度曲線。",
                 "The estimate shifts somewhat with the window; see the sensitivity curve in the interactive tab.")
    else:
        status = "red"
        head = t(lang, "估計對視窗很敏感，不同視窗結論差很多，要特別小心。",
                 "The estimate is highly sensitive to the window — conclusions differ a lot; treat with care.")
    return {
        "id": "R5",
        "title": t(lang, "換個觀察視窗，結論還站得住嗎？", "Does the conclusion hold up when the window changes?"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "RDD 只用斷點附近的人估計。視窗（bandwidth）取太窄，資料太少、雜訊大；取太寬，"
            "會把離斷點很遠、不夠相似的人也算進來而產生偏誤。好的 RDD 結論不該過度依賴某一個視窗。"
            "我們把視窗從 3 年掃到 12 年，看估計穩不穩。",
            "RDD uses only people near the cutoff. Too narrow a window leaves too little data and high noise; "
            "too wide a window pulls in dissimilar people far from the cutoff and adds bias. A good RDD conclusion "
            "should not hinge on one window. We sweep the bandwidth from 3 to 12 years and check stability.",
        ),
        "term": t(lang, "專有名詞：頻寬敏感度（bandwidth sensitivity）。",
                  "Term: bandwidth sensitivity."),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Run the full RDD dashboard
# ---------------------------------------------------------------------------
def run_dashboard(df, x_name, y_name, d_name, c, h, cov_names, fuzzy=True, lang="zh"):
    x = df[x_name]
    checks = [
        check_r1_continuity(lang),
        check_r2_manipulation(x, c, lang),
        check_r3_balance(df, x_name, cov_names, c, h, lang),
    ]
    if fuzzy and d_name is not None:
        checks.append(check_r4_firststage(x, df[d_name], c, h, lang))
    d_arr = np.asarray(df[d_name], dtype=float) if (fuzzy and d_name is not None) else None
    checks.append(check_r5_bandwidth(x, df[y_name], c, d=d_arr, lang=lang))
    return {"checks": checks, "cutoff": float(c), "h": float(h)}
