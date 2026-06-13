"""External-control assumption dashboard (C1–C5)."""
import numpy as np
from extctrl_gen import generate

try:
    from i18n import t
except Exception:  # pragma: no cover
    def t(lang, zh, en):
        return zh if lang == "zh" else en


def _chk(status, title, headline, plain, term, metrics):
    return {"status": status, "title": title, "headline": headline, "plain": plain, "term": term, "metrics": metrics}


def run_dashboard(mu_ext=-0.5, lang="zh"):
    d = generate(mu_ext=float(mu_ext))
    xt, xe = d["trial_X"], d["ext_X"]
    # C4 positivity/overlap: does the trial's X fall inside the external controls' support?
    lo, hi = np.quantile(xe, 0.01), np.quantile(xe, 0.99)
    outside = float(np.mean((xt < lo) | (xt > hi)))
    c4 = "green" if outside < 0.02 else ("amber" if outside < 0.10 else "red")
    # C5 covariate distance (standardised mean difference between the two arms)
    smd = abs(xt.mean() - xe.mean()) / np.sqrt((xt.var() + xe.var()) / 2 + 1e-9)
    c5 = "green" if smd < 0.3 else ("amber" if smd < 0.8 else "red")

    checks = [
        _chk("info",
             t(lang, "C1 可比性／可交換性（扣掉已測共變項後外部對照可當對照）", "C1 Comparability / exchangeability (external controls are valid controls given measured covariates)"),
             t(lang, "最關鍵、不可檢驗。", "The key, untestable one."),
             t(lang, "在已測共變項之下，外部對照的結果＝試驗病人『若沒治療』的結果。若還有沒量到、又在兩組分布不同的預後因子，效果就會偏。",
                     "Given the measured covariates, the external controls' outcomes equal what the trial patients' outcomes would be if untreated. An unmeasured prognostic factor that differs between the groups biases the effect."),
             t(lang, "可交換性＝兩組除了治療之外，（在校正後）沒有系統性差異。", "Exchangeability = after adjustment the two groups differ only by the treatment."),
             []),
        _chk("info",
             t(lang, "C2 無時代／來源效應（outcome 不因年代或資料源而漂移）", "C2 No era / source effect (the outcome doesn't drift by calendar time or data source)"),
             t(lang, "歷史對照的頭號風險。", "The top risk with historical controls."),
             t(lang, "外部對照常來自更早的年代或不同醫療系統：照護標準、診斷與基線風險可能隨時間／來源改變，造成『時代偏誤』。越同期、越同源越好。",
                     "External controls often come from an earlier era or a different health system: standards of care, diagnosis and baseline risk can drift over time/source, causing time-trend bias. The more contemporaneous and same-source, the better."),
             t(lang, "時代偏誤＝不是治療造成、而是『時間／來源不同』造成的結果差異。", "Era bias = an outcome difference driven by time/source, not by the treatment."),
             []),
        _chk("info",
             t(lang, "C3 一致的結果定義與納入條件", "C3 Consistent outcome definition & eligibility"),
             t(lang, "蘋果比蘋果。", "Compare like with like."),
             t(lang, "結果要用同樣的方式量、在同樣的時間點評估；外部對照要符合試驗的納入／排除條件。定義不一致會假裝成治療效果。",
                     "The outcome must be measured the same way at the same time points, and the external controls must meet the trial's eligibility criteria. Inconsistent definitions masquerade as a treatment effect."),
             t(lang, "可用『校準』法：先在共同的陰性結果上檢查兩源是否一致。", "A calibration step can check that the two sources agree on a shared negative-control outcome."),
             []),
        _chk(c4,
             t(lang, "C4 正性／重疊（試驗病人落在外部對照的支持範圍內）", "C4 Positivity / overlap (trial patients lie within the external controls' support)"),
             t(lang, f"試驗約 {outside*100:.0f}% 落在外部對照 X 範圍外。", f"~{outside*100:.0f}% of the trial is outside the external controls' X range."),
             t(lang, "試驗病人的每種共變項組合，外部對照裡都要有人——否則加權／標準化是在外推到沒資料的地方。落在範圍外越多越危險。",
                     "Every covariate value among the trial patients must occur among the external controls; otherwise weighting/standardisation extrapolates into regions with no data. The more outside the support, the riskier."),
             t(lang, "正性／重疊＝試驗的共變項支持 ⊆ 外部對照的支持。", "Positivity / overlap = the trial's covariate support ⊆ the external controls'."),
             [{"name": t(lang, "試驗落在外部範圍外", "trial outside external support"), "value": f"{outside*100:.0f}%", "note": ""}]),
        _chk(c5,
             t(lang, "C5 兩組差多少（建模需求）", "C5 How far apart the arms are (modelling demand)"),
             t(lang, f"共變項標準化差異 SMD ≈ {smd:.2f}。", f"Covariate SMD ≈ {smd:.2f}."),
             t(lang, "兩組共變項差越大，校正越仰賴模型設定正確、外推越多。差距小→穩；差距大→對模型敏感、要小心（並做敏感度分析）。",
                     "The further apart the arms' covariates, the more the adjustment leans on a correctly specified model and extrapolation. Small gap → stable; large gap → model-sensitive, handle with care (and run sensitivity analyses)."),
             t(lang, "SMD＝標準化平均差，衡量兩分布相距多遠。", "SMD = standardised mean difference, how far apart two distributions are."),
             [{"name": "SMD", "value": f"{smd:.2f}", "note": ""}]),
    ]
    order = {"red": 0, "amber": 1, "info": 2, "green": 3}
    worst = min((c["status"] for c in checks), key=lambda s: order[s])
    head = {"red": t(lang, "借外部對照有風險：正性或差距亮紅燈。", "Borrowing external controls is risky: positivity or the gap is red."),
            "amber": t(lang, "可借，但要留意正性、兩組差距與時代效應。", "Borrowable, but mind positivity, the arm gap and era effects."),
            "info": t(lang, "可檢項看起來合理；關鍵假設仍需領域判斷。", "The checkable items look reasonable; the key assumptions need judgement."),
            "green": t(lang, "可檢項通過；關鍵假設仍需領域判斷。", "Checkable items pass; the key assumptions still need judgement.")}[worst]
    return {"overall_status": worst, "overall_headline": head, "checks": checks}
