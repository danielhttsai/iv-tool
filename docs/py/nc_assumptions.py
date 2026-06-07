"""NC / Proximal assumption checks C1–C5. The identifying assumptions are mostly about the
VALIDITY of the negative controls and the existence of a confounding bridge — untestable from
data alone (blue info cards). The testable ones are proxy relevance (a first-stage strength,
like IV) and whether there is enough data.

Each check returns {id, title, status, headline, plain, term, metrics:[...]};
run_dashboard returns {"checks": [...]}.
"""
from __future__ import annotations

import numpy as np

import nc_core
from i18n import t


def run_dashboard(df, treat="A", outcome="Y", cov="X", nco="W", nce="Z", lang="zh"):
    res = nc_core.full_nc(df, treat, outcome, cov, nco, nce, n_boot=0, lang=lang)
    A = np.asarray(df[treat], dtype=float); X = np.asarray(df[cov], dtype=float)
    W = np.asarray(df[nco], dtype=float); Z = np.asarray(df[nce], dtype=float)
    b1, se1 = nc_core._ols([A, Z, X], W)                     # first stage W~A+Z+X
    z_t = float(abs(b1[2] / se1[2])) if se1[2] > 0 else 0.0   # NCE→NCO relevance t-stat
    checks = [
        _c1_nco(res, lang),
        _c2_nce(lang),
        _c3_bridge(lang),
        _c4_relevance(z_t, res, lang),
        _c5_data(res, lang),
    ]
    return {"checks": checks}


def _c1_nco(res, lang="zh"):
    return {
        "id": "C1",
        "title": t(lang, "陰性對照結果（NCO）有效嗎？暴露對它應<b>無因果</b>、又與混淆相關（關鍵、不可檢驗）",
                   "Is the negative-control outcome (NCO) valid? exposure has <b>no causal effect</b> on it, yet it shares the confounder (key, untestable)"),
        "status": "info",
        "headline": t(lang, "NCO 要『疫苗不可能影響、但和未測混淆 U 同源』。選錯，偵測與校正都會失靈。",
                      "The NCO must be 'something the vaccine cannot affect, yet driven by the same unmeasured U'. Choose wrong and both detection and correction fail."),
        "plain": t(
            lang,
            "<b>陰性對照結果（NCO，本工具的 W）</b>必須滿足兩件事：(1) 暴露 A 對它<b>沒有因果效應</b>（所以天真估 A→W 的"
            "『真值』是 0）；(2) 它和產生偏誤的<b>未測混淆 U 相關</b>（所以能當 U 的代理）。常見選擇：接種<b>之前</b>就量到的"
            "結果、或一個機轉上不可能被疫苗影響、卻同樣反映健康／就醫傾向的事件。第 (1) 點是設計假設、無法用資料證明；選錯 NCO"
            "（其實會被暴露影響）會讓偵測誤判、校正引入新偏誤。",
            "The <b>negative-control outcome (NCO, our W)</b> must satisfy two things: (1) the exposure A has <b>no causal effect</b> "
            "on it (so the true value of a naive A→W is 0); and (2) it is <b>associated with the unmeasured confounder U</b> (so it can "
            "proxy U). Typical choices: an outcome measured <b>before</b> vaccination, or an event the vaccine cannot mechanistically "
            "affect but that still reflects health / care-seeking. Point (1) is a design assumption, unprovable from data; a bad NCO "
            "(secretly affected by exposure) makes detection misleading and correction introduce new bias.",
        ),
        "term": t(lang, "專有名詞：陰性對照結果（negative-control outcome）；偽陽性查核；Lipsitch 2010。",
                  "Term: negative-control outcome; falsification check; Lipsitch 2010."),
        "metrics": [
            {"name": t(lang, "天真 A → W（本應 0）", "naive A → W (should be 0)"),
             "value": f"{res['detect']:+.2f}",
             "note": t(lang, "離 0 越遠＝未測混淆越強（偵測訊號）", "the further from 0, the stronger the unmeasured confounding (the detection signal)")},
        ],
    }


def _c2_nce(lang="zh"):
    return {
        "id": "C2",
        "title": t(lang, "陰性對照暴露（NCE）有效嗎？它對結果應<b>無因果</b>、又與混淆相關（關鍵、不可檢驗）",
                   "Is the negative-control exposure (NCE) valid? it has <b>no causal effect</b> on the outcome, yet shares the confounder (key, untestable)"),
        "status": "info",
        "headline": t(lang, "NCE 要『不可能影響結果、但和未測混淆 U 同源』；它是近端校正能成立的另一半。",
                      "The NCE must be 'something that cannot affect the outcome, yet driven by the same U'; it is the other half that makes proximal correction work."),
        "plain": t(
            lang,
            "<b>陰性對照暴露（NCE，本工具的 Z）</b>必須：(1) 對結果 Y <b>沒有因果效應</b>（除了透過 U 的關聯）；(2) 與<b>未測混淆 U "
            "相關</b>。近端因果（P2SLS）就是用 NCE（Z）把 NCO（W）裡「U 的部分」抓出來，再回去把結果模型中 U 的效應扣掉。"
            "若 Z 其實會直接影響 Y（不是有效的 NCE），校正就會偏。和 NCO 一樣，這是設計假設、靠領域知識挑選。",
            "The <b>negative-control exposure (NCE, our Z)</b> must (1) have <b>no causal effect</b> on the outcome Y (other than via its "
            "association with U) and (2) be <b>associated with the unmeasured U</b>. Proximal P2SLS uses the NCE (Z) to extract the "
            "'U part' of the NCO (W), then subtracts U's effect from the outcome model. If Z actually affects Y directly (not a valid "
            "NCE), the correction is biased. Like the NCO, this is a design assumption chosen by domain knowledge.",
        ),
        "term": t(lang, "專有名詞：陰性對照暴露（negative-control exposure）；治療混淆代理；confounding bridge。",
                  "Term: negative-control exposure; treatment confounding proxy; confounding bridge."),
        "metrics": [],
    }


def _c3_bridge(lang="zh"):
    return {
        "id": "C3",
        "title": t(lang, "存在「混淆橋」嗎？這對陰性對照足以代表 U（完備性，關鍵、不可檢驗）",
                   "Does a confounding bridge exist? the proxy pair captures U well enough (completeness — key, untestable)"),
        "status": "info",
        "headline": t(lang, "近端因果要求一個解得出的「混淆橋函數」＋完備性條件；資料證明不了，是它的核心代價。",
                      "Proximal causal inference requires a solvable 'confounding bridge' + a completeness condition; this can't be proven from data — it is the core price."),
        "plain": t(
            lang,
            "近端因果（雙陰性對照）的識別，靠的是存在一個<b>混淆橋函數</b>，把結果用暴露與 NCO 表示出來，並滿足<b>完備性</b>"
            "（兩個代理對 U 的資訊『夠豐富』，能把 U 解出來）。這比 IV 的排除限制更抽象，且<b>無法用資料檢驗</b>。實務上：盡量挑"
            "與 U 強相關、彼此資訊互補的一對陰性對照；做敏感度分析；若可能，和已知答案／隨機試驗對照。線性 P2SLS 是它在"
            "<b>可加可乘線性</b>下的特例；非線性時需更彈性的橋函數（ML，見 ⑤ 之延伸）。",
            "Identification in proximal (double negative control) inference rests on the existence of a <b>confounding bridge function</b> "
            "expressing the outcome via the exposure and the NCO, plus a <b>completeness</b> condition (the two proxies carry 'rich enough' "
            "information about U to recover it). This is more abstract than IV's exclusion restriction and <b>cannot be tested from data</b>. "
            "In practice: pick a pair of negative controls each strongly related to U and informationally complementary; do sensitivity "
            "analyses; benchmark against a known answer / trial when possible. Linear P2SLS is the special case under <b>additive-linear</b> "
            "structure; non-linear settings need flexible bridge functions (ML — see the ⑤ extension).",
        ),
        "term": t(lang, "專有名詞：混淆橋函數（confounding bridge）；完備性（completeness）；近端因果學習（proximal causal learning）。",
                  "Term: confounding bridge function; completeness; proximal causal learning."),
        "metrics": [],
    }


def _c4_relevance(z_t, res, lang="zh"):
    if z_t >= 8:
        status, head = "green", t(lang, "陰性對照與混淆代理的相關性很強——近端校正穩定。",
                                  "The negative controls are strongly related (relevant proxies) — proximal correction is stable.")
    elif z_t >= 3:
        status, head = "amber", t(lang, "代理相關性中等——校正可行但信賴區間偏寬，留意精確度。",
                                  "Moderate proxy relevance — correction works but the CI is wider; mind precision.")
    else:
        status, head = "red", t(lang, "代理太弱（類似弱工具）——近端校正會很不穩，別過度解讀。",
                                "Proxies too weak (like a weak instrument) — proximal correction is unstable; don't over-interpret.")
    return {
        "id": "C4",
        "title": t(lang, "代理夠相關嗎？（可檢驗：第一階段 NCE→NCO 強度）",
                   "Are the proxies relevant enough? (testable: first-stage NCE→NCO strength)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "和 IV 的「工具要夠強」一樣，近端因果也要求兩個陰性對照<b>夠相關</b>（都和 U 牢牢綁住、彼此資訊互補）。可檢的指標是"
            "<b>第一階段</b>（W ~ A + Z + X）裡 <b>NCE（Z）對 NCO（W）的強度</b>（t 值）——太弱就像弱工具，P2SLS 會放大誤差、"
            "不穩。相關性是<b>可檢</b>的；但『無因果』（C1、C2）與完備性（C3）仍不可檢。",
            "Just as IV needs a strong instrument, proximal inference needs the two negative controls to be <b>relevant enough</b> (both "
            "tightly tied to U and informationally complementary). The testable metric is the <b>first-stage</b> (W ~ A + Z + X) "
            "<b>strength of the NCE (Z) on the NCO (W)</b> (t-statistic) — too weak is like a weak instrument and P2SLS becomes noisy and "
            "unstable. Relevance is <b>testable</b>; the 'no causal effect' (C1, C2) and completeness (C3) conditions remain untestable.",
        ),
        "term": t(lang, "專有名詞：代理相關性（proxy relevance）；弱工具類比；第一階段強度。",
                  "Term: proxy relevance; weak-instrument analogy; first-stage strength."),
        "metrics": [
            {"name": t(lang, "第一階段 NCE→NCO（t 值）", "first-stage NCE→NCO (t-stat)"), "value": f"{z_t:.1f}",
             "note": t(lang, "越大越好；類比 IV 的工具強度", "larger is better; analogous to IV instrument strength")},
        ],
    }


def _c5_data(res, lang="zh"):
    n = int(res.get("n", 0))
    if n >= 3000:
        status, head = "green", t(lang, "樣本充足——近端估計與信賴區間穩定。",
                                  "Ample sample — the proximal estimate and CI are stable.")
    elif n >= 800:
        status, head = "amber", t(lang, "樣本中等——P2SLS 的兩階段會放大誤差，信賴區間偏寬。",
                                  "Moderate sample — the two stages of P2SLS amplify noise; the CI is wide.")
    else:
        status, head = "red", t(lang, "樣本太少——兩階段估計很不穩，別過度解讀。",
                                "Too few observations — the two-stage estimate is unstable; don't over-interpret.")
    return {
        "id": "C5",
        "title": t(lang, "資料量夠嗎？（可檢驗）", "Is there enough data? (testable)"),
        "status": status, "headline": head,
        "plain": t(
            lang,
            "近端因果是<b>兩階段</b>估計（第一階段配 Ŵ、第二階段估效應），每一階段的雜訊都會往下傳，所以對<b>樣本量</b>比天真單階段"
            "更敏感。樣本太少時信賴區間會很寬、甚至不穩；務必看自助信賴區間，別只看點估計。",
            "Proximal inference is a <b>two-stage</b> estimator (fit Ŵ first, then estimate the effect), and noise from each stage "
            "propagates, so it is more sensitive to <b>sample size</b> than a naive one-stage analysis. With too few observations the CI is "
            "wide or unstable — always read the bootstrap CI, not just the point estimate.",
        ),
        "term": t(lang, "專有名詞：兩階段估計；誤差傳遞；自助信賴區間。",
                  "Term: two-stage estimation; error propagation; bootstrap CI."),
        "metrics": [
            {"name": t(lang, "樣本數", "sample size"), "value": f"{n}",
             "note": t(lang, "兩階段估計需要較大樣本", "two-stage estimation needs a larger sample")},
        ],
    }
