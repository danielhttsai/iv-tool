"""Built-in DEMO dataset for the Test-Negative Design (TND) tabs.

Fully fictional teaching scenario — NOT real data:

    情境：想估「疫苗對<b>目標病原</b>（如流感）的效力 VE」。麻煩在於<b>就醫傾向</b>（care-seeking）是個
          未測混淆：愛就醫／健康意識高的人<b>較常接種</b>，<b>也較常因症狀就醫而被檢驗</b>。用「一般族群」
          當對照的天真病例對照，會被就醫傾向偏掉。<b>陰性檢驗設計（TND）</b>只在「<b>有來檢驗的人</b>」裡比：
          case＝目標病原檢驗陽性、control＝檢驗陰性（其他呼吸道病原）。因為兩組都<b>來檢驗過</b>，就醫傾向
          被「條件在有檢驗」抵銷掉 → 用接種勝算比估 <b>VE ＝ 1 − OR</b>，不再被就醫傾向偏。

每筆資料：pid、vaccinated（接種 1／否 0）、tested（是否因症狀就醫檢驗）、case（檢驗陽性＝目標病原 1，
陰性＝其他病原 0；僅 tested 者有意義）、infected（真的得到目標病原，供天真世代對照用）。真值＝VE_TRUE。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 5
N = 24000
VE_TRUE = 0.60          # true vaccine effectiveness against the TARGET pathogen
H_V = 1.1               # care-seeking → vaccination (confounding)
CSEEK = 1.0             # care-seeking → being tested (the selection TND conditions on)
BASE_T = -2.2           # baseline log-odds of target infection
BASE_N = -2.2           # baseline log-odds of a non-target ILI (test-negative)
COLUMNS = ["pid", "vaccinated", "tested", "case", "infected"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "demo_tnd.csv")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(seed=SEED, n=N, ve=VE_TRUE, cseek=CSEEK):
    """Simulate vaccine effectiveness under care-seeking confounding. A naive case-control with
    population controls is biased by care-seeking; the test-negative design recovers VE."""
    rng = np.random.default_rng(seed)
    H = rng.normal(0.0, 1.0, n)                                   # care-seeking propensity (unmeasured)
    V = (rng.random(n) < _sig(0.2 + H_V * H)).astype(float)       # care-seekers vaccinate more
    pT = _sig(BASE_T + np.log(max(1e-6, 1.0 - ve)) * V)           # vaccine protects against target
    T = (rng.random(n) < pT).astype(float)
    Npath = (rng.random(n) < _sig(BASE_N)).astype(float)          # non-target ILI, independent of V
    sympt = ((T + Npath) > 0).astype(float)
    tested = ((rng.random(n) < _sig(-0.3 + cseek * 1.3 * H)) * sympt).astype(float)
    # case among the tested: target-positive = 1, test-negative (non-target only) = 0
    case = np.where((tested == 1) & (T == 1), 1.0,
                    np.where((tested == 1) & (Npath == 1) & (T == 0), 0.0, -1.0))
    return pd.DataFrame({
        "pid": np.arange(n), "vaccinated": V, "tested": tested,
        "case": case, "infected": T,
    }, columns=COLUMNS)


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    m = df.tested == 1
    print(f"wrote {OUT}  ({len(df)} rows; {int(m.sum())} tested, {int((df.case == 1).sum())} cases)")
    print(f"true VE = {VE_TRUE}")
