"""Built-in DEMO dataset for the Propensity Score (PS) tabs.

Fully fictional teaching scenario — NOT real data:

    情境：想估「疫苗 A 對結果 Y 的平均因果效應（ATE）」。麻煩在於<b>適應症混淆</b>：
          病重／體弱的人（共變項 X 大）<b>較容易接種</b>，<b>也較容易發生結果</b>——所以直接比較
          「接種 vs 沒接種」會被 X 偏掉（粗估高估）。
          <b>傾向分數（propensity score, PS）</b>＝在共變項 X 下「會接種的機率」P(A=1|X)。
          只要把 X 平衡掉——用 PS <b>配對 / IPTW 加權 / 重疊權重</b>——就能還原真正的 ATE。

每筆資料：pid、A（接種 1／否 0）、Y（結果，連續）、X（嚴重度／體弱，已測共變項）。
真實平均因果效應 TRUE_ATE（這裡 2.0）。混淆來自 X 同時驅動 A 與 Y。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 67
N = 4000
TRUE_ATE = 2.0
# treatment model: severity X drives vaccination (confounding by indication)
A0, A_X = 0.0, 1.0
# outcome model: A's true effect = TRUE_ATE; X also raises the outcome (confounding)
B_X = 1.3
COLUMNS = ["pid", "A", "Y", "X"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "demo_ps.csv")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(seed=SEED, n=N, true_ate=TRUE_ATE, conf=1.0):
    """Simulate data where severity X confounds vaccination and the outcome. `conf` scales
    how strongly X drives both (the confounding strength). Truth = `true_ate`.
    The crude difference is biased; PS matching / IPTW / overlap weighting recover the truth."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, n)                              # severity / frailty (measured)
    A = (rng.random(n) < _sig(A0 + A_X * conf * X)).astype(float)
    Y = true_ate * A + B_X * conf * X + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "pid": np.arange(n), "A": A, "Y": np.round(Y, 4), "X": np.round(X, 4),
    }, columns=COLUMNS)


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    A = df.A.values; Y = df.Y.values; X = df.X.values
    crude = Y[A == 1].mean() - Y[A == 0].mean()
    ps = _sig(A0 + A_X * X)
    w = np.where(A == 1, 1.0 / ps, 1.0 / (1.0 - ps))        # IPTW (true PS)
    iptw = (np.sum(w * A * Y) / np.sum(w * A)) - (np.sum(w * (1 - A) * Y) / np.sum(w * (1 - A)))
    print(f"wrote {OUT}  ({len(df)} rows)")
    print(f"crude diff  = {crude:.3f}   (biased; true {TRUE_ATE})")
    print(f"IPTW (true PS) = {iptw:.3f}   (true {TRUE_ATE})")
