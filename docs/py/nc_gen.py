"""Built-in DEMO dataset for the Negative Control & Proximal Causal Inference (NC) tabs.

Fully fictional teaching scenario — NOT real data:

    情境：想估「疫苗 A 對某健康結果 Y 的因果效應」。麻煩在於有一個<b>沒測到的混淆 U</b>
          （健康／就醫傾向，healthy-vaccinee）——它同時影響「會不會接種」與「結果」，天真比較就被它偏掉。
          我們手上多了兩個<b>陰性對照（negative controls）</b>：
            ・<b>陰性對照結果 NCO（W）</b>：疫苗<b>不可能</b>影響、但與 U 相關的結果（如接種前就量過的事件）。
            ・<b>陰性對照暴露 NCE（Z）</b>：<b>不可能</b>影響 Y、但與 U 相關的暴露代理。
          (1) <b>偵測</b>：天真估「A → W」本應為 0；若 ≠0，就是未測混淆的警訊（Lipsitch 2010）。
          (2) <b>校正</b>：W、Z 都是 U 的代理，用<b>近端因果／雙陰性對照</b>（proximal two-stage least squares,
              P2SLS；Miao 2018、regression-based PCI 2025）把 U 的效應扣掉，在有 U 下仍還原真值。

每筆資料：pid、A（接種 1／否 0）、Y（健康結果，連續）、X（已測共變項）、W（NCO）、Z（NCE）。
          真實因果效應 TRUE_TAU（這裡 1.0）。U 為未測，不放進輸出。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 67
N = 8000
TRUE_TAU = 1.0            # true causal effect of A on Y
# treatment model: unmeasured U (and measured X) drive vaccination → confounding
A0, A_U, A_X = -0.2, 1.1, 0.4
# outcome model: A's true effect = TRUE_TAU; U confounds (B_U), X measured
B_U, B_X = 1.3, 0.6
# negative controls: both are proxies of U (no A→W path, no Z→Y path)
W0, W_U = 0.0, 1.2        # NCO W = W0 + W_U*U + noise         (NO A term)
Z0, Z_U = 0.0, 1.15       # NCE Z = Z0 + Z_U*U + noise         (NO Y path)
COLUMNS = ["pid", "A", "Y", "X", "W", "Z"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "demo_nc.csv")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(seed=SEED, n=N, true_tau=TRUE_TAU, conf=1.0):
    """Simulate data with an UNMEASURED confounder U. `conf` scales how strongly U drives
    both vaccination and the outcome (the confounding strength). NCO W and NCE Z are pure
    proxies of U (W has no A effect; Z has no Y effect). Truth = `true_tau`.
    Naive Y~A+X is biased by U; the A→W 'effect' is the bias signal; proximal P2SLS
    (using W and Z) recovers the truth."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0.0, 1.0, n)                              # UNMEASURED confounder
    X = rng.normal(0.0, 1.0, n)                              # measured covariate
    A = (rng.random(n) < _sig(A0 + A_U * conf * U + A_X * X)).astype(float)
    Y = true_tau * A + B_U * conf * U + B_X * X + rng.normal(0, 1.0, n)
    W = W0 + W_U * U + rng.normal(0, 1.0, n)                 # negative-control OUTCOME (no A)
    Z = Z0 + Z_U * U + rng.normal(0, 1.0, n)                 # negative-control EXPOSURE (no Y)
    return pd.DataFrame({
        "pid": np.arange(n), "A": A, "Y": np.round(Y, 4), "X": np.round(X, 4),
        "W": np.round(W, 4), "Z": np.round(Z, 4),
    }, columns=COLUMNS)


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    A = df.A.values; Y = df.Y.values; X = df.X.values; W = df.W.values; Z = df.Z.values
    def ols(Xcols, y):
        M = np.column_stack([np.ones(len(y))] + Xcols)
        return np.linalg.lstsq(M, y, rcond=None)[0]
    naive = ols([A, X], Y)[1]
    detect = ols([A, X], W)[1]                               # A→W should be ~0 if no confounding
    # P2SLS: stage1 W~A+Z+X ; stage2 Y~A+What+X
    b1 = ols([A, Z, X], W); What = b1[0] + b1[1]*A + b1[2]*Z + b1[3]*X
    prox = ols([A, What, X], Y)[1]
    print(f"wrote {OUT}  ({len(df)} rows)")
    print(f"naive Y~A     = {naive:.3f}   (true {TRUE_TAU})")
    print(f"A->W detect   = {detect:.3f}   (should be ~0; nonzero = bias signal)")
    print(f"proximal P2SLS= {prox:.3f}   (true {TRUE_TAU})")
