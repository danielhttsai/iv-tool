"""Generate the built-in DEMO cohort for the Prior Event Rate Ratio (PERR) tabs.

Fully fictional teaching scenario — NOT real data. A drug is started by some
patients (treated) and not others (control). Sicker patients (an unmeasured
'frailty') are BOTH more likely to be put on the drug AND more likely to have the
event — confounding by indication. PERR compares the treated-vs-control event-rate
ratio in a PRIOR period (before anyone is treated) and a POST period (after), and
divides them: time-invariant, multiplicative confounding cancels.

    每個病人有：group（1＝之後會用藥的處置組，0＝對照組）、看不見的 frailty（體弱），
    以及兩段期間的事件數與人時：
      events_prior / pt_prior （事前期：兩組都還沒用藥）
      events_post  / pt_post  （事後期：處置組用藥，對照組沒有）
    真實因果率比 RR = 0.70（藥有保護效果）。事前期率比反映純混淆（處置組較體弱→較高）。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 41
N = 8000
TRUE_RR = 0.70          # true causal rate ratio of the drug on the event
BASE_RATE = 0.06        # baseline event rate per person-year (rare-ish)
G_FRAIL = np.log(3.0)   # frailty multiplies the rate ~3x
PT = 3.0                # person-years observed in each period

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "demo_perr.csv")

COLUMNS = ["pid", "group", "events_prior", "pt_prior", "events_post", "pt_post"]


def generate(n=N, seed=SEED, true_rr=TRUE_RR, drift=0.0):
    """drift = how much the frailty (confounder) effect CHANGES from prior to post.
    0 = time-invariant confounding (PERR recovers the truth);
    >0 = the confounder effect grows over time (PERR becomes biased — P1 fails)."""
    rng = np.random.default_rng(seed)
    frail = rng.binomial(1, 0.4, n)
    # sicker (frail) patients are more likely to be started on the drug
    p_treat = 1.0 / (1.0 + np.exp(-(-0.2 + 1.6 * frail)))
    group = rng.binomial(1, p_treat)

    # prior period: nobody is treated yet -> rate depends only on frailty
    lam_prior = BASE_RATE * np.exp(G_FRAIL * frail)
    events_prior = rng.poisson(lam_prior * PT)

    # post period: treated get the drug (true_rr); frailty effect may drift
    g_post = G_FRAIL * (1.0 + float(drift))
    lam_post = BASE_RATE * np.exp(g_post * frail) * np.where(group == 1, true_rr, 1.0)
    events_post = rng.poisson(lam_post * PT)

    return pd.DataFrame({
        "pid": np.arange(n), "group": group,
        "events_prior": events_prior, "pt_prior": np.full(n, PT),
        "events_post": events_post, "pt_post": np.full(n, PT),
    }, columns=COLUMNS)


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    g = df.groupby("group")
    rTp = g.get_group(1).events_prior.sum() / g.get_group(1).pt_prior.sum()
    rCp = g.get_group(0).events_prior.sum() / g.get_group(0).pt_prior.sum()
    rTs = g.get_group(1).events_post.sum() / g.get_group(1).pt_post.sum()
    rCs = g.get_group(0).events_post.sum() / g.get_group(0).pt_post.sum()
    print(f"wrote {OUT}  ({len(df)} people)")
    print("RR_prior =", round(rTp / rCp, 3), " RR_post =", round(rTs / rCs, 3))
    print("PERR =", round((rTs / rCs) / (rTp / rCp), 3), " (true RR =", TRUE_RR, ")")
