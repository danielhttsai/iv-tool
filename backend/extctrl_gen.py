"""Synthetic data for the External-Control teaching demo.

You ran a SINGLE-ARM study: every patient got the new treatment, so there is no
concurrent control group. To estimate an effect you borrow an EXTERNAL (historical
/ registry / real-world) control arm of UNTREATED patients. The catch: the external
controls have a DIFFERENT covariate distribution from your trial (e.g. they are
sicker, or from an earlier era), so a naive trial-vs-external comparison is
confounded by that imbalance.

Potential untreated outcome is the same function of X in both groups (exchangeable
given X): Y(0) = B0 + B_X * X + noise. Treatment adds TAU. So
  trial (all treated):     Y = B0 + TAU + B_X * X_trial + noise
  external (all control):  Y = B0       + B_X * X_ext   + noise
and the true effect is TAU. Purely synthetic, for teaching only.
"""
import numpy as np

SEED = 9
N_TRIAL = 1500       # single-arm trial: everyone is treated
N_EXT = 1500         # external control pool: everyone is untreated
TAU = 2.0            # the true treatment effect we want to recover
B0 = 0.0             # baseline outcome level
B_X = 1.0            # effect of the prognostic covariate X on the outcome
MU_TRIAL = 0.5       # the trial enrols higher-X (e.g. fitter) patients...
MU_EXT = -0.5        # ...the external controls are lower-X (e.g. sicker) → imbalance


def generate(n_trial=N_TRIAL, n_ext=N_EXT, mu_trial=MU_TRIAL, mu_ext=MU_EXT, seed=SEED):
    rng = np.random.default_rng(seed)
    # single-arm trial: all treated
    xt = rng.normal(mu_trial, 1.0, n_trial)
    yt = B0 + TAU + B_X * xt + rng.normal(0.0, 1.0, n_trial)
    # external controls: all untreated, shifted covariate distribution
    xe = rng.normal(mu_ext, 1.0, n_ext)
    ye = B0 + B_X * xe + rng.normal(0.0, 1.0, n_ext)
    return {
        "trial_X": xt, "trial_Y": yt, "ext_X": xe, "ext_Y": ye,
        "true_tau": float(TAU), "mu_trial": float(mu_trial), "mu_ext": float(mu_ext),
    }
