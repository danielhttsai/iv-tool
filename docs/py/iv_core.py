"""IV computation core — Python port of iv_example.R.

Implements naive regression, first stage, reduced form, the Wald estimator,
the manual two-stage estimator, and 2SLS (with/without covariates) using only
numpy + scipy so it runs on any Python without statsmodels/linearmodels.

Sugar-tax example variables:
    Y = hdl_change   (outcome)
    A = low_sugar    (treatment)
    Z = sugartax     (instrument)
    covariates: age, sex, race, smokeyrs, exercise
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from i18n import t


# ---------------------------------------------------------------------------
# Design-matrix helpers
# ---------------------------------------------------------------------------
def _column(df, name):
    return np.asarray(df[name], dtype=float)


def _design(df, names, add_const=True):
    """Build a design matrix from a list of column names (all numeric)."""
    cols = []
    labels = []
    if add_const:
        cols.append(np.ones(len(df)))
        labels.append("const")
    for nm in names:
        cols.append(_column(df, nm))
        labels.append(nm)
    X = np.column_stack(cols) if cols else np.empty((len(df), 0))
    return X, labels


# ---------------------------------------------------------------------------
# Ordinary least squares with classical standard errors
# ---------------------------------------------------------------------------
def ols(y, X, labels):
    """Classical OLS. Returns a dict with coefficients, SE, t, p and fit stats."""
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof if dof > 0 else np.nan
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan

    return {
        "labels": labels,
        "coef": {labels[i]: float(beta[i]) for i in range(k)},
        "se": {labels[i]: float(se[i]) for i in range(k)},
        "t": {labels[i]: float(tval[i]) for i in range(k)},
        "p": {labels[i]: float(pval[i]) for i in range(k)},
        "n": int(n),
        "dof": int(dof),
        "rss": rss,
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "sigma2": float(sigma2),
        "_beta": beta,
        "_XtX_inv": XtX_inv,
    }


def _terms_list(res, exclude=("const",)):
    return [
        {
            "term": lbl,
            "coef": res["coef"][lbl],
            "se": res["se"][lbl],
            "t": res["t"][lbl],
            "p": res["p"][lbl],
        }
        for lbl in res["labels"]
        if lbl not in exclude
    ]


# ---------------------------------------------------------------------------
# iv_example.R step-by-step
# ---------------------------------------------------------------------------
def naive_regression(df, Y, A, covariates, lang="zh"):
    """glm(Y ~ A + covariates) — the confounded comparison."""
    X, labels = _design(df, [A] + list(covariates))
    res = ols(_column(df, Y), X, labels)
    coef = res["coef"][A]
    return {
        "estimate": coef,
        "se": res["se"][A],
        "p": res["p"][A],
        "terms": _terms_list(res),
        "n": res["n"],
        "interpretation": t(
            lang,
            f"未調整（naive）估計：有接受處置者其結果平均高 {coef:.2f} 單位。"
            "此估計只是把『有接受處置』與『沒有』的人直接相比，會被未量測的干擾因子"
            "（例如本來就比較注重健康的人）灌水，通常高估真正的因果效應。",
            f"Naive estimate: those who received the treatment score on average "
            f"{coef:.2f} units higher. This simply compares treated vs. untreated "
            "people directly, so it is inflated by unmeasured confounders (e.g. "
            "people who were already more health-conscious) and usually overstates "
            "the true causal effect.",
        ),
    }


def first_stage(df, A, Z, covariates=(), lang="zh"):
    """lm(A ~ Z + covariates). Returns the instrument coefficient and its F-stat."""
    X, labels = _design(df, [Z] + list(covariates))
    res = ols(_column(df, A), X, labels)
    coef = res["coef"][Z]
    tv = res["t"][Z]
    f_stat = tv * tv  # partial F for a single excluded instrument == t^2
    return {
        "coef": coef,
        "se": res["se"][Z],
        "t": tv,
        "p": res["p"][Z],
        "f_stat": f_stat,
        "complier_share": coef,  # linear-probability first stage
        "n": res["n"],
        "interpretation": t(
            lang,
            f"第一階段：工具每變動一單位，處置比例改變 {coef:.3f}"
            f"（約 {coef*100:.1f}% 的人是 complier，會因工具而改變行為）。"
            f"F 統計量 = {f_stat:.1f}（>10 代表工具夠強，不是弱工具）。",
            f"First stage: a one-unit change in the instrument shifts the treatment "
            f"probability by {coef:.3f} (about {coef*100:.1f}% of people are "
            f"compliers who change behaviour because of the instrument). "
            f"F-statistic = {f_stat:.1f} (>10 means the instrument is strong, not weak).",
        ),
    }


def reduced_form(df, Y, Z, covariates=(), lang="zh"):
    """lm(Y ~ Z + covariates) — the intention-to-treat / reduced form."""
    X, labels = _design(df, [Z] + list(covariates))
    res = ols(_column(df, Y), X, labels)
    coef = res["coef"][Z]
    return {
        "coef": coef,
        "se": res["se"][Z],
        "p": res["p"][Z],
        "n": res["n"],
        "interpretation": t(
            lang,
            f"簡化式（reduced form）：工具每變動一單位，結果平均改變 {coef:.3f} 單位。"
            "這是工具對結果的『總效果』，包含了透過處置傳遞的部分。",
            f"Reduced form: a one-unit change in the instrument changes the outcome "
            f"by {coef:.3f} units on average. This is the instrument's total effect "
            "on the outcome, including the part that flows through the treatment.",
        ),
    }


def wald_estimator(df, Y, A, Z, lang="zh"):
    """Wald = reduced-form coef / first-stage coef."""
    fs = first_stage(df, A, Z)
    rf = reduced_form(df, Y, Z)
    est = rf["coef"] / fs["coef"]
    return {
        "estimate": est,
        "reduced_coef": rf["coef"],
        "first_coef": fs["coef"],
        "interpretation": t(
            lang,
            f"Wald 估計 = 簡化式係數 ÷ 第一階段係數 = "
            f"{rf['coef']:.3f} ÷ {fs['coef']:.3f} = {est:.2f}。"
            "在 complier 族群中，處置使結果平均改變約這個數值（這就是 LATE）。",
            f"Wald estimate = reduced-form coef ÷ first-stage coef = "
            f"{rf['coef']:.3f} ÷ {fs['coef']:.3f} = {est:.2f}. "
            "Among compliers, the treatment changes the outcome by about this much "
            "on average (this is the LATE).",
        ),
    }


def two_stage_manual(df, Y, A, Z, covariates=(), lang="zh"):
    """predict(first.stage) then regress Y on the fitted treatment."""
    X1, lab1 = _design(df, [Z] + list(covariates))
    fs = ols(_column(df, A), X1, lab1)
    a_hat = X1 @ fs["_beta"]
    # second stage: Y ~ a_hat (+ covariates)
    extra = [_column(df, c) for c in covariates]
    cols = [np.ones(len(df)), a_hat] + extra
    X2 = np.column_stack(cols)
    lab2 = ["const", A + "_hat"] + list(covariates)
    res = ols(_column(df, Y), X2, lab2)
    est = res["coef"][A + "_hat"]
    return {
        "estimate": est,
        "interpretation": t(
            lang,
            f"兩階段手動估計：先用工具預測處置，再把『預測的處置』放進迴歸，"
            f"得到 {est:.2f}（與 Wald／2SLS 一致）。注意：手動算法的標準誤是錯的，"
            "正式分析應使用 2SLS。",
            f"Manual two-stage estimate: first predict the treatment from the "
            f"instrument, then regress the outcome on the predicted treatment, "
            f"giving {est:.2f} (the same as Wald / 2SLS). Note: the standard errors "
            "from this manual route are wrong — use 2SLS for formal analysis.",
        ),
    }


def iv_2sls(df, Y, A, Z, covariates=(), lang="zh"):
    """Just-identified 2SLS with correct standard errors.

    Mirrors ivreg(Y ~ A + covs | Z + covs).
    """
    covariates = list(covariates)
    y = _column(df, Y)
    n = len(df)
    const = np.ones(n)

    # Regressors X = [const, A, covariates]; instruments Zmat = [const, Z, covariates]
    X = np.column_stack([const, _column(df, A)] + [_column(df, c) for c in covariates])
    Zmat = np.column_stack([const, _column(df, Z)] + [_column(df, c) for c in covariates])
    labels = ["const", A] + covariates

    ZtZ_inv = np.linalg.pinv(Zmat.T @ Zmat)
    Pz_X = Zmat @ (ZtZ_inv @ (Zmat.T @ X))  # projection of X onto instrument space
    A_mat = Pz_X.T @ X
    A_inv = np.linalg.pinv(A_mat)
    beta = A_inv @ (Pz_X.T @ y)

    resid = y - X @ beta
    k = X.shape[1]
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Pz_X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)

    idx = labels.index(A)
    est = float(beta[idx])
    ci_lo = est - 1.96 * se[idx]
    ci_hi = est + 1.96 * se[idx]
    terms = [
        {
            "term": labels[i],
            "coef": float(beta[i]),
            "se": float(se[i]),
            "t": float(tval[i]),
            "p": float(pval[i]),
        }
        for i in range(k)
        if labels[i] != "const"
    ]
    return {
        "estimate": est,
        "se": float(se[idx]),
        "t": float(tval[idx]),
        "p": float(pval[idx]),
        "ci": [float(ci_lo), float(ci_hi)],
        "terms": terms,
        "n": int(n),
        "covariates": covariates,
        "interpretation": t(
            lang,
            f"2SLS（工具變數）估計：在 complier 族群中，處置使結果平均改變 "
            f"{est:.2f} 單位（95% 信賴區間 {ci_lo:.2f} 到 {ci_hi:.2f}）。"
            + ("（已調整共變項）" if covariates else "（未調整共變項）"),
            f"2SLS (instrumental-variable) estimate: among compliers, the treatment "
            f"changes the outcome by {est:.2f} units on average "
            f"(95% confidence interval {ci_lo:.2f} to {ci_hi:.2f})."
            + (" (covariate-adjusted)" if covariates else " (no covariate adjustment)"),
        ),
    }


def full_analysis(df, Y, A, Z, covariates=(), lang="zh"):
    """Run every step of iv_example.R and return a single structured payload."""
    covariates = list(covariates)
    naive = naive_regression(df, Y, A, covariates, lang=lang)
    fs = first_stage(df, A, Z, lang=lang)
    rf = reduced_form(df, Y, Z, lang=lang)
    wald = wald_estimator(df, Y, A, Z, lang=lang)
    ts = two_stage_manual(df, Y, A, Z, lang=lang)
    iv = iv_2sls(df, Y, A, Z, lang=lang)
    iv_cov = iv_2sls(df, Y, A, Z, covariates, lang=lang) if covariates else None

    return {
        "columns": {"outcome": Y, "treatment": A, "instrument": Z, "covariates": covariates},
        "naive": naive,
        "first_stage": fs,
        "reduced_form": rf,
        "wald": wald,
        "two_stage_manual": ts,
        "iv": iv,
        "iv_with_covariates": iv_cov,
        "n": len(df),
    }
