"""瀏覽器端 API 橋接 —— 在 Pyodide 裡重現 app.py 的端點,完全不需要伺服器。

繁重的計算照舊由 backend 的 iv_core / assumptions / ml_iv / gen_data 負責,
這裡只是把 FastAPI 的路由邏輯改寫成一個可被 JavaScript 呼叫的 `route()` 函式。
回傳值是 JSON 字串,格式為 {"status": int, "body": ...},由前端的橋接包成 Response。
"""
from __future__ import annotations

import io
import json
import uuid

import numpy as np
import pandas as pd

import assumptions
import iv_core
import ml_iv
import gen_data

EXAMPLE_DEFAULTS = {
    "outcome": "health_score_change",
    "treatment": "vaccinated",
    "instrument": "vaccine_reminder",
    "covariates": ["age", "female", "bmi", "chronic_conditions", "income_band"],
}

DISCLAIMER = "⚠ 純屬虛構的合成示範資料,非真實病人/個資,僅供教學展示。"

_UPLOADS: dict[str, pd.DataFrame] = {}
_DEMO: pd.DataFrame | None = None


def _demo() -> pd.DataFrame:
    global _DEMO
    if _DEMO is None:
        _DEMO = gen_data.generate()
    return _DEMO


def _clean(obj):
    """Make numpy / NaN values JSON-safe (mirrors app.py._clean)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _load(source: str) -> pd.DataFrame:
    if source == "example":
        return _demo()
    df = _UPLOADS.get(source)
    if df is None:
        raise ValueError("找不到資料，請重新上傳。")
    return df


def _validate(df: pd.DataFrame, req: dict) -> None:
    needed = [req["outcome"], req["treatment"], req["instrument"], *req.get("covariates", [])]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("資料缺少欄位：" + ", ".join(missing))


# ---------------------------------------------------------------------------
# Endpoint implementations
# ---------------------------------------------------------------------------
def _example() -> dict:
    df = _demo()
    return {
        "columns": list(df.columns),
        "defaults": EXAMPLE_DEFAULTS,
        "n": len(df),
        "synthetic": True,
        "disclaimer": DISCLAIMER,
        "preview": df.head(8).to_dict(orient="records"),
        "story": {
            "instrument": "vaccine_reminder（住在被隨機抽中、收到免費接種提醒的社區）",
            "treatment": "vaccinated（是否接種疫苗）",
            "outcome": "health_score_change（一年後健康分數的變化）",
        },
    }


def _upload(text: str) -> dict:
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"無法讀取 CSV：{exc}")
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        raise ValueError("資料中找不到數值欄位。")
    token = uuid.uuid4().hex
    _UPLOADS[token] = df
    return {
        "token": token,
        "columns": df.columns.tolist(),
        "numeric_columns": numeric,
        "n": len(df),
        "preview": df.head(8).to_dict(orient="records"),
    }


def _analyze(req: dict) -> dict:
    df = _load(req.get("source", "example"))
    _validate(df, req)
    return iv_core.full_analysis(
        df, req["outcome"], req["treatment"], req["instrument"],
        req.get("covariates", []), lang=req.get("lang", "zh"),
    )


def _assumptions(req: dict) -> dict:
    df = _load(req.get("source", "example"))
    _validate(df, req)
    return assumptions.check_all(
        df, req["outcome"], req["treatment"], req["instrument"],
        req.get("covariates", []), lang=req.get("lang", "zh"),
    )


def _interactive(q: dict) -> dict:
    complier_share = float(np.clip(float(q.get("complier_share", 0.063)), 0.01, 0.5))
    strength = float(np.clip(float(q.get("strength", 1.0)), 0.2, 2.0))
    seed = int(q.get("seed", 7))
    rng = np.random.default_rng(seed)
    n = int(2000 * strength)
    true_late = 1.80

    Z = rng.binomial(1, 0.5, n)
    U = rng.standard_normal(n)
    p = np.clip(0.30 + complier_share * Z + 0.12 * U, 0.02, 0.98)
    A = rng.binomial(1, p)
    Y = true_late * A + 1.50 * U + rng.normal(0, 3.0, n)
    df = pd.DataFrame({"Y": Y, "A": A, "Z": Z})

    fs = iv_core.first_stage(df, "A", "Z")
    iv = iv_core.iv_2sls(df, "Y", "A", "Z")
    return {
        "complier_share": complier_share,
        "strength": strength,
        "n": n,
        "first_coef": fs["coef"],
        "f_stat": fs["f_stat"],
        "estimate": iv["estimate"],
        "ci": iv["ci"],
        "true_late": true_late,
    }


def _ml_synthesis(q: dict) -> dict:
    return ml_iv.synthesis_demo(
        n=int(q.get("n", 6000)),
        k_candidates=int(q.get("k_candidates", 12)),
        per_strength=float(q.get("per_strength", 0.03)),
        seed=int(q.get("seed", 7)),
    )


def _ml_nonlinear(q: dict) -> dict:
    return ml_iv.nonlinear_demo(n=int(q.get("n", 8000)), seed=int(q.get("seed", 11)))


def _ml_forbidden(q: dict) -> dict:
    return ml_iv.forbidden_demo(seed=int(q.get("seed", 7)))


def _ml_compare(q: dict) -> dict:
    return ml_iv.compare(seed=int(q.get("seed", 7)), lang=q.get("lang", "zh"))


_ROUTES = {
    ("GET", "/api/example"): lambda q, b: _example(),
    ("POST", "/api/upload"): lambda q, b: _upload(b.get("_csv_text", "")),
    ("POST", "/api/analyze"): lambda q, b: _analyze(b),
    ("POST", "/api/assumptions"): lambda q, b: _assumptions(b),
    ("GET", "/api/interactive"): lambda q, b: _interactive(q),
    ("GET", "/api/ml_synthesis"): lambda q, b: _ml_synthesis(q),
    ("GET", "/api/ml_nonlinear"): lambda q, b: _ml_nonlinear(q),
    ("GET", "/api/ml_forbidden"): lambda q, b: _ml_forbidden(q),
    ("GET", "/api/ml_compare"): lambda q, b: _ml_compare(q),
}


def route(method: str, path: str, query_json: str, body_json: str) -> str:
    """Dispatch one API call. Returns a JSON string {"status", "body"}."""
    try:
        query = json.loads(query_json) if query_json else {}
        body = json.loads(body_json) if body_json else {}
        handler = _ROUTES.get((method.upper(), path))
        if handler is None:
            return json.dumps({"status": 404, "body": {"detail": f"未知的 API 路徑：{path}"}},
                              ensure_ascii=False)
        data = handler(query, body)
        return json.dumps({"status": 200, "body": _clean(data)}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": 400, "body": {"detail": str(exc)}}, ensure_ascii=False)
