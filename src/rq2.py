"""
rq2.py — RQ2 forecastability of regional register demand from its own recent history.

Question
--------
Can regional register demand be forecast from its own recent history more accurately
than a naive baseline, and over what horizon does that accuracy hold?

Design (settled in supervision)
-------------------------------
  target      : demand_rate (level, per 1,000) — forecast h quarters ahead
  baseline    : naive random walk = carry last level forward (predict no change).
                Also reported vs a random-walk-WITH-drift (harder bar for a
                trending series): last level + h * the TA's mean historical change.
  forecaster  : register-only. AR(p) on the STATIONARY change series with TA
                effects (drift) and NO time effects (a future quarter's time
                effect is unknowable -> would leak the future). Iterated h steps
                and cumulated back to a level. p = 1 primary, p = 2 tested.
  validation  : expanding-window walk-forward. First forecast origin at t0 (~q20)
                so the model has enough history; forecast every TA at each origin;
                horizons 1..4. Only past data is used to fit each forecast.
  metrics     : out-of-sample RMSE and MAE per horizon, and skill = 1 - RMSE_model
                / RMSE_baseline (>0 means the model beats the baseline).

RQ3 hook: fe_ar_fit / fe_ar_forecast are written so market-signal columns can be
added as exogenous regressors without reworking the engine.
Framing is prediction, never causation.
"""
import numpy as np
import pandas as pd

import config

LEVEL, CHANGE = "demand_rate", "demand_rate_change"
ENTITY, TIME = "ta_key", "q"


# ───────────────────────── data grids ─────────────────────────
def load_panel():
    return pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")


def make_grids(panel):
    """Return level grid L, change grid C (both ti x TA), and the quarter list."""
    qs = sorted(panel[TIME].unique())
    qidx = {q: i for i, q in enumerate(qs)}
    p = panel.copy()
    p["ti"] = p[TIME].map(qidx)
    L = p.pivot(index="ti", columns=ENTITY, values=LEVEL).sort_index()
    C = p.pivot(index="ti", columns=ENTITY, values=CHANGE).sort_index()
    return L, C, qs


# ───────────────────────── register-only AR(p) with TA effects ─────────────────────────
def fe_ar_fit(C, tas, plags):
    """Fixed-effects AR(plags) on the change series. Returns (phi, c_i) or None."""
    rows = []
    for i in tas:
        s = C[i].dropna()
        sidx = set(s.index)
        for t in s.index:
            if all((t - l) in sidx for l in range(1, plags + 1)):
                rows.append([i, s[t]] + [s[t - l] for l in range(1, plags + 1)])
    if len(rows) < 50:
        return None
    df = pd.DataFrame(rows, columns=["i", "y"] + [f"x{l}" for l in range(1, plags + 1)])
    for c in ["y"] + [f"x{l}" for l in range(1, plags + 1)]:
        df[c + "_d"] = df[c] - df.groupby("i")[c].transform("mean")   # within-TA demean = TA effects
    Xd = df[[f"x{l}_d" for l in range(1, plags + 1)]].values
    yd = df["y_d"].values
    phi, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    ci = {i: g["y"].mean() - sum(phi[l - 1] * g[f"x{l}"].mean() for l in range(1, plags + 1))
          for i, g in df.groupby("i")}
    return phi, ci


def fe_ar_forecast(i, t, L, C, phi, ci, plags, h):
    """Iterate the AR(p) h steps from origin t and cumulate to a level forecast."""
    hist = [C[i][t - l] if (t - l) >= 0 else np.nan for l in range(plags)]
    if any(np.isnan(hist)) or i not in ci:
        return np.nan
    ds = []
    for _ in range(h):
        d = ci[i] + sum(phi[l] * hist[l] for l in range(plags))
        ds.append(d)
        hist = [d] + hist[:-1]
    return L[i][t] + np.sum(ds)


# ───────────────────────── walk-forward + scoring ─────────────────────────
def walk_forward(L, C, t0=20, horizons=(1, 2, 3, 4), ar_lags=(1, 2)):
    T = L.shape[0]
    tas = list(L.columns)
    models = ["rw_flat", "rw_drift"] + [f"ar{p}_fe" for p in ar_lags]
    err = {m: {h: {"se": [], "ae": []} for h in horizons} for m in models}
    for t in range(t0, T):
        fits = {p: fe_ar_fit(C.loc[:t], tas, p) for p in ar_lags}
        for i in tas:
            Lt = L[i][t]
            if np.isnan(Lt):
                continue
            drift = C[i].loc[:t].mean()
            for h in horizons:
                if t + h >= T:
                    continue
                a = L[i][t + h]
                if np.isnan(a):
                    continue
                fc = {"rw_flat": Lt, "rw_drift": Lt + h * drift}
                for p in ar_lags:
                    fc[f"ar{p}_fe"] = (fe_ar_forecast(i, t, L, C, *fits[p], p, h)
                                       if fits[p] else np.nan)
                for m in models:
                    if not np.isnan(fc[m]):
                        err[m][h]["se"].append((a - fc[m]) ** 2)
                        err[m][h]["ae"].append(abs(a - fc[m]))
    return err, models


def score(err, models, horizons=(1, 2, 3, 4), baseline="rw_flat"):
    rmse = lambda m, h: np.sqrt(np.mean(err[m][h]["se"])) if err[m][h]["se"] else np.nan
    mae = lambda m, h: np.mean(err[m][h]["ae"]) if err[m][h]["ae"] else np.nan
    idx = [f"h{h}" for h in horizons]
    R = pd.DataFrame({m: [rmse(m, h) for h in horizons] for m in models}, index=idx).T
    M = pd.DataFrame({m: [mae(m, h) for h in horizons] for m in models}, index=idx).T
    S = pd.DataFrame({m: [1 - rmse(m, h) / rmse(baseline, h) for h in horizons]
                      for m in models if m != baseline}, index=idx).T
    n = {f"h{h}": len(err[baseline][h]["se"]) for h in horizons}
    return {"rmse": R.round(4), "mae": M.round(4), f"skill_vs_{baseline}": S.round(3), "n_test": n}


def run(panel=None, t0=20):
    panel = load_panel() if panel is None else panel
    L, C, qs = make_grids(panel)
    err, models = walk_forward(L, C, t0=t0)
    out = {"t0": t0, "origin_quarter": qs[t0], "err": err, "models": models}
    out.update(score(err, models, baseline="rw_flat"))
    out["skill_vs_rw_drift"] = score(err, models, baseline="rw_drift")["skill_vs_rw_drift"]
    return out


def print_report(out=None):
    out = run() if out is None else out
    print(f"RQ2 — forecastability: expanding-window walk-forward, first forecast at {out['origin_quarter']}")
    print("Target: demand_rate (level, per 1,000). Errors are out-of-sample, pooled over TAs.\n")
    print("RMSE by horizon:")
    print(out["rmse"].to_string(), "\n")
    print("MAE by horizon:")
    print(out["mae"].to_string(), "\n")
    print("Skill vs naive random walk  (1 - RMSE_model/RMSE_rw_flat; >0 beats naive):")
    print(out["skill_vs_rw_flat"].to_string(), "\n")
    print("Skill vs random-walk-with-drift (harder bar for a trending series):")
    print(out["skill_vs_rw_drift"].to_string(), "\n")
    print("Out-of-sample test points per horizon:", out["n_test"])
    return out
