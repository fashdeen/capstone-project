"""
rq3.py — RQ3 incremental value of rental-market indicators over register-only forecasts.

Question
--------
Do rental-market indicators improve forecast accuracy beyond the register's own
history alone? (Rung 2 vs Rung 1.) The naive baseline is also reported, so we can
separately say whether the fuller model is *useful* (beats naive), not only whether
market data *helps* (beats register-only).

Design (settled in supervision)
-------------------------------
  models      : rw_flat  — naive random walk (carry last level; the usefulness bar)
                reg      — register-only AR(1) on the change, TA effects, no time effects
                mkt      — reg PLUS the four RQ1 market signals at lag 1
  market set  : rent_growth_lag1, net_bond_flow_lag1, dwelling_units_lag1,
                multi_unit_share_lag1  (same features as RQ1)
  primary     : horizon 1 — market lag-1 values are genuinely known at forecast time.
                Horizons 2-4 are SECONDARY and caveated: multi-step needs future market
                values, so the last known market state is carried forward (stale).
  validation  : expanding-window walk-forward, first origin t0 (~q20), all TAs.
  metrics     : out-of-sample RMSE / MAE per horizon; skill vs naive; incremental
                skill of mkt over reg; and a Diebold-Mariano test (reg vs mkt)
                CLUSTERED BY TARGET QUARTER to respect cross-sectional dependence.

Framing is prediction, never causation.
"""
import numpy as np
import pandas as pd
from scipy import stats

import config

LEVEL, CHANGE = "demand_rate", "demand_rate_change"
ENTITY, TIME = "ta_key", "q"
MARKET = ["rent_growth_lag1", "net_bond_flow_lag1",
          "dwelling_units_lag1", "multi_unit_share_lag1"]


def load_panel():
    return pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")


def make_grids(panel):
    qs = sorted(panel[TIME].unique())
    qidx = {q: i for i, q in enumerate(qs)}
    p = panel.copy()
    p["ti"] = p[TIME].map(qidx)
    L = p.pivot(index="ti", columns=ENTITY, values=LEVEL).sort_index()
    C = p.pivot(index="ti", columns=ENTITY, values=CHANGE).sort_index()
    M = {c: p.pivot(index="ti", columns=ENTITY, values=c).sort_index() for c in MARKET}
    return L, C, M, qs


# ── fixed-effects AR(1) (+ optional market lag-1 exog), within-TA demeaned ──
def fit(Ctrain, M, tas, exog):
    rows = []
    for i in tas:
        s = Ctrain[i].dropna()
        sidx = set(s.index)
        for t in s.index:
            if (t - 1) in sidx:
                mk = [M[c][i][t] for c in exog]
                if any(pd.isna(mk)):
                    continue
                rows.append([i, s[t], s[t - 1]] + mk)
    if len(rows) < 50:
        return None
    feats = ["x0"] + list(exog)
    df = pd.DataFrame(rows, columns=["i", "y", "x0"] + list(exog))
    for c in ["y"] + feats:
        df[c + "_d"] = df[c] - df.groupby("i")[c].transform("mean")
    b, *_ = np.linalg.lstsq(df[[c + "_d" for c in feats]].values, df["y_d"].values, rcond=None)
    ci = {i: g["y"].mean() - sum(b[k] * g[feats[k]].mean() for k in range(len(feats)))
          for i, g in df.groupby("i")}
    return b, ci


def fcast(i, t, L, C, M, fitres, exog, h):
    b, ci = fitres
    if i not in ci or np.isnan(C[i][t]):
        return np.nan
    mk = [M[c][i][t + 1] for c in exog]     # market known at origin t (exact for h=1; carried forward for h>1)
    if any(pd.isna(mk)):
        return np.nan
    prev, ds = C[i][t], []
    for _ in range(h):
        d = ci[i] + b[0] * prev + sum(b[k + 1] * mk[k] for k in range(len(exog)))
        ds.append(d)
        prev = d
    return L[i][t] + np.sum(ds)


# ───────────────────────── walk-forward + scoring ─────────────────────────
def walk_forward(L, C, M, t0=20, horizons=(1, 2, 3, 4)):
    T = L.shape[0]
    tas = list(L.columns)
    models = ["rw_flat", "reg", "mkt"]
    err = {m: {h: {"se": [], "ae": []} for h in horizons} for m in models}
    dm = {h: [] for h in horizons}     # (target_quarter, e_reg^2 - e_mkt^2)
    for t in range(t0, T):
        fr = fit(C.loc[:t], M, tas, [])
        fm = fit(C.loc[:t], M, tas, MARKET)
        for i in tas:
            Lt = L[i][t]
            if np.isnan(Lt):
                continue
            for h in horizons:
                if t + h >= T:
                    continue
                a = L[i][t + h]
                if np.isnan(a):
                    continue
                fc = {"rw_flat": Lt,
                      "reg": fcast(i, t, L, C, M, fr, [], h) if fr else np.nan,
                      "mkt": fcast(i, t, L, C, M, fm, MARKET, h) if fm else np.nan}
                for m in models:
                    if not np.isnan(fc[m]):
                        err[m][h]["se"].append((a - fc[m]) ** 2)
                        err[m][h]["ae"].append(abs(a - fc[m]))
                if not np.isnan(fc["reg"]) and not np.isnan(fc["mkt"]):
                    dm[h].append((t + h, (a - fc["reg"]) ** 2 - (a - fc["mkt"]) ** 2))
    return err, models, dm


def diebold_mariano(dm_h):
    """Clustered-by-quarter DM: H0 reg and mkt have equal forecast accuracy.
    mean loss diff = mean(e_reg^2 - e_mkt^2); >0 => mkt better."""
    d = pd.DataFrame(dm_h, columns=["q", "d"])
    dbar, N = d["d"].mean(), len(d)
    v = sum((g["d"].sum() - len(g) * dbar) ** 2 for _, g in d.groupby("q")) / N ** 2
    DM = dbar / np.sqrt(v)
    return {"DM": DM, "p_value": 2 * (1 - stats.norm.cdf(abs(DM))),
            "mean_loss_diff": dbar, "n": N}


def score(err, models, dm, horizons=(1, 2, 3, 4)):
    rmse = lambda m, h: np.sqrt(np.mean(err[m][h]["se"])) if err[m][h]["se"] else np.nan
    mae = lambda m, h: np.mean(err[m][h]["ae"]) if err[m][h]["ae"] else np.nan
    idx = [f"h{h}" for h in horizons]
    R = pd.DataFrame({m: [rmse(m, h) for h in horizons] for m in models}, index=idx).T
    Ma = pd.DataFrame({m: [mae(m, h) for h in horizons] for m in models}, index=idx).T
    skill_naive = pd.DataFrame({m: [1 - rmse(m, h) / rmse("rw_flat", h) for h in horizons]
                                for m in ["reg", "mkt"]}, index=idx).T
    incr = pd.Series([1 - rmse("mkt", h) / rmse("reg", h) for h in horizons], index=idx,
                     name="mkt_vs_reg")
    dm_res = {f"h{h}": diebold_mariano(dm[h]) for h in horizons}
    return {"rmse": R.round(4), "mae": Ma.round(4),
            "skill_vs_naive": skill_naive.round(3),
            "incremental_skill_mkt_vs_reg": incr.round(3), "dm": dm_res}


def run(panel=None, t0=20):
    panel = load_panel() if panel is None else panel
    L, C, M, qs = make_grids(panel)
    err, models, dm = walk_forward(L, C, M, t0=t0)
    out = {"t0": t0, "origin_quarter": qs[t0], "err": err, "models": models, "dm_raw": dm}
    out.update(score(err, models, dm))
    return out


def print_report(out=None):
    out = run() if out is None else out
    print(f"RQ3 — incremental value of market signals. Walk-forward, first forecast {out['origin_quarter']}.")
    print("Models: rw_flat (naive), reg (register-only), mkt (register + 4 market signals).\n")
    print("RMSE by horizon:"); print(out["rmse"].to_string(), "\n")
    print("Skill vs naive (both should be checked for absolute usefulness):")
    print(out["skill_vs_naive"].to_string(), "\n")
    print("RQ3 direct — incremental skill of mkt over reg (1 - RMSE_mkt/RMSE_reg; >0 = market helps):")
    print(out["incremental_skill_mkt_vs_reg"].to_string(), "\n")
    print("Diebold-Mariano (reg vs mkt), clustered by target quarter  [h1 = primary]:")
    for h, r in out["dm"].items():
        print(f"  {h}: DM = {r['DM']:+.2f}  p = {r['p_value']:.3f}  "
              f"(mean loss diff = {r['mean_loss_diff']:+.4f}; >0 => market better; n = {r['n']})")
    return out
