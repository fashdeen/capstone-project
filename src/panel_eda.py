"""panel_eda.py — panel-level EDA helpers (integrity, relationships, stationarity,
heterogeneity, multicollinearity)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------- Section 0: integrity ----------
def integrity(panel):
    n_ta, n_q = panel["ta_key"].nunique(), panel["q"].nunique()
    return {"rows": len(panel), "TAs": n_ta, "quarters": n_q,
            "balanced": len(panel) == n_ta * n_q,
            "duplicates": bool(panel.duplicated(["ta_key", "q"]).any()),
            "coverage": f"{panel['q'].min()}–{panel['q'].max()}"}

def missingness(panel):
    m = panel.isna().sum(); m = m[m > 0].sort_values(ascending=False)
    return pd.DataFrame({"n_missing": m, "pct": (100 * m / len(panel)).round(2)})

def outlier_scan(panel, cols):
    rows = []
    for c in cols:
        s = panel[c].dropna()
        rows.append({"variable": c, "min": s.min(), "p1": s.quantile(.01),
                     "median": s.median(), "p99": s.quantile(.99), "max": s.max()})
    return pd.DataFrame(rows).round(3).set_index("variable")

# ---------- Section 1: co-movement ----------
def _within(df, x, y):
    d = df[[x, y, "ta_key"]].dropna()
    dx = d[x] - d.groupby("ta_key")[x].transform("mean")
    dy = d[y] - d.groupby("ta_key")[y].transform("mean")
    return dx.corr(dy), len(d)

def comovement(panel, preds, lags, outcome="demand_rate"):
    rows = []
    for v in preds:
        for L in lags:
            c = f"{v}_lag{L}"; d = panel[[c, outcome]].dropna()
            pr = d[c].corr(d[outcome]); wr, n = _within(panel, c, outcome)
            rows.append({"predictor": v, "lag": L, "pooled_r": round(pr, 2),
                         "within_TA_r": round(wr, 2), "n": n})
    return pd.DataFrame(rows)

# ---------- Section 2: lag alignment ----------
def self_within(panel, x, y):
    r, _ = _within(panel, x, y); return r

def lag_decay(panel, preds, lags, outcome="demand_rate"):
    rows = []
    for v in preds:
        rs = [self_within(panel, f"{v}_lag{L}", outcome) for L in lags]; a = np.abs(rs)
        rows.append({"predictor": v, **{f"lag{L}": round(r, 2) for L, r in zip(lags, rs)},
                     "best_lag": lags[int(np.argmax(a))], "decay_range": round(a.max() - a.min(), 3)})
    return pd.DataFrame(rows)

# ---------- Section 3: stationarity ----------
def panel_adf(panel, col, diff=False, min_len=12):
    ps = []
    for ta, g in panel.sort_values("q").groupby("ta_key"):
        s = g[col].dropna()
        if diff: s = s.diff().dropna()
        if s.nunique() <= 1 or len(s) < min_len: continue
        try: ps.append(adfuller(s, autolag="AIC")[1])
        except Exception: pass
    ps = np.array(ps); return len(ps), round((ps < 0.05).mean(), 2)

def stationarity_table(panel, series_map):
    rows = []
    for name, col in series_map.items():
        nL, sL = panel_adf(panel, col, False); nD, sD = panel_adf(panel, col, True)
        rows.append({"series": name, "n_TA": nL,
                     "pct_stationary_levels": f"{sL:.0%}", "pct_stationary_diffs": f"{sD:.0%}"})
    return pd.DataFrame(rows).set_index("series")

# ---------- Section 4: heterogeneity ----------
def ta_relationship(panel, pred="geom_mean_rent_lag1", outcome="demand_rate", min_len=12):
    rows = []
    for ta, g in panel.sort_values("q").groupby("ta_key"):
        g = g.dropna(subset=[pred, outcome])
        if len(g) < min_len: continue
        r_lvl = g[pred].corr(g[outcome])
        gc = pd.DataFrame({"dp": g[pred].diff(), "do": g[outcome].diff()}).dropna()
        rows.append({"ta_key": ta, "ta_name": g["ta_name"].iloc[0],
                     "r_levels": round(r_lvl, 2), "r_changes": round(gc["dp"].corr(gc["do"]), 2)})
    return pd.DataFrame(rows)

# ---------- Section 5: multicollinearity ----------
def predictor_corr(panel, cols):
    return panel[cols].dropna().corr()

def vif_table(panel, cols):
    X = panel[cols].dropna(); Xv = (X - X.mean()) / X.std()
    return pd.DataFrame({"predictor": cols,
                         "VIF": [round(variance_inflation_factor(Xv.values, i), 2)
                                 for i in range(len(cols))]}).set_index("predictor")
