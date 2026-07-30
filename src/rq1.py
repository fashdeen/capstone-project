"""
rq1.py — RQ1 association model.

Estimand
--------
To what extent is the quarter-on-quarter CHANGE in social-housing register
demand associated with lagged open housing-market indicators, after absorbing
area (TA) and time (quarter) fixed effects. Framing is association / prediction,
never causation.

Specification (settled in supervision)
--------------------------------------
  outcome : demand_rate_change          demand_rate is I(1) -> differenced
  exog    : rent_growth_lag1            already a change  -> I(0)
            net_bond_flow_lag1          already a flow    -> I(0)
            dwelling_units_lag1         stationary flow   -> I(0)  (within-TA AR1 ~0.82)
            multi_unit_share_lag1       stationary share  -> I(0)  (within-TA AR1 ~0.27)
  effects : entity (TA) + time (quarter) — two-way fixed effects
  ident.  : within-quarter, cross-TA deviations (trend + common shocks absorbed)
  infer.  : cluster-robust by TA (primary; robust to within-TA serial correlation)
            Pesaran CD on residuals (is there residual cross-sectional dependence?)
            Driscoll-Kraay kernel SEs (robustness to c/s + serial dependence)
            population-weighted fit (sensitivity: does the small/large TA weighting matter?)

Open design choices, made explicit
-----------------------------------
  * Sample: complete-case on outcome + all four predictors. Differencing turns one
    suppressed quarter into two lost rows and hits small TAs harder, so build_frame()
    returns a missingness AUDIT split by TA size — report it, do not bury it.
  * Weighting: UNWEIGHTED is primary (each TA-quarter equal — honours the small-provider
    framing). Population-weighted is reported as a sensitivity, not the headline.
  Both are single-argument toggles below; revisit as robustness, not by editing logic.
"""
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS

import config

OUTCOME = "demand_rate_change"
EXOG = ["rent_growth_lag1", "net_bond_flow_lag1",
        "dwelling_units_lag1", "multi_unit_share_lag1"]
ENTITY, TIME = "ta_key", "q"
SMALL_REG_MEDIAN = 50   # small-TA cutoff — for the missingness audit only


# ───────────────────────── data ─────────────────────────
def load_panel():
    return pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")


def build_frame(panel):
    """Select RQ1 columns, sort, complete-case. Returns (frame, audit).

    audit: usable-row share split by TA size, so the small-TA cost of
    differencing + suppression is visible rather than silent.
    """
    keep = [ENTITY, "ta_name", TIME, OUTCOME, "register_count", "population"] + EXOG
    df = panel[keep].sort_values([ENTITY, TIME]).copy()

    need = [OUTCOME] + EXOG
    complete = df[need].notna().all(axis=1)

    med = df.groupby(ENTITY)["register_count"].transform("median")
    small = med < SMALL_REG_MEDIAN
    audit = (pd.DataFrame({"small_TA": small.values, "usable": complete.values})
             .groupby("small_TA")["usable"].agg(rows="size", usable="sum"))
    audit["pct_kept"] = (100 * audit["usable"] / audit["rows"]).round(1)

    frame = df.loc[complete].reset_index(drop=True)
    return frame, audit


def _panelize(frame):
    d = frame.copy()
    d["_t"] = pd.PeriodIndex(d[TIME], freq="Q").to_timestamp()   # date-like time index
    return d.set_index([ENTITY, "_t"])


# ───────────────────────── estimation ─────────────────────────
def fit(frame, weighted=False, cov="clustered"):
    """Two-way FE fit. cov: 'clustered' (by TA) | 'kernel' (Driscoll-Kraay) | 'unadjusted'."""
    d = _panelize(frame)
    kw = dict(entity_effects=True, time_effects=True, drop_absorbed=True)
    mod = (PanelOLS(d[OUTCOME], d[EXOG], weights=d["population"], **kw) if weighted
           else PanelOLS(d[OUTCOME], d[EXOG], **kw))
    if cov == "clustered":
        return mod.fit(cov_type="clustered", cluster_entity=True)
    if cov == "kernel":
        return mod.fit(cov_type="kernel", kernel="bartlett")
    return mod.fit()


def pesaran_cd(res, frame):
    """Pesaran (2004) CD test for cross-sectional dependence in residuals.
    Unbalanced variant: CD = sqrt(2/(N(N-1))) * sum_{i<j} sqrt(T_ij) * rho_ij  ~ N(0,1).
    H0: cross-sectional independence."""
    e = pd.Series(np.asarray(res.resids).ravel(), index=frame.index)
    d = pd.DataFrame({"e": e.values, "i": frame[ENTITY].values, "t": frame[TIME].values})
    M = d.pivot_table(index="t", columns="i", values="e")
    N = M.shape[1]
    total = 0.0
    for a in range(N):
        ea = M.iloc[:, a]
        for b in range(a + 1, N):
            pair = pd.concat([ea, M.iloc[:, b]], axis=1).dropna()
            if len(pair) < 3:
                continue
            r = np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]
            if not np.isnan(r):
                total += np.sqrt(len(pair)) * r
    CD = np.sqrt(2.0 / (N * (N - 1))) * total
    return {"CD": CD, "p_value": 2 * (1 - stats.norm.cdf(abs(CD))), "N_entities": N}


def coef_table(frame, res):
    """Coefficients with clustered inference + standardized betas (comparable across scales)."""
    sdy = frame[OUTCOME].std()
    tab = pd.DataFrame({
        "coef": res.params, "se": res.std_errors,
        "t": res.tstats, "p": res.pvalues,
        "std_beta": [res.params[k] * frame[k].std() / sdy for k in res.params.index],
    })
    return tab.round(4)


def meta(frame, res):
    return {"n_obs": int(res.nobs), "n_entities": frame[ENTITY].nunique(),
            "n_quarters": frame[TIME].nunique(),
            "within_r2": round(float(res.rsquared_within), 4),
            "overall_r2": round(float(res.rsquared), 4)}


# ───────────────────────── orchestration ─────────────────────────
def run(panel=None):
    """Full RQ1 pass: primary + CD + Driscoll-Kraay + weighted sensitivity."""
    panel = load_panel() if panel is None else panel
    frame, audit = build_frame(panel)
    primary = fit(frame, weighted=False, cov="clustered")
    return {
        "frame": frame, "audit": audit,
        "primary": primary,
        "coef": coef_table(frame, primary),
        "meta": meta(frame, primary),
        "cd": pesaran_cd(primary, frame),
        "driscoll_kraay": fit(frame, weighted=False, cov="kernel"),
        "weighted": fit(frame, weighted=True, cov="clustered"),
    }


def print_report(out=None):
    """Thin console report for the notebook driver."""
    out = run() if out is None else out
    m = out["meta"]
    print("RQ1 — association: two-way FE (TA + quarter), clustered SE by TA\n")
    print(f"n = {m['n_obs']}  |  TAs = {m['n_entities']}  |  quarters = {m['n_quarters']}"
          f"  |  within R^2 = {m['within_r2']}  |  overall R^2 = {m['overall_r2']}\n")
    print("Complete-case audit (usable rows by TA size):")
    print(out["audit"].to_string(), "\n")
    print("Primary coefficients (clustered by TA):")
    print(out["coef"].to_string(), "\n")
    cd = out["cd"]
    print(f"Pesaran CD = {cd['CD']:.3f}  (p = {cd['p_value']:.3f})  "
          f"— H0: no residual cross-sectional dependence\n")
    dk, w = out["driscoll_kraay"], out["weighted"]
    print("Robustness — Driscoll-Kraay SEs:")
    print(pd.DataFrame({"coef": dk.params, "DK_se": dk.std_errors, "p": dk.pvalues}).round(4).to_string(), "\n")
    print("Sensitivity — population-weighted:")
    print(pd.DataFrame({"coef": w.params, "se": w.std_errors, "p": w.pvalues}).round(4).to_string())
    return out
