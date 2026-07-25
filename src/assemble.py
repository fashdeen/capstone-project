"""
assemble.py — build the analysis panel from the four processed sources.
Base: 63-TA register spine + interpolated population + demand_rate.
Predictors: rent_growth, net_bond_flow, lags 1–4, change-form outcome.
"""
import numpy as np
import pandas as pd
import config

PANEL_START = pd.Period(config.PANEL_START, freq="Q")
PANEL_END   = pd.Period(config.PANEL_END,   freq="Q")
LAGS  = [1, 2, 3, 4]
PVARS = ["geom_mean_rent", "rent_growth", "net_bond_flow", "dwelling_units", "multi_unit_share"]

def excluded_keys():
    return pd.read_csv(config.DATA_PROCESSED / "excluded_tas.csv")["excluded_ta_key"].tolist()

def _quarterly_population(pop, excl, qindex):
    pop = pop[~pop["ta_key"].isin(excl)].copy()
    pop["q"] = pop["year"].apply(lambda y: pd.Period(f"{y}Q2", freq="Q"))
    frames = []
    for ta, g in pop.groupby("ta_key"):
        s = (g.set_index("q")["population"].reindex(qindex)
               .interpolate(method="linear", limit_direction="both"))
        frames.append(pd.DataFrame({"ta_key": ta, "q": qindex, "population": s.values}))
    return pd.concat(frames, ignore_index=True)

def _build_base(excl):
    reg = pd.read_excel(config.DATA_PROCESSED / "register_long.xlsx")
    pop = pd.read_excel(config.DATA_PROCESSED / "population_long.xlsx")
    reg["q"] = pd.PeriodIndex(reg["quarter"], freq="Q")
    reg = reg[(reg["q"] >= PANEL_START) & (reg["q"] <= PANEL_END) & (~reg["ta_key"].isin(excl))].copy()
    qindex = pd.period_range(PANEL_START - 2, PANEL_END, freq="Q")
    popq = _quarterly_population(pop, excl, qindex)
    base = reg.merge(popq, on=["ta_key", "q"], how="left")
    base["demand_rate"] = base["register_count"] / base["population"] * 1000
    return base[["ta_key", "ta_name", "q", "register_count", "suppressed", "population", "demand_rate"]]

def _build_predictors(excl):
    """Predictor grid on the full 2015+ range (so lags fill for 2016Q1)."""
    bon = pd.read_excel(config.DATA_PROCESSED / "bonds_long.xlsx")
    con = pd.read_excel(config.DATA_PROCESSED / "consents_long.xlsx")
    for d in (bon, con):
        d["q"] = pd.PeriodIndex(d["quarter"], freq="Q")
    bon = bon[~bon["ta_key"].isin(excl)].sort_values(["ta_key", "q"]).copy()
    con = con[~con["ta_key"].isin(excl)].copy()
    bon["net_bond_flow"] = bon["lodged_bonds"] - bon["closed_bonds"]        # carried; active_bonds dropped (Section E identity)
    bon["rent_growth"] = bon.groupby("ta_key")["geom_mean_rent"].transform(lambda s: np.log(s).diff())
    pred = bon[["ta_key", "q", "geom_mean_rent", "rent_growth", "net_bond_flow"]].merge(
           con[["ta_key", "q", "dwelling_units", "multi_unit_share"]], on=["ta_key", "q"], how="outer")
    pred = pred.sort_values(["ta_key", "q"])
    for v in PVARS:
        for L in LAGS:
            pred[f"{v}_lag{L}"] = pred.groupby("ta_key")[v].shift(L)
    return pred

def build_panel():
    """Full analysis panel: base outcome + lagged predictors + change-form outcome."""
    excl = excluded_keys()
    base = _build_base(excl)
    pred = _build_predictors(excl)
    lagcols = [f"{v}_lag{L}" for v in PVARS for L in LAGS]
    panel = base.merge(pred[["ta_key", "q"] + lagcols], on=["ta_key", "q"], how="left")
    panel = panel.sort_values(["ta_key", "q"]).reset_index(drop=True)
    panel["demand_rate_change"] = panel.groupby("ta_key")["demand_rate"].diff()
    _validate_panel(panel, lagcols)
    return panel

def _validate_panel(df, lagcols):
    n_ta, n_q = df["ta_key"].nunique(), df["q"].nunique()
    assert n_ta == 63 and len(df) == n_ta * n_q, f"shape wrong: {df.shape}"
    assert df["population"].notna().all(), "population gaps"
    assert not df.duplicated(["ta_key", "q"]).any(), "duplicate rows"
    # lags on the first quarter must be filled from the 2015 runway
    q1 = df[df["q"] == PANEL_START]
    assert q1["geom_mean_rent_lag4"].notna().all(), "lag4 not filled at panel start — runway problem"

def save_panel(panel, path=None):
    path = path or (config.DATA_PROCESSED / "panel.parquet")
    out = panel.copy(); out["q"] = out["q"].astype(str)   # Period -> string for parquet
    out.to_parquet(path, index=False)
    return path
