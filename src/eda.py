
"""eda.py — reusable exploratory helpers for the source-level EDA."""
import numpy as np
import pandas as pd
import config

def load_processed():
    return {
        "register":   pd.read_excel(config.DATA_PROCESSED / "register_long.xlsx"),
        "bonds":      pd.read_excel(config.DATA_PROCESSED / "bonds_long.xlsx"),
        "consents":   pd.read_excel(config.DATA_PROCESSED / "consents_long.xlsx"),
        "population": pd.read_excel(config.DATA_PROCESSED / "population_long.xlsx"),
    }

# ---------- Section A ----------
def integrity_summary(panels):
    rows = []
    for name, df in panels.items():
        unit = "year" if name == "population" else "quarter"
        n_ta, n_t = df["ta_key"].nunique(), df[unit].nunique()
        rows.append({"source": name, "rows": len(df), "TAs": n_ta, "periods": n_t,
                     "coverage": f"{df[unit].min()}–{df[unit].max()}",
                     "balanced": len(df) == n_ta * n_t,
                     "duplicates": bool(df.duplicated(["ta_key", unit]).any())})
    return pd.DataFrame(rows).set_index("source")

# ---------- Section B ----------
def missingness_summary(panels):
    rows = []
    for name, df in panels.items():
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                rows.append({"source": name, "column": col, "n_missing": n,
                             "pct": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source","column","n_missing","pct"])

def suppression_by_size(register):
    g = register.groupby("ta_name").agg(mean_register=("register_count", "mean"),
                                         n_suppressed=("suppressed", "sum"))
    g["ever_suppressed"] = g["n_suppressed"] > 0
    summary = (g.groupby("ever_suppressed")["mean_register"]
                 .agg(["count", "mean", "median", "min", "max"]).round(1))
    return g.sort_values("n_suppressed", ascending=False), summary

def thin_quarters(bonds, min_months=3):
    thin = bonds[bonds["n_months"] < min_months]
    return (thin.groupby("ta_name").size().sort_values(ascending=False)
                .rename("n_thin_quarters").to_frame())

# ---------- Section C ----------
def distribution_summary(panels):
    checks = [("register","register_count"),("bonds","geom_mean_rent"),
              ("bonds","lodged_bonds"),("bonds","active_bonds"),
              ("consents","dwelling_units"),("consents","multi_unit_share"),
              ("population","population")]
    rows = []
    for src, col in checks:
        s = pd.to_numeric(panels[src][col], errors="coerce").dropna()
        med = s.median()
        rows.append({"source": src, "variable": col, "skew": round(s.skew(), 1),
                     "median": round(med, 1), "max": round(s.max(), 1),
                     "max_over_median": round(s.max()/med, 1) if med else np.nan})
    return pd.DataFrame(rows)

def rent_spread(bonds):
    return bonds.groupby("ta_name")["geom_mean_rent"].mean().dropna().sort_values()

# ---------- Section D ----------
def national_register_trend(register):
    r = register.copy(); r["q"] = pd.PeriodIndex(r["quarter"], freq="Q")
    return r.groupby("q")["register_count"].sum()          # suppressed excluded

def growth_concentration(register, start="2016Q1", end="2025Q3", top=5):
    r = register.copy(); r["q"] = pd.PeriodIndex(r["quarter"], freq="Q")
    w0, w1 = pd.Period(start), pd.Period(end)
    g = r[r["q"].isin([w0, w1])].pivot_table(index="ta_name", columns="q", values="register_count")
    g.columns = [str(start), str(end)]
    g["growth"] = g[str(end)] - g[str(start)]
    g = g.sort_values("growth", ascending=False)
    out = g.head(top).copy()
    out["pct_of_national_increase"] = (100 * out["growth"] / g["growth"].sum()).round(0)
    return out

def seasonality_summary(panels):
    specs = [("bonds","lodged_bonds"),("consents","dwelling_units"),("register","register_count")]
    rows = []
    for src, col in specs:
        d = panels[src].copy(); d["qoy"] = pd.PeriodIndex(d["quarter"], freq="Q").quarter
        m = d.groupby("qoy")[col].mean(); base = m.mean()
        rows.append({"source": src, "variable": col,
                     **{f"Q{k}_pct": round(100*v/base) for k, v in m.items()}})
    return pd.DataFrame(rows)

def covid_deltas(panels, baseline="2019Q4", shock="2020Q2"):
    b, s = pd.Period(baseline), pd.Period(shock)
    def natl(df, col, geo=False):
        d = df.copy(); d["q"] = pd.PeriodIndex(d["quarter"], freq="Q")
        if geo:
            return d.groupby("q")[col].apply(
                lambda x: np.exp(np.log(pd.to_numeric(x, errors="coerce").dropna()).mean()))
        return d.groupby("q")[col].sum()
    rows = []
    for nm, src, col, geo in [("bond lodgements","bonds","lodged_bonds",False),
                              ("rent (geo mean)","bonds","geom_mean_rent",True),
                              ("consents","consents","dwelling_units",False)]:
        ser = natl(panels[src], col, geo)
        rows.append({"series": nm, baseline: round(ser.loc[b]), shock: round(ser.loc[s]),
                     "pct_change": round(100*(ser.loc[s]-ser.loc[b])/ser.loc[b])})
    return pd.DataFrame(rows)

    # ---------- Section E: within-source relationships ----------
def bond_corr(bonds, within_ta=False):
    """Correlation matrix of bond activity/price variables.
    within_ta=True demeans each variable per TA first, removing size dominance."""
    b = bonds.copy()
    b["net_bond_flow"] = b["lodged_bonds"] - b["closed_bonds"]
    cols = ["lodged_bonds", "closed_bonds", "active_bonds",
            "net_bond_flow", "geom_mean_rent", "log_std_dev_rent"]
    if within_ta:
        for c in cols:
            b[c] = b.groupby("ta_key")[c].transform(lambda x: x - x.mean())
    return b[cols].corr()

def active_vs_netflow(bonds):
    """The stock=cumulative-flow identity check. Returns (df, correlation)."""
    b = bonds.sort_values(["ta_key", "quarter"]).copy()
    b["net_bond_flow"] = b["lodged_bonds"] - b["closed_bonds"]
    b["d_active"] = b.groupby("ta_key")["active_bonds"].diff()
    sub = b.dropna(subset=["d_active"])
    return sub, sub["d_active"].corr(sub["net_bond_flow"])