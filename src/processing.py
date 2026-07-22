"""
processing.py — source cleaning functions. One home for all file processing.
  load_register()   -> MSD housing register (quarterly panel of counts)
  load_population()  -> Stats NZ subnational population (annual, 30 June)
Bonds / consents added here as we go.
"""
import re
import pandas as pd
import openpyxl

import config
from ta_registry import normalise, CANONICAL_TAS
import validate as V

# ───────────────── shared helpers ─────────────────
_MONTH_Q = {"Mar": 1, "Jun": 2, "Sep": 3, "Dec": 4}

def _parse_quarter(label):
    mon, yy = label.split("-")
    return pd.Period(f"{2000 + int(yy)}Q{_MONTH_Q[mon.strip()]}", freq="Q")

def _display(key):
    return CANONICAL_TAS.get(key, key)

# ───────────────── register ─────────────────
def _is_junk(name):
    r = str(name).replace("\xa0", " ").strip()
    return r == "" or r.startswith(config.REGISTER_JUNK_PREFIXES)

def _clean_count(raw):
    if raw is None:
        return (pd.NA, False)
    s = str(raw).replace("\xa0", " ").strip()
    if s == "":
        return (pd.NA, False)
    if s.upper() == "S":
        return (pd.NA, True)
    s = s.replace(",", "")
    try:
        return (int(float(s)), False)
    except ValueError:
        return (pd.NA, False)

def _read_ta_summary(path, vintage):
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["TA summary"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = next(i for i, r in enumerate(rows) if r and r[1] and str(r[1]).strip() == "TA")
    header = rows[hdr]
    qcols = [(j, _parse_quarter(str(header[j]).strip()))
             for j in range(2, len(header))
             if header[j] and re.match(r"[A-Za-z]{3}-\d{2}$", str(header[j]).strip())]
    recs = []
    for r in rows[hdr + 1:]:
        name = r[1]
        if not name or _is_junk(name):
            continue
        key = normalise(name)
        if key is None:
            continue
        for j, q in qcols:
            val, supp = _clean_count(r[j])
            recs.append((key, _display(key), q, val, supp, vintage))
    df = pd.DataFrame(recs, columns=["ta_key", "ta_name", "quarter",
                                     "register_count", "suppressed", "vintage"])
    df["register_count"] = df["register_count"].astype("Int64")
    V.assert_ta_set(df["ta_key"].unique(), f"register-{vintage}")
    return df

def _assert_complete_panel(df):
    n_q = df["quarter"].nunique()
    if df.duplicated(["ta_key", "quarter"]).any():
        raise V.ValidationError("register: duplicate TA-quarter rows")
    if not (df.groupby("quarter")["ta_key"].nunique() == config.N_TAS).all():
        raise V.ValidationError("register: some quarters lack all 66 TAs")
    if not (df.groupby("ta_key")["quarter"].nunique() == n_q).all():
        raise V.ValidationError("register: some TAs are missing quarters")

def load_register():
    d21 = _read_ta_summary(config.REGISTER_2021, 2021)
    d26 = _read_ta_summary(config.REGISTER_2026, 2026)
    overlap = sorted(set(d21["quarter"]) & set(d26["quarter"]))
    a = (d21[d21["quarter"].isin(overlap)][["ta_key", "quarter", "register_count"]]
         .rename(columns={"register_count": "v2021"}))
    b = (d26[d26["quarter"].isin(overlap)][["ta_key", "quarter", "register_count"]]
         .rename(columns={"register_count": "v2026"}))
    seam = a.merge(b, on=["ta_key", "quarter"])
    seam["diff"] = seam["v2026"] - seam["v2021"]
    df = (pd.concat([d21[~d21["quarter"].isin(overlap)], d26], ignore_index=True)
            .sort_values(["ta_key", "quarter"]).reset_index(drop=True))
    _assert_complete_panel(df)
    return df, seam

def save_register_xlsx(df, path):
    out = df.copy()
    out["quarter"] = out["quarter"].astype(str)
    out.to_excel(path, index=False)
    return path

# ───────────────── population ─────────────────
def _assert_complete_annual(df):
    n_y = df["year"].nunique()
    if df.duplicated(["ta_key", "year"]).any():
        raise V.ValidationError("population: duplicate TA-year rows")
    if not (df.groupby("year")["ta_key"].nunique() == config.N_TAS).all():
        raise V.ValidationError("population: some years lack all 66 TAs")
    if not (df.groupby("ta_key")["year"].nunique() == n_y).all():
        raise V.ValidationError("population: some TAs are missing years")
    if (df["population"] <= 0).any():
        raise V.ValidationError("population: non-positive population found")

def load_population():
    """Stats NZ subnational population (POPES_SUB_006), 30 June estimates.
    Faithful annual clean-up: one row per TA-year. Quarterly expansion and
    demand_rate are deferred to panel assembly."""
    p = pd.read_csv(config.POPULATION, encoding="utf-8")
    p = p.rename(columns={"Year at 30 June": "year", "OBS_VALUE": "population"})
    p = p[["year", "Area", "population"]].copy()
    p = p[~p["Area"].isin(config.POPULATION_DROP)].copy()
    p["ta_key"] = p["Area"].map(normalise)
    p["ta_name"] = p["ta_key"].map(_display)
    p = (p[["ta_key", "ta_name", "year", "population"]]
         .sort_values(["ta_key", "year"]).reset_index(drop=True))
    p["population"] = p["population"].astype("int64")
    p["year"] = p["year"].astype(int)
    V.assert_ta_set(p["ta_key"].unique(), "population")
    _assert_complete_annual(p)
    return p

def save_population_xlsx(df, path):
    df.to_excel(path, index=False)
    return path
