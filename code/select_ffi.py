"""Nearby red dwarfs with only full-frame-image light curves (TESS-SPOC or QLP), no 2-min data,
and no known planet/candidate. Sorted by number of FFI sectors (more data first)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from astroquery.mast import Catalogs, Observations

tic = Catalogs.query_criteria(catalog="Tic", Teff=[2800, 3900], d=[0, 40], Tmag=[9, 13.5],
                              objType="STAR", lumclass="DWARF").to_pandas()
print("TIC M dwarfs <40 pc, Tmag 9-13.5:", len(tic))
known = set()
for f in ("toi.csv", "ctoi.csv"):
    known |= set(pd.read_csv(f"out/catalogs/{f}", comment="#")["TIC ID"].astype(int))
ps = pd.read_csv("out/catalogs/ps.csv")
known |= set(ps.tic_id.dropna().str.replace("TIC ", "").astype(int))
known.add(264151965)
tic["ID"] = tic.ID.astype(int)
tic = tic[~tic.ID.isin(known)]
print("unknown:", len(tic))
parts = []
ids = [str(i) for i in tic.ID]
for k in range(0, len(ids), 200):
    for prov in ("SPOC", "TESS-SPOC", "QLP"):
        try:
            o = Observations.query_criteria(target_name=ids[k:k+200], obs_collection=["TESS", "HLSP"],
                                            dataproduct_type="timeseries", provenance_name=prov).to_pandas()
            parts.append(o[["target_name", "provenance_name", "sequence_number", "t_exptime"]])
        except Exception as e:
            print("query fail", k, prov, e)
    print(k, flush=True)
obs = pd.concat(parts)
obs["tid"] = obs.target_name.astype(int)
spoc2 = obs[(obs.provenance_name == "SPOC") & obs.t_exptime.between(100, 140)].groupby("tid").sequence_number.nunique()
ffi = obs[obs.provenance_name.isin(["TESS-SPOC", "QLP"])].groupby("tid").sequence_number.nunique()
recent = obs[obs.provenance_name.isin(["TESS-SPOC", "QLP"]) & (obs.sequence_number >= 90)].groupby("tid").sequence_number.nunique()
tic["n_2min"] = tic.ID.map(spoc2).fillna(0).astype(int)
tic["n_ffi"] = tic.ID.map(ffi).fillna(0).astype(int)
tic["n_recent"] = tic.ID.map(recent).fillna(0).astype(int)
sel = tic[(tic.n_2min == 0) & (tic.n_ffi >= 3)].copy()
sel["control"] = ""
sel = sel.sort_values(["n_recent", "n_ffi"], ascending=False)
cols = ["ID", "ra", "dec", "Tmag", "Teff", "rad", "mass", "d", "n_ffi", "n_recent", "control"]
sel[cols].to_csv("out/targets_ffi.csv", index=False)
print("FFI-only targets with >=3 sectors:", len(sel))
print(sel[cols].head(15).to_string(index=False))
print(sel.n_ffi.describe())
