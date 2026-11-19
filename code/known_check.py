"""Flag pixel-passing signals that sit within 60 arcsec of a known TOI/CTOI or confirmed host.
Usage: .venv/bin/python known_check.py RUN_DIR TARGETS_CSV  -> appends to out/FINDINGS.md"""
import json, glob, sys, datetime
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

run, targets = sys.argv[1], sys.argv[2]
toi = pd.read_csv("out/catalogs/toi.csv", comment="#")
ctoi = pd.read_csv("out/catalogs/ctoi.csv", comment="#")
def coords(df):
    rc = [c for c in df.columns if c.lower().startswith("ra")][0]
    dc = [c for c in df.columns if c.lower().startswith("dec")][0]
    if ":" in str(df[rc].iloc[0]):
        return SkyCoord(df[rc].astype(str).values, df[dc].astype(str).values, unit=(u.hourangle, u.deg))
    return SkyCoord(df[rc].values * u.deg, df[dc].values * u.deg)
kt, kc = coords(toi), coords(ctoi)
tg = pd.read_csv(f"out/{targets}").set_index("ID")
lines = []
for f in glob.glob(f"out/{run}/*.json"):
    r = json.load(open(f))
    for c in r.get("signals", []):
        if not c.get("pixel_pass"):
            continue
        s = SkyCoord(tg.loc[r["tic"], "ra"] * u.deg, tg.loc[r["tic"], "dec"] * u.deg)
        dt, dc = s.separation(kt).arcsec, s.separation(kc).arcsec
        known = ""
        if dt.min() < 60: known += f"TOI {toi.iloc[dt.argmin()]['TOI']} (P={toi.iloc[dt.argmin()]['Period (days)']}) {dt.min():.0f}\" away. "
        if dc.min() < 60: known += f"CTOI {ctoi.iloc[dc.argmin()].iloc[1]} {dc.min():.0f}\" away."
        status = "KNOWN: " + known if known else "NEW - needs manual review"
        lines.append(f"- TIC {r['tic']} P={c['period']:.5f} d, SDE {c['sde']:.1f}, depth {c['depth_ppm']:.0f} ppm, "
                     f"Rp~{c.get('radius_earth') or 0:.2f} Re, d={r['d_pc']:.1f} pc, pixel depth {c['pixel']['depth']:.0f}+-{c['pixel']['err']:.0f} ppm -> {status}")
with open("out/FINDINGS.md", "a") as fh:
    fh.write(f"\n## {run} ({targets}) - {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC\n")
    fh.write("\n".join(lines) if lines else "- no signals passed the pixel check")
    fh.write("\n")
print("\n".join(lines) if lines else "no pixel passes")
