"""TRICERATOPS false-positive probability for TIC 144439305.01.

Apertures: 3x3 pixels on the target (proper-motion-propagated), in CCD coordinates, for each
sector with 200 s FFIs (69, 96, 103, 104, 105, 106). Light curve: detrended 10-min binned
TESS data from all sectors, folded on the fitted ephemeris.
Usage: FFI=1 .venv/bin/python tric_run.py [N_samples] [n_runs]"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.mast import Catalogs
import triceratops.triceratops as tr

import search_v2 as S

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MOLUSC = sys.argv[3] if len(sys.argv) > 3 else None
TAG = "_molusc" if MOLUSC else ""
tic = 144439305
fit = json.load(open(f"out/fit_{tic}.json"))
P, T0, T14 = fit["P_d"][0], fit["T0_BTJD"][0], fit["T14_h"][0] / 24
depth = fit["depth_ppm"][0] / 1e6
sectors = [69, 96, 103, 104, 105, 106]
star = Catalogs.query_object(f"TIC {tic}", radius=0.001, catalog="TIC").to_pandas().iloc[0]

# apertures in CCD pixel coordinates
res = lk.search_tesscut(f"TIC {tic}")
apertures = []
for s in sectors:
    i = [j for j in range(len(res)) if int(str(res[j].mission[0]).split()[-1]) == s][0]
    tpf = res[i].download(cutout_size=13, download_dir="out/mast_cache")
    yr = Time(np.median(tpf.time.value) + 2457000, format="jd").jyear
    dra = star.pmRA * (yr - 2000) / 3.6e6 / np.cos(np.radians(star.dec))
    dde = star.pmDEC * (yr - 2000) / 3.6e6
    x, y = tpf.wcs.world_to_pixel(SkyCoord((star.ra + dra) * u.deg, (star.dec + dde) * u.deg))
    ix, iy = int(round(float(x))), int(round(float(y)))
    ap = [[tpf.column + ix + dx, tpf.row + iy + dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    apertures.append(np.array(ap))
    print(f"S{s}: aperture centre CCD col {tpf.column + ix}, row {tpf.row + iy}")

target = tr.target(ID=tic, sectors=sectors, search_radius=10)
print(target.stars[["ID", "Tmag", "ra", "dec", "mass", "rad", "Teff", "plx", "sep (arcsec)"]].head(10).to_string(index=False))
target.calc_depths(tdepth=depth, all_ap_pixels=apertures)
print(target.stars[["ID", "Tmag", "sep (arcsec)", "fluxratio", "tdepth"]].head(10).to_string(index=False))

# folded light curve
S.MAX_SECTORS = 99
raw = S.download_ffi(tic)
prot, ppow, _ = S.spin_period(raw)
t, f, sec, _ = S.detrend(raw, prot, ppow)
ph = ((t - T0) / P + 0.5) % 1 - 0.5
m = np.abs(ph * P) < 3 * T14
time = ph[m] * P
flux = f[m]
sig = 1.4826 * np.median(np.abs(flux - np.median(flux)))
o = np.argsort(time)
time, flux = time[o], flux[o]
# bin to 5-min in phase to keep TRICERATOPS fast
bins = np.arange(time.min(), time.max() + 1e-9, 5 / 1440)
idx = np.digitize(time, bins)
tb = np.array([time[idx == k].mean() for k in np.unique(idx)])
fb = np.array([flux[idx == k].mean() for k in np.unique(idx)])
eb = np.array([sig / np.sqrt((idx == k).sum()) for k in np.unique(idx)])
print(f"folded LC: {len(tb)} phase bins, per-bin error ~{np.median(eb)*1e6:.0f} ppm")

fpps, nfpps, scen = [], [], None
for k in range(RUNS):
    target.calc_probs(time=tb, flux_0=fb, flux_err_0=float(np.median(eb)), P_orb=P, N=N, parallel=False, verbose=0,
                      molusc_file=MOLUSC)
    fpps.append(target.FPP); nfpps.append(target.NFPP)
    scen = target.probs
    print(f"run {k+1}: FPP = {target.FPP:.4f}, NFPP = {target.NFPP:.2e}", flush=True)
print(f"\nFPP = {np.mean(fpps):.4f} +- {np.std(fpps):.4f}; NFPP = {np.mean(nfpps):.2e}")
print(scen.sort_values("prob", ascending=False).head(8)[["ID", "scenario", "prob"]].to_string(index=False))
json.dump(dict(FPP=float(np.mean(fpps)), FPP_std=float(np.std(fpps)), NFPP=float(np.mean(nfpps)), N=N, runs=RUNS,
               scenarios=scen.sort_values("prob", ascending=False).head(10)[["ID", "scenario", "prob"]].astype(str).values.tolist()),
          open(f"out/triceratops_{tic}{TAG}.json", "w"), indent=1)
