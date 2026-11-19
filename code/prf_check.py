"""Recheck 2: locate the transit source with the real TESS PRF instead of a Gaussian.

Per sector: TESScut 13x13 cutout; per-transit, per-pixel linear baselines give a difference image
(out - in) and its error. For each candidate star (Tmag < 21, within 70", proper motion applied),
fit diff = a * PRF(star position) + c and record chi2. Also a free-position PRF fit.
Usage: .venv/bin/python prf_check.py TIC [sectors]"""
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
import PRF

tic = int(sys.argv[1])
fit = json.load(open(f"out/fit_{tic}.json"))
P, T0, D = fit["P_d"][0], fit["T0_BTJD"][0], fit["T14_h"][0] / 24
SIZE = 13
cat = Catalogs.query_object(f"TIC {tic}", radius=70 / 3600, catalog="TIC").to_pandas()
cat = cat[cat.Tmag < 21].sort_values("dstArcSec").reset_index(drop=True)
tc = lk.search_tesscut(f"TIC {tic}")
want = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None


def diff_image(t, cube):
    ep = np.round((t - T0) / P).astype(int)
    dt = t - (T0 + ep * P)
    deps, errs = [], []
    for e in np.unique(ep[np.abs(dt) < D / 2]):
        mi = (ep == e) & (np.abs(dt) < D / 2)
        mo = (ep == e) & (np.abs(dt) > D * 0.75) & (np.abs(dt) < D * 3)
        if mi.sum() < 2 or mo.sum() < 6 or (dt[mo] < 0).sum() < 3 or (dt[mo] > 0).sum() < 3:
            continue
        A = np.vstack([np.ones(mo.sum()), dt[mo]]).T
        y = cube[mo].reshape(mo.sum(), -1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pin = np.vstack([np.ones(mi.sum()), dt[mi]]).T @ coef
        deps.append((pin - cube[mi].reshape(mi.sum(), -1)).mean(0))
        errs.append((y - A @ coef).std(0) * np.sqrt(1 / mi.sum() + 1 / mo.sum()))
    deps, errs = np.array(deps), np.array(errs)
    w = 1 / errs ** 2
    return ((deps * w).sum(0) / w.sum(0)).reshape(SIZE, SIZE), (w.sum(0) ** -0.5).reshape(SIZE, SIZE), len(deps)


def fit_model(diff, err, model, region):
    d, e, m = diff[region], err[region], model[region]
    A = np.vstack([m / e, 1 / e]).T
    coef, *_ = np.linalg.lstsq(A, d / e, rcond=None)
    chi2 = ((d / e - A @ coef) ** 2).sum()
    chi0 = (((d - np.average(d, weights=1 / e ** 2)) / e) ** 2).sum()
    return coef[0], chi2, chi0


tot = {}
free = []
for j in range(len(tc)):
    s = int(str(tc[j].mission[0]).split()[-1])
    if s == 2 or (want and s not in want):
        continue
    tpf = tc[j].download(cutout_size=SIZE, download_dir="out/mast_cache")
    t = tpf.time.value
    cube = np.asarray(tpf.flux.value, float)
    good = np.isfinite(cube).all(axis=(1, 2)) & (tpf.quality == 0)
    t, cube = t[good], cube[good]
    diff, err, ntr = diff_image(t, cube)
    yr = Time(np.median(t) + 2457000, format="jd").jyear
    prf = PRF.TESS_PRF(tpf.camera, tpf.ccd, s, tpf.column + SIZE // 2, tpf.row + SIZE // 2)
    pos = []
    for _, r in cat.iterrows():
        pmra = r.pmRA if np.isfinite(r.pmRA) else 0.0
        pmde = r.pmDEC if np.isfinite(r.pmDEC) else 0.0
        ra = r.ra + pmra * (yr - 2000) / 3.6e6 / np.cos(np.radians(r.dec))
        de = r.dec + pmde * (yr - 2000) / 3.6e6
        x, y = tpf.wcs.world_to_pixel(SkyCoord(ra * u.deg, de * u.deg))
        pos.append((int(r.ID), float(x), float(y), float(r.Tmag), float(r.dstArcSec)))
    tx, ty = pos[0][1], pos[0][2]
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    region = (np.abs(xx - tx) <= 4) & (np.abs(yy - ty) <= 4)
    for sid, x, y, mag, sep in pos:
        if not (-1 < x < SIZE and -1 < y < SIZE):
            continue
        a, chi2, chi0 = fit_model(diff, err, prf.locate(x, y, (SIZE, SIZE)), region)
        c = tot.setdefault(sid, dict(mag=mag, sep=sep, chi2=0.0, chi0=0.0, amp=[]))
        c["chi2"] += chi2; c["chi0"] += chi0; c["amp"].append(round(float(a), 1))
    # free position
    best = None
    for x0 in np.arange(tx - 2.5, tx + 2.51, 0.1):
        for y0 in np.arange(ty - 2.5, ty + 2.51, 0.1):
            a, chi2, chi0 = fit_model(diff, err, prf.locate(x0, y0, (SIZE, SIZE)), region)
            if a > 0 and (best is None or chi2 < best[0]):
                best = (chi2, x0, y0, chi0)
    off = np.hypot(best[1] - tx, best[2] - ty) * 21
    free.append((s, off, best[3] - best[0]))
    print(f"S{s}: {ntr} transits; free-position PRF fit {off:.1f}\" from target (dchi2 vs flat {best[3]-best[0]:.1f})", flush=True)

ref = tot[tic]["chi2"]
print("\nJoint PRF source test over all sectors (delta chi2 vs target; positive = worse than target):")
for sid, c in sorted(tot.items(), key=lambda kv: kv[1]["chi2"])[:10]:
    print(f"  TIC {sid:>10}  Tmag {c['mag']:5.1f}  {c['sep']:5.1f}\"  delta {c['chi2']-ref:+8.1f}  "
          f"signal fit improvement {c['chi0']-c['chi2']:7.1f}  sectors {len(c['amp'])}")
w = np.array([x[2] for x in free]); offs = np.array([x[1] for x in free])
print(f"\nSignal-weighted mean free-position offset: {np.average(offs, weights=np.clip(w, 0, None)+1e-9):.1f}\"")
json.dump(dict(stars={str(k): v for k, v in tot.items()}, free=free), open(f"out/prf_{tic}.json", "w"), indent=1, default=float)
