"""Pixel-level check: where on the sky does the dimming come from?

For each sector: download a TESScut full-frame-image cutout, remove slow trends from
every pixel, then build a difference image = (just outside transit) - (in transit),
averaged over all transits. Fit the position of the difference signal and compare with
the target and with each nearby star.

Usage: .venv/bin/python pixel_check.py TIC PERIOD T0_BTJD DURATION_H
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Catalogs
from scipy.ndimage import median_filter

tic, P, t0, dur_h = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
dur = dur_h / 24
SIZE = 13
cat = Catalogs.query_object(f"TIC {tic}", radius=0.035, catalog="TIC").to_pandas()
cat = cat[cat.Tmag < float(os.environ.get("MAXMAG", "17"))].sort_values("dstArcSec")
print(cat[["ID", "dstArcSec", "Tmag"]].head(8).to_string(index=False))

res = lk.search_tesscut(f"TIC {tic}")
sectors = sorted({int(str(m).split()[-1]) for m in res.mission})
want = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else sectors
summary = []
joint = []
ap_rows = []
fig, axes = plt.subplots(2, len(want), figsize=(4.2 * len(want), 8), squeeze=False)
for k, s in enumerate(want):
    i = [j for j in range(len(res)) if int(str(res[j].mission[0]).split()[-1]) == s][0]
    tpf = res[i].download(cutout_size=SIZE, download_dir="out/mast_cache")
    t = tpf.time.value
    cube = np.asarray(tpf.flux.value, float)
    good = np.isfinite(cube).all(axis=(1, 2)) & (tpf.quality == 0)
    t, cube = t[good], cube[good]
    # detrend each pixel: divide out a running median of ~0.75 d (transits masked)
    ph = np.abs(((t - t0) / P + 0.5) % 1 - 0.5) * P
    intr = ph < dur / 2
    near = (ph > dur * 0.75) & (ph < dur * 2.5)
    cad = np.median(np.diff(t))
    w = max(int(0.75 / cad) | 1, 5)
    base = np.empty_like(cube)
    masked = cube.copy()
    masked[intr] = np.nan
    for a in range(SIZE):
        for b in range(SIZE):
            x = masked[:, a, b].copy()
            # fill masked points by interpolation before filtering
            ok = np.isfinite(x)
            x[~ok] = np.interp(t[~ok], t[ok], x[ok])
            base[:, a, b] = median_filter(x, size=w, mode="nearest")
    # per-transit local fit: for each transit and pixel, fit a line to the flux just
    # outside the transit (0.75-3 durations away) and measure the in-transit deficit
    ep = np.round((t - t0) / P).astype(int)
    dt = t - (t0 + ep * P)
    deps, errs = [], []
    for e in np.unique(ep[intr]):
        mi = (ep == e) & (np.abs(dt) < dur / 2)
        mo = (ep == e) & (np.abs(dt) > dur * 0.75) & (np.abs(dt) < dur * 3)
        if mi.sum() < 2 or mo.sum() < 6 or (dt[mo] < 0).sum() < 3 or (dt[mo] > 0).sum() < 3:
            continue
        A = np.vstack([np.ones(mo.sum()), dt[mo]]).T
        y = cube[mo].reshape(mo.sum(), -1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred_in = (np.vstack([np.ones(mi.sum()), dt[mi]]).T @ coef)
        d = (pred_in - cube[mi].reshape(mi.sum(), -1)).mean(axis=0)
        rs = (y - A @ coef).std(axis=0) * np.sqrt(1 / mi.sum() + 1 / mo.sum())
        deps.append(d.reshape(SIZE, SIZE)); errs.append(rs.reshape(SIZE, SIZE))
    deps, errs = np.array(deps), np.array(errs)
    wts = 1 / errs ** 2
    diff = (deps * wts).sum(0) / wts.sum(0)
    err = 1 / np.sqrt(wts.sum(0))
    n_tr = len(deps)
    resid = cube - base
    # sanity: transit depth in a 3x3 aperture on the target from the same cutout
    ap = np.zeros((SIZE, SIZE), bool)
    oot = np.nanmedian(cube[~intr], axis=0)
    # pixel positions of catalogue stars
    wcs = tpf.wcs
    stars = []
    from astropy.time import Time
    yr = Time(np.median(t) + 2457000, format="jd").jyear
    for _, r in cat.iterrows():
        # TIC positions are epoch 2000; move each star by its proper motion to the sector epoch
        pmra = r.pmRA if np.isfinite(r.pmRA) else 0.0
        pmde = r.pmDEC if np.isfinite(r.pmDEC) else 0.0
        dra = pmra * (yr - 2000.0) / 3.6e6 / np.cos(np.radians(r.dec))
        dde = pmde * (yr - 2000.0) / 3.6e6
        x, y = wcs.world_to_pixel(SkyCoord((r.ra + dra) * u.deg, (r.dec + dde) * u.deg))
        stars.append((int(r.ID), float(x), float(y), r.Tmag, r.dstArcSec))
    if k == 0 or True:
        sep_now = np.hypot(stars[0][1] - stars[1][1], stars[0][2] - stars[1][2]) * 21
        print(f"  S{s} epoch {yr:.2f}: target to nearest neighbour TIC {stars[1][0]} now {sep_now:.1f}\" apart")
    tx, ty = stars[0][1], stars[0][2]
    ix, iy = int(round(tx)), int(round(ty))
    ap[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2] = True
    ap_depth = diff[ap].sum() / oot[ap].sum() * 1e6
    ap_err = np.sqrt((err[ap] ** 2).sum()) / oot[ap].sum() * 1e6
    print(f"  S{s} 3x3 aperture on target: depth {ap_depth:.0f} +- {ap_err:.0f} ppm (raw, undiluted by catalog)")
    ap_rows.append(dict(sector=s, depth=float(ap_depth), err=float(ap_err), n_tr=int(n_tr)))
    # flux-weighted centroid of the difference image (positive part, 5x5 around peak)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    if len(deps) == 0:
        print(f"S{s}: no usable transits in this sector"); continue
    pk = np.unravel_index(np.nanargmax(np.where(np.isfinite(diff), diff, -np.inf)), diff.shape)
    box = (np.abs(yy - pk[0]) <= 2) & (np.abs(xx - pk[1]) <= 2)
    wgt = np.clip(diff, 0, None) * box
    cx, cy = (wgt * xx).sum() / wgt.sum(), (wgt * yy).sum() / wgt.sum()
    wo = oot * box
    ox, oy = (wo * xx).sum() / wo.sum(), (wo * yy).sum() / wo.sum()
    # centroid shift relative to the out-of-transit centroid, compared with target offset
    off_target = np.hypot(cx - tx, cy - ty) * 21.0
    # best-matching star: which catalogue star is nearest the difference centroid
    d_stars = [(np.hypot(cx - x, cy - y) * 21.0, sid, mag, sep) for sid, x, y, mag, sep in stars]
    d_stars.sort()
    snr_peak = diff[pk] / err[pk]
    joint.append((s, diff.copy(), err.copy(), stars, oot.copy()))
    summary.append((s, n_tr, off_target, d_stars[0], snr_peak, pk, (round(tx, 2), round(ty, 2))))
    print(f"S{s}: {n_tr} transits, diff-image peak S/N {snr_peak:.1f} at pixel {pk[::-1]}, "
          f"target at ({tx:.2f},{ty:.2f}); diff centroid ({cx:.2f},{cy:.2f}) is {off_target:.1f}\" from target; "
          f"nearest star to centroid: TIC {d_stars[0][1]} (Tmag {d_stars[0][2]:.1f}) {d_stars[0][0]:.1f}\" away", flush=True)
    for row, img, title in ((0, oot, f"S{s} out-of-transit"), (1, diff / err, f"S{s} difference (S/N)")):
        a = axes[row, k]
        a.imshow(img, origin="lower", cmap="viridis")
        for sid, x, y, mag, sep in stars:
            if -0.5 < x < SIZE - 0.5 and -0.5 < y < SIZE - 0.5:
                a.plot(x, y, "x" if sid != tic else "*", color="w" if sid != tic else "r",
                       ms=6 if sid != tic else 12)
        if row == 1:
            a.plot(cx, cy, "o", mfc="none", mec="orange", ms=12, mew=2)
        a.set_title(title, fontsize=9)
fig.tight_layout()
fig.savefig(f"out/pixels_{tic}.png", dpi=80)
print("saved", f"out/pixels_{tic}.png")


# ---- joint source test: which star's position best explains the difference images?
def psf_model(x0, y0, sigma=0.85):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def fit_at(diff, err, x0, y0, region):
    m = psf_model(x0, y0)[region]
    d, e = diff[region], err[region]
    a = (m * d / e ** 2).sum() / (m ** 2 / e ** 2).sum()
    chi2 = (((d - a * m) / e) ** 2).sum()
    chi0 = ((d / e) ** 2).sum()
    return a, chi2, chi0

cands = {}
for s_, diff, err, stars, oot in joint:
    tx, ty = stars[0][1], stars[0][2]
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    region = (np.abs(xx - tx) <= 3.5) & (np.abs(yy - ty) <= 3.5)
    for sid, x, y, mag, sep in stars:
        if sep > 60:
            continue
        a, chi2, chi0 = fit_at(diff, err, x, y, region)
        c = cands.setdefault(sid, dict(mag=mag, sep=sep, chi2=0.0, chi0=0.0, amps=[]))
        c["chi2"] += chi2; c["chi0"] += chi0; c["amps"].append(round(float(a), 1))
    # free-position fit on a fine grid, for information
    best = None
    for x0 in np.arange(tx - 3, tx + 3.01, 0.1):
        for y0 in np.arange(ty - 3, ty + 3.01, 0.1):
            a, chi2, chi0 = fit_at(diff, err, x0, y0, region)
            if a > 0 and (best is None or chi2 < best[0]):
                best = (chi2, x0, y0, a)
    if best is None:
        print(f"S{s_}: no positive dip anywhere near the target"); continue
    print(f"S{s_}: free-position best fit {np.hypot(best[1]-tx, best[2]-ty)*21:.1f}\" from target, "
          f"dchi2 vs no-signal {chi0 - best[0]:.1f}")
ref = cands[tic]["chi2"]
print("\nJoint test over all sectors (lower chi2 = better; delta vs target):")
for sid, c in sorted(cands.items(), key=lambda kv: kv[1]["chi2"]):
    print(f"  TIC {sid:>10}  Tmag {c['mag']:5.1f}  {c['sep']:5.1f}\"  chi2 {c['chi2']:7.1f}  "
          f"delta {c['chi2']-ref:+7.1f}  improvement over no-signal {c['chi0']-c['chi2']:6.1f}  amps {c['amps']}")

import json as _json
best_star = min(cands.items(), key=lambda kv: kv[1]["chi2"])
_json.dump(dict(tic=tic, period=P, sectors=ap_rows,
                joint={str(k): dict(mag=float(v["mag"]), sep=float(v["sep"]), chi2=float(v["chi2"]),
                                    delta=float(v["chi2"] - ref), improvement=float(v["chi0"] - v["chi2"]))
                       for k, v in cands.items()},
                best_star=int(best_star[0])), open(f"out/pixels_{tic}.json", "w"), indent=1)
