"""Run the frozen test in out/PREDICTION_TIC144439305_S105_S106.json on TESScut sectors 105-106.

Same aperture measurement as pixel_check.py (per-transit, per-pixel linear baselines; 3x3
aperture on the target at its proper-motion-propagated position), evaluated at the predicted
ephemeris and at 40 wrong phases."""
import hashlib
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

PRED = "out/PREDICTION_TIC144439305_S105_S106.json"
txt = open(PRED).read()
print("prediction file SHA-256:", hashlib.sha256(txt.encode()).hexdigest())
pred = json.loads(txt)
T0 = pred["ephemeris"]["T0_BTJD"][0]
P = pred["ephemeris"]["P_d"][0]
D = pred["ephemeris"]["T14_h"][0] / 24
tic = 144439305
SIZE = 13
star = Catalogs.query_object(f"TIC {tic}", radius=0.001, catalog="TIC").to_pandas().iloc[0]
res = lk.search_tesscut(f"TIC {tic}")


def aperture_depth(t, cube, ap, t0):
    ep = np.round((t - t0) / P).astype(int)
    dt = t - (t0 + ep * P)
    deps, errs = [], []
    for e in np.unique(ep[np.abs(dt) < D / 2]):
        mi = (ep == e) & (np.abs(dt) < D / 2)
        mo = (ep == e) & (np.abs(dt) > D * 0.75) & (np.abs(dt) < D * 3)
        if mi.sum() < 2 or mo.sum() < 6 or (dt[mo] < 0).sum() < 3 or (dt[mo] > 0).sum() < 3:
            continue
        y = cube[:, ap].sum(axis=1)
        A = np.vstack([np.ones(mo.sum()), dt[mo]]).T
        coef, *_ = np.linalg.lstsq(A, y[mo], rcond=None)
        pred_in = np.vstack([np.ones(mi.sum()), dt[mi]]).T @ coef
        base = np.mean(pred_in)
        deps.append((pred_in - y[mi]).mean() / base)
        errs.append((y[mo] - A @ coef).std() / base * np.sqrt(1 / mi.sum() + 1 / mo.sum()))
    deps, errs = np.array(deps), np.array(errs)
    if len(deps) == 0:
        return np.nan, np.nan, 0
    w = 1 / errs ** 2
    return (deps * w).sum() / w.sum() * 1e6, w.sum() ** -0.5 * 1e6, len(deps)


results = {}
for s in (105, 106):
    i = [j for j in range(len(res)) if int(str(res[j].mission[0]).split()[-1]) == s][0]
    tpf = res[i].download(cutout_size=SIZE, download_dir="out/mast_cache")
    t = tpf.time.value
    cube = np.asarray(tpf.flux.value, float)
    good = np.isfinite(cube).all(axis=(1, 2)) & (tpf.quality == 0)
    t, cube = t[good], cube[good]
    yr = Time(np.median(t) + 2457000, format="jd").jyear
    dra = star.pmRA * (yr - 2000) / 3.6e6 / np.cos(np.radians(star.dec))
    dde = star.pmDEC * (yr - 2000) / 3.6e6
    x, y = tpf.wcs.world_to_pixel(SkyCoord((star.ra + dra) * u.deg, (star.dec + dde) * u.deg))
    ix, iy = int(round(float(x))), int(round(float(y)))
    ap = np.zeros((SIZE, SIZE), bool)
    ap[iy - 1:iy + 2, ix - 1:ix + 2] = True
    d, e, n = aperture_depth(t, cube, ap, T0)
    nulls = []
    for ph in np.linspace(0.1, 0.9, 43):
        if abs(ph - 0.5) < 0.03:
            continue
        nulls.append(aperture_depth(t, cube, ap, T0 + ph * P)[0])
    nulls = np.array(nulls)
    results[s] = dict(depth=d, err=e, n_tr=n, nulls=nulls, t=(t.min(), t.max()))
    print(f"S{s}: BTJD {t.min():.2f}-{t.max():.2f}; {n} transits at predicted times; "
          f"depth {d:.0f} +- {e:.0f} ppm ({d/e:.1f} sigma); wrong-phase nulls: mean {np.nanmean(nulls):.0f}, "
          f"std {np.nanstd(nulls):.0f}, 95th pct {np.nanpercentile(nulls, 95):.0f} ppm")

# combined
ds = np.array([results[s]["depth"] for s in results]); es = np.array([results[s]["err"] for s in results])
w = 1 / es ** 2
Dc, sDc = (ds * w).sum() / w.sum(), w.sum() ** -0.5
nullc = np.array([(results[105]["nulls"][k] / results[105]["err"] ** 2 + results[106]["nulls"][k] / results[106]["err"] ** 2)
                  / (1 / results[105]["err"] ** 2 + 1 / results[106]["err"] ** 2) for k in range(len(results[105]["nulls"]))])
p95 = np.nanpercentile(nullc, 95)
print(f"\nCOMBINED S105+S106: D = {Dc:.0f} +- {sDc:.0f} ppm ({Dc/sDc:.1f} sigma); combined-null 95th pct {p95:.0f} ppm; "
      f"fraction of nulls >= D: {(nullc >= Dc).mean():.3f}")
if Dc / sDc >= 3 and Dc > p95:
    verdict = "RECOVERED"
elif Dc / sDc < 0.5 and (300 - Dc) / sDc >= 3:
    verdict = "EXCLUDED"
else:
    verdict = "INCONCLUSIVE"
print("VERDICT (frozen rule):", verdict)
json.dump(dict(verdict=verdict, combined_depth=Dc, combined_err=sDc, null_p95=p95,
               per_sector={str(s): dict(depth=results[s]["depth"], err=results[s]["err"], n_tr=results[s]["n_tr"]) for s in results}),
          open("out/PREREG_RESULT_TIC144439305.json", "w"), indent=1, default=float)
