"""Recheck 1: is the transit present, at a consistent depth, in light curves from independent pipelines?

Sources: TARS and TESS-SPOC HLSP light curves (MAST), and our own background-subtracted 3x3-aperture
photometry from TESScut cutouts. Each is detrended the same way (biweight, 0.5 d), binned to 10 min,
and the depth measured at the fitted ephemeris; noise is estimated from 38 wrong phases.
Usage: .venv/bin/python recheck_lc.py TIC"""
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
from wotan import flatten

tic = int(sys.argv[1])
fit = json.load(open(f"out/fit_{tic}.json"))
P, T0, D = fit["P_d"][0], fit["T0_BTJD"][0], fit["T14_h"][0] / 24
star = Catalogs.query_object(f"TIC {tic}", radius=0.001, catalog="TIC").to_pandas().iloc[0]


def prep(t, f):
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok] / np.nanmedian(f[ok])
    fl = flatten(t, f, method="biweight", window_length=0.5, break_tolerance=0.3)
    ok = np.isfinite(fl)
    t, fl = t[ok], fl[ok]
    med, sd = np.median(fl), 1.4826 * np.median(np.abs(fl - np.median(fl)))
    keep = fl < med + 4 * sd
    t, fl = t[keep], fl[keep]
    b = np.floor(t / (10 / 1440))
    _, idx, cnt = np.unique(b, return_inverse=True, return_counts=True)
    return np.bincount(idx, t) / cnt, np.bincount(idx, fl) / cnt


def depth(t, f, shift=0.0):
    x = np.abs(((t - T0) / P - shift + 0.5) % 1 - 0.5) * P
    i, o = x < D / 2, (x > D) & (x < 4 * D)
    if i.sum() < 5:
        return np.nan
    return (np.median(f[o]) - np.mean(f[i])) * 1e6


def measure(label, t, f):
    d = depth(t, f)
    nulls = np.array([depth(t, f, s) for s in np.linspace(0.1, 0.9, 41) if abs(s - 0.5) > 0.03])
    e = np.nanstd(nulls)
    return dict(source=label, depth=float(d), err=float(e), sigma=float(d / e))


rows = []
# 1) other pipelines
res = lk.search_lightcurve(f"TIC {tic}", author=["TARS", "TESS-SPOC"])
for i in range(len(res)):
    s = int(str(res[i].mission[0]).split()[-1]); a = str(res[i].author[0]); exp = float(res[i].exptime[0].value)
    if exp > 700:
        continue
    if a == "TARS":  # TARS files are a plain TIME/FLUX table that lightkurve cannot parse
        from astropy.io import fits
        import glob
        try:
            res[i].download(download_dir="out/mast_cache")
        except Exception:
            pass  # file is fetched, only the parse fails
        path = sorted(glob.glob(f"out/mast_cache/mastDownload/HLSP/hlsp_tars_tess_ffi_s{s:04d}-{tic:016d}_tess/*_lc.fits"))[0]
        d = fits.open(path)[1].data
        t, f = prep(np.asarray(d["TIME"], float), np.asarray(d["FLUX"], float))
    else:
        lc = res[i].download(download_dir="out/mast_cache")
        t, f = prep(np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float))
    rows.append(dict(sector=s, **measure(a, t, f)))
    print(f"S{s:3d} {a:10s} depth {rows[-1]['depth']:6.0f} +- {rows[-1]['err']:4.0f} ppm ({rows[-1]['sigma']:.1f} sigma)", flush=True)

# 2) our own photometry from TESScut
tc = lk.search_tesscut(f"TIC {tic}")
for j in range(len(tc)):
    s = int(str(tc[j].mission[0]).split()[-1])
    if s == 2:
        continue
    tpf = tc[j].download(cutout_size=13, download_dir="out/mast_cache")
    t = tpf.time.value
    cube = np.asarray(tpf.flux.value, float)
    good = np.isfinite(cube).all(axis=(1, 2)) & (tpf.quality == 0)
    t, cube = t[good], cube[good]
    yr = Time(np.median(t) + 2457000, format="jd").jyear
    dra = star.pmRA * (yr - 2000) / 3.6e6 / np.cos(np.radians(star.dec)); dde = star.pmDEC * (yr - 2000) / 3.6e6
    x, y = tpf.wcs.world_to_pixel(SkyCoord((star.ra + dra) * u.deg, (star.dec + dde) * u.deg))
    ix, iy = int(round(float(x))), int(round(float(y)))
    edge = np.ones((13, 13), bool); edge[2:-2, 2:-2] = False
    bkg = np.median(cube[:, edge], axis=1)  # per-frame background from the cutout border
    flux = (cube[:, iy - 1:iy + 2, ix - 1:ix + 2] - bkg[:, None, None]).sum(axis=(1, 2))
    tt, ff = prep(t, flux)
    rows.append(dict(sector=s, **measure("own-phot", tt, ff)))
    print(f"S{s:3d} {'own-phot':10s} depth {rows[-1]['depth']:6.0f} +- {rows[-1]['err']:4.0f} ppm ({rows[-1]['sigma']:.1f} sigma)", flush=True)

for src in ("TARS", "TESS-SPOC", "own-phot"):
    r = [x for x in rows if x["source"] == src and np.isfinite(x["depth"])]
    if not r:
        continue
    d = np.array([x["depth"] for x in r]); e = np.array([x["err"] for x in r]); w = 1 / e ** 2
    mu, se = (d * w).sum() / w.sum(), w.sum() ** -0.5
    chi2r = (((d - mu) / e) ** 2).sum() / max(len(d) - 1, 1)
    print(f"{src:10s}: combined {mu:.0f} +- {se:.0f} ppm ({mu/se:.1f} sigma) over {len(d)} sectors; consistency chi2_red {chi2r:.2f}")
json.dump(rows, open(f"out/recheck_lc_{tic}.json", "w"), indent=1)
