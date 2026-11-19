"""MCMC transit fit for one candidate on all available FFI data.

Free parameters: T0, P, Rp/R*, impact parameter b, stellar density (Gaussian prior from TIC
mass and radius), log jitter. Quadratic limb darkening fixed to typical TESS-band values for
an early-mid M dwarf. Models are supersampled to the 10-min bins.

Usage: FFI=1 .venv/bin/python fit_candidate.py TIC PERIOD T0_BTJD DURATION_H
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import batman
import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs

import search_v2 as S

tic, P0, T00, dur_h = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
star = Catalogs.query_object(f"TIC {tic}", radius=0.001, catalog="TIC").to_pandas().iloc[0]
R, M = float(star.rad), float(star.mass)
eR = float(star.e_rad) if np.isfinite(star.e_rad) else 0.1 * R
eM = float(star.e_mass) if np.isfinite(star.e_mass) else 0.1 * M
rho_sun = 1.408  # g/cc
rho0 = M / R ** 3 * rho_sun
erho = rho0 * np.hypot(eM / M, 3 * eR / R)
print(f"star: R={R:.3f}+-{eR:.3f} Rsun, M={M:.3f}+-{eM:.3f} Msun, Teff={star.Teff:.0f} K, "
      f"rho={rho0:.1f}+-{erho:.1f} g/cc, d={star.d:.1f} pc")

S.MAX_SECTORS = 99
raw = S.download_ffi(tic)
prot, ppow, _ = S.spin_period(raw)
t, f, sec, _ = S.detrend(raw, prot, ppow)
# keep data within 4 transit durations of each transit (speeds the fit)
x = np.abs(((t - T00) / P0 + 0.5) % 1 - 0.5) * P0
m = x < 4 * dur_h / 24
t, f, sec = t[m], f[m], sec[m]
sig0 = 1.4826 * np.median(np.abs(f - np.median(f)))
print(f"{len(t)} points in {len(set(sec))} sectors {sorted(set(sec))}; per-point scatter {sig0*1e6:.0f} ppm")
# reference epoch near the middle of the data, to decorrelate T0 and P
n_mid = np.round((np.median(t) - T00) / P0)
Tref = T00 + n_mid * P0

params = batman.TransitParams()
params.t0, params.per, params.rp, params.a, params.inc = Tref, P0, 0.02, 10.0, 89.0
params.ecc, params.w, params.limb_dark, params.u = 0.0, 90.0, "quadratic", [0.25, 0.35]
model = batman.TransitModel(params, t, supersample_factor=7, exp_time=10 / 1440)
G = 6.674e-8


def a_over_r(rho, P):
    return ((G * rho * (P * 86400) ** 2) / (3 * np.pi)) ** (1 / 3)


def lc(theta):
    T0, P, rp, b, rho, lnjit, f0 = theta
    params.t0, params.per, params.rp = T0, P, rp
    a = a_over_r(rho, P)
    params.a = a
    cosi = b / a
    if cosi >= 1:
        return None, a
    params.inc = np.degrees(np.arccos(cosi))
    return f0 * model.light_curve(params), a


def lnprob(theta):
    T0, P, rp, b, rho, lnjit, f0 = theta
    if not (0.001 < rp < 0.3 and 0 <= b < 1 + rp and 0.5 < rho < 100 and -15 < lnjit < -3
            and abs(T0 - Tref) < 0.1 and abs(P - P0) < 0.01):
        return -np.inf
    mod, a = lc(theta)
    if mod is None:
        return -np.inf
    s2 = sig0 ** 2 + np.exp(2 * lnjit)
    lp = -0.5 * ((rho - rho0) / erho) ** 2
    return lp - 0.5 * np.sum((f - mod) ** 2 / s2 + np.log(s2))


depth0 = 5e-4
p0 = np.array([Tref, P0, np.sqrt(depth0), 0.3, rho0, -9.0, 1.0])
scale = np.array([5e-4, 2e-6, 2e-3, 0.1, erho * 0.3, 0.5, 2e-5])
nw, nd = 28, len(p0)
pos = p0 + scale * np.random.default_rng(0).standard_normal((nw, nd))
pos[:, 3] = np.abs(pos[:, 3])
sampler = emcee.EnsembleSampler(nw, nd, lnprob)
sampler.run_mcmc(pos, 4000, progress=False)
chain = sampler.get_chain(discard=1500, thin=10, flat=True)
lnp = sampler.get_log_prob(discard=1500, thin=10, flat=True)
print(f"acceptance {np.mean(sampler.acceptance_fraction):.2f}")


def q(v):
    lo, mid, hi = np.percentile(v, [16, 50, 84])
    return mid, hi - mid, mid - lo


T0s, Ps, rps, bs, rhos = chain[:, 0], chain[:, 1], chain[:, 2], chain[:, 3], chain[:, 4]
Rs = np.random.default_rng(1).normal(R, eR, len(rps))
Rp = rps * Rs * 109.08
dep = rps ** 2 * 1e6
aR = a_over_r(rhos, Ps)
inc = np.degrees(np.arccos(bs / aR))
T14 = Ps / np.pi * np.arcsin(np.sqrt((1 + rps) ** 2 - bs ** 2) / aR / np.sin(np.radians(inc))) * 24
Teff = float(star.Teff)
S_earth = (R ** 2) * (Teff / 5772) ** 4 / ((aR * R * 0.00465047) ** 2)
Teq = Teff * np.sqrt(1 / (2 * aR)) * (1 - 0.3) ** 0.25
res = {}
for name, v in (("T0_BTJD", T0s), ("P_d", Ps), ("Rp_Rs", rps), ("depth_ppm", dep), ("b", bs),
                ("rho_star", rhos), ("a_Rs", aR), ("T14_h", T14), ("Rp_Re", Rp),
                ("S_Searth", S_earth), ("Teq_K_A0.3", Teq)):
    res[name] = q(v)
    mid, up, lo = res[name]
    print(f"  {name:12s} {mid:.7g} +{up:.3g} -{lo:.3g}")
best = chain[np.argmax(lnp)]
res["T0_BJD"] = (res["T0_BTJD"][0] + 2457000, res["T0_BTJD"][1], res["T0_BTJD"][2])
json.dump({k: list(map(float, v)) for k, v in res.items()}, open(f"out/fit_{tic}.json", "w"), indent=1)
np.save(f"out/fit_{tic}_chain.npy", chain)

# figure: folded data + best model, per-sector folds
mod, _ = lc(best)
ph = ((t - best[0]) / best[1] + 0.5) % 1 - 0.5
hrs = ph * best[1] * 24
o = np.argsort(hrs)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot(hrs, (f - 1) * 1e6, ".", ms=1.5, color="#b8c2d0", zorder=0)
e = np.linspace(-4 * dur_h, 4 * dur_h, 41)
mid = (e[1:] + e[:-1]) / 2
idx = np.digitize(hrs, e) - 1
bm = np.array([np.mean((f[idx == k] - 1) * 1e6) for k in range(40)])
be = np.array([np.std((f[idx == k] - 1) * 1e6) / np.sqrt((idx == k).sum()) for k in range(40)])
ax[0].errorbar(mid, bm, be, fmt="o", ms=4, color="#1f2d44")
ax[0].plot(hrs[o], (mod[o] - 1) * 1e6, color="#c0392b", lw=1.5)
ax[0].set_ylim(-1500, 900); ax[0].set_xlim(-4 * dur_h, 4 * dur_h)
ax[0].set_xlabel("Hours from mid-transit"); ax[0].set_ylabel("Relative flux (ppm)")
ax[0].set_title(f"TIC {tic}: all TESS sectors folded, P = {res['P_d'][0]:.6f} d")
for k, s in enumerate(sorted(set(sec))):
    mm = sec == s
    idx = np.digitize(hrs[mm], e) - 1
    b2 = np.array([np.mean((f[mm][idx == j] - 1) * 1e6) if (idx == j).any() else np.nan for j in range(40)])
    ax[1].plot(mid, b2 - 900 * k, "o-", ms=3, lw=0.8, label=f"S{s}")
    ax[1].plot(hrs[o], (mod[o] - 1) * 1e6 - 900 * k, color="#c0392b", lw=1)
ax[1].set_xlim(-4 * dur_h, 4 * dur_h); ax[1].set_xlabel("Hours from mid-transit")
ax[1].set_title("Each sector separately (offset)"); ax[1].legend(fontsize=7, loc="lower right")
ax[1].set_yticks([])
fig.tight_layout(); fig.savefig(f"out/fit_{tic}.png", dpi=120)
print("saved", f"out/fit_{tic}.png")
