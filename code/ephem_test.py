"""Pre-dicted-ephemeris test: fit P, T0 on the newer data only, predict transit times in
older data, and measure the depth there; compare with depths at wrong (shifted) phases.
Usage: FFI=1 .venv/bin/python ephem_test.py TIC PERIOD DURATION_H OLD_MAX_SECTOR"""
import sys, warnings, os
warnings.filterwarnings("ignore")
import numpy as np
from astropy.timeseries import BoxLeastSquares
import search_v2 as S

tic, P0, dur_h, old_max = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
S.MAX_SECTORS = 99
raw = S.download_ffi(tic)
prot, ppow, _ = S.spin_period(raw)
t, f, sec, win = S.detrend(raw, prot, ppow)
new, old = sec > old_max, sec <= old_max
print("new sectors", sorted(set(sec[new])), " old sectors", sorted(set(sec[old])))
dur = dur_h / 24
# 1) fine ephemeris from new data only
bls = BoxLeastSquares(t[new], f[new])
grid = np.linspace(P0 * 0.999, P0 * 1.001, 4001)
r = bls.power(grid, [dur * 0.8, dur, dur * 1.25], objective="snr")
j = np.argmax(r.power); P, T0, D = r.period[j], r.transit_time[j], r.duration[j]
# bootstrap the period uncertainty by resampling transits
ep = np.round((t[new] - T0) / P).astype(int)
eps = np.unique(ep[np.abs(t[new] - (T0 + ep * P)) < D / 2])
rng = np.random.default_rng(0); Ps = []
for _ in range(60):
    pick = rng.choice(eps, len(eps), replace=True)
    m = np.isin(ep, pick)
    rr = BoxLeastSquares(t[new][m], f[new][m]).power(grid, [D], objective="snr")
    Ps.append(rr.period[np.argmax(rr.power)])
sP = np.std(Ps)
# move T0 to the middle of the new data
n_mid = np.round((np.median(t[new]) - T0) / P); T0m = T0 + n_mid * P
n_back = (np.median(t[new]) - np.median(t[old])) / P
print(f"new-data ephemeris: P = {P:.6f} +- {sP:.6f} d, duration {D*24:.2f} h; "
      f"timing uncertainty at old data ~ {n_back * sP * 1440:.0f} min")
# 2) depth at predicted times in old data, and at 40 wrong phases
def depth_at(tt, ff, phase_shift):
    ph = ((tt - T0m) / P - phase_shift + 0.5) % 1 - 0.5
    x = np.abs(ph * P)
    intr, oot = x < D / 2, (x > D) & (x < 4 * D)
    if intr.sum() < 5: return np.nan, np.nan, 0
    d = np.median(ff[oot]) - np.mean(ff[intr])
    e = 1.4826 * np.median(np.abs(ff[oot] - np.median(ff[oot]))) / np.sqrt(intr.sum())
    n_ev = len(np.unique(np.round((tt[intr] - T0m) / P - phase_shift)))
    return d * 1e6, e * 1e6, n_ev
d_new = depth_at(t[new], f[new], 0)
d_old = depth_at(t[old], f[old], 0)
nulls = np.array([depth_at(t[old], f[old], s)[0] for s in np.linspace(0.1, 0.9, 41) if abs(s - 0.5) > 0.03])
print(f"new data depth at ephemeris: {d_new[0]:.0f} +- {d_new[1]:.0f} ppm ({d_new[2]} transits)")
print(f"OLD data depth at PREDICTED times: {d_old[0]:.0f} +- {d_old[1]:.0f} ppm ({d_old[2]} transits), "
      f"{d_old[0]/d_old[1]:.1f} sigma")
print(f"OLD data at {len(nulls)} wrong phases: mean {np.nanmean(nulls):.0f}, std {np.nanstd(nulls):.0f} ppm; "
      f"fraction >= predicted-time depth: {(nulls >= d_old[0]).mean():.3f}")

# 3) joint search over the allowed period window using ALL data: does old data add signal?
win = np.linspace(P - 4 * sP, P + 4 * sP, 6001)
rj = BoxLeastSquares(t, f).power(win, [D], objective="snr")
k = np.argmax(rj.power); Pj, T0j = rj.period[k], rj.transit_time[k]
T0m, P = T0j, Pj
dn, do = depth_at(t[new], f[new], 0), depth_at(t[old], f[old], 0)
nulls = np.array([depth_at(t[old], f[old], s)[0] for s in np.linspace(0.1, 0.9, 41) if abs(s - 0.5) > 0.03])
print(f"\nJOINT best period in window: {Pj:.6f} d (window {win[0]:.5f}-{win[-1]:.5f})")
print(f"  at joint ephemeris -> new data {dn[0]:.0f}+-{dn[1]:.0f} ppm, old data {do[0]:.0f}+-{do[1]:.0f} ppm "
      f"({do[0]/do[1]:.1f} sigma); old-data wrong phases std {np.nanstd(nulls):.0f} ppm")
# how special is the old-data depth at the joint best period vs other periods in the window?
olds = []
for Pt in win[::60]:
    P = Pt
    bb = BoxLeastSquares(t[new], f[new]).power([Pt], [D], objective="snr")
    T0m = bb.transit_time[0]
    olds.append(depth_at(t[old], f[old], 0)[0] / max(depth_at(t[old], f[old], 0)[1], 1))
olds = np.array(olds)
print(f"  old-data depth significance across {len(olds)} trial periods in window: max {np.nanmax(olds):.1f}, "
      f"median {np.nanmedian(olds):.1f}; fraction >= 3 sigma: {(olds >= 3).mean():.2f}")
