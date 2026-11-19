"""Per-transit look at one signal: does every predicted transit show the dip?
Usage: .venv/bin/python inspect_transits.py TIC PERIOD T0_BTJD DURATION_H"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import search_v2 as S

tic, P, t0, dur_h = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
raw = S.download(tic)
prot, ppow, _ = S.spin_period(raw)
t, f, sec, win = S.detrend(raw, prot, ppow)
dur = dur_h / 24
ep = np.round((t - t0) / P).astype(int); dt = t - (t0 + ep * P)
sd = 1.4826 * np.median(np.abs(f - np.median(f)))
rows = []
for e in np.unique(ep[np.abs(dt) < dur / 2]):
    m = (ep == e) & (np.abs(dt) < dur / 2)
    o = (ep == e) & (np.abs(dt) > dur) & (np.abs(dt) < 4 * dur)
    if m.sum() < 3 or o.sum() < 6: continue
    d = (np.median(f[o]) - np.mean(f[m])) * 1e6
    rows.append((e, int(sec[m][0]), d, sd / np.sqrt(m.sum()) * 1e6))
print(f"spin {prot:.2f} d, window {win:.2f} d, scatter {sd*1e6:.0f} ppm")
for e, s, d, err in rows:
    print(f"epoch {e:5d} sector {s:3d} depth {d:7.0f} +- {err:4.0f} ppm  ({d/err:5.1f} sigma)")
dd = np.array([r[2] for r in rows]); ee = np.array([r[3] for r in rows])
w = 1 / ee**2; mean = (dd * w).sum() / w.sum()
chi2 = (((dd - mean) / ee) ** 2).sum()
print(f"mean depth {mean:.0f} ppm; consistency chi2 {chi2:.1f} for {len(dd)-1} dof; "
      f"events >2 sigma: {(dd/ee>2).sum()}/{len(dd)}")
n = len(rows); cols = 6; rws = int(np.ceil(n / cols))
fig, ax = plt.subplots(rws, cols, figsize=(cols * 2.6, rws * 2.1), sharey=True, squeeze=False)
for a, (e, s, d, err) in zip(ax.flat, rows):
    m = (ep == e) & (np.abs(dt) < 5 / 24)
    a.plot(dt[m] * 24, (f[m] - 1) * 1e6, ".", ms=3, color="#35507a")
    a.axvspan(-dur_h / 2, dur_h / 2, color="#c0392b", alpha=0.12)
    a.set_title(f"E{e} S{s} {d:.0f}±{err:.0f}", fontsize=8)
for a in ax.flat[n:]: a.axis("off")
fig.tight_layout(); fig.savefig(f"out/inspect_{tic}.png", dpi=80)
