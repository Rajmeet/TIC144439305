"""Look-elsewhere-corrected check that OLD data independently supports a period found in NEW data.
Statistic: max over the allowed period window of the old-data depth significance at the
new-data ephemeris for each trial period. Null: same statistic after shifting the old data in
time by a random fraction of the period (destroys a real signal, keeps noise properties).
Usage: FFI=1 .venv/bin/python null_test.py TIC PERIOD DURATION_H OLD_MAX_SECTOR [N_NULL]"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.timeseries import BoxLeastSquares
import search_v2 as S

tic, P0, dur_h, old_max = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
NN = int(sys.argv[5]) if len(sys.argv) > 5 else 40
S.MAX_SECTORS = 99
raw = S.download_ffi(tic)
prot, ppow, _ = S.spin_period(raw)
t, f, sec, _ = S.detrend(raw, prot, ppow)
new, old = sec > old_max, sec <= old_max
tn, fn, to, fo = t[new], f[new], t[old], f[old]
D = dur_h / 24
bn = BoxLeastSquares(tn, fn)
r = bn.power(np.linspace(P0 * 0.999, P0 * 1.001, 4001), [D], objective="snr")
P = r.period[np.argmax(r.power)]
span_back = np.median(tn) - np.median(to)
# window: periods for which the old-data transit times differ by up to ~ +-1 full period
win = np.linspace(P - P * P / span_back, P + P * P / span_back, 401)
T0s = bn.power(win, [D], objective="snr").transit_time

def stat(to_, fo_):
    best = -9
    for Pt, T0 in zip(win, T0s):
        x = np.abs(((to_ - T0) / Pt + 0.5) % 1 - 0.5) * Pt
        intr, oot = x < D / 2, (x > D) & (x < 4 * D)
        if intr.sum() < 5: continue
        d = np.median(fo_[oot]) - np.mean(fo_[intr])
        e = 1.4826 * np.median(np.abs(fo_[oot] - np.median(fo_[oot]))) / np.sqrt(intr.sum())
        best = max(best, d / e)
    return best

real = stat(to, fo)
rng = np.random.default_rng(1)
nulls = np.array([stat(to + rng.uniform(0.15, 0.85) * P, fo) for _ in range(NN)])
print(f"TIC {tic}: new-data P={P:.6f}; window +-{P*P/span_back:.5f} d ({len(win)} trials)")
print(f"REAL old-data max significance: {real:.2f} sigma")
print(f"NULL (old data time-shifted, {NN} trials): median {np.median(nulls):.2f}, 95th pct {np.percentile(nulls,95):.2f}, max {nulls.max():.2f}")
print(f"fraction of nulls >= real: {(nulls >= real).mean():.3f}  (p-value)")
