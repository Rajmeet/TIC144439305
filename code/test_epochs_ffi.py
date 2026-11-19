"""Independent re-detection test: search older TESS groups separately for a given period.
Usage: .venv/bin/python test_epochs.py TIC PERIOD"""
import sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, lightkurve as lk
import search_v2 as S

tic, Pc = int(sys.argv[1]), float(sys.argv[2])
groups = {"2018-20 (S1-26)": range(1, 27), "2020-22 (S27-55)": range(27, 56),
          "2022-24 (S56-83)": range(56, 84), "2024-26 (S84-110)": range(84, 111)}
S.MAX_SECTORS = 99
raw_all = S.download_ffi(tic)
res = []

out = {}
for name, rng in groups.items():
    raw = [r for r in raw_all if r[0] in rng]
    if not raw: continue
    prot, ppow, _ = S.spin_period(raw)
    t, f, sec, win = S.detrend(raw, prot, ppow)
    periods = S.period_grid(t.max() - t.min())
    # strip the spot wave at the spin period first, as the main search does
    f = S.subtract_wave(t, f, sec, prot)
    P, t0, dur, sde, spec, periods = S.bls_search(t, f)
    near = np.abs(periods - Pc) / Pc < 0.003
    jn = np.argmax(np.where(near, spec, -99))
    rank = int((spec > spec[jn]).sum()) + 1
    c = S.score(t, f, sec, periods[jn], t0, dur)
    # depth at the best period near Pc
    from astropy.timeseries import BoxLeastSquares
    b = BoxLeastSquares(t, f).power(periods[near], np.array([1.0, 1.5, 2.0]) / 24, objective="snr")
    k = np.argmax(b.power)
    st = S.score(t, f, sec, b.period[k], b.transit_time[k], b.duration[k])
    out[name] = dict(sectors=[r[0] for r in raw], spin=prot, top_period=P, top_sde=sde,
                     near_period=float(b.period[k]), near_sde=float(spec[jn]), near_rank=rank,
                     near_depth_ppm=st["depth_ppm"], near_snr=st["snr"], n_tr=st["n_transits"])
    print(name, json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in out[name].items()}), flush=True)
