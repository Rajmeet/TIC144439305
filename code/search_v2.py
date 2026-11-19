"""Transit search v2 over a target list.

Changes from v1 (search_run.py):
  * Starspot removal adapts to each star: measure its spin period, then use a
    robust biweight filter (wotan) with a window short enough to follow the spots
    but long enough (>= 3 h) not to eat a ~1 h transit.
  * Up to N_SIGNALS searches per star: after each peak, its transits are masked and
    the search repeats, so a planet cannot hide behind a stronger false signal.
  * New false-alarm checks:
      - wave test: a smooth 2-harmonic sine wave at the period must not fit the data
        about as well as a transit-shaped box does;
      - shoulders on BOTH sides of the dip must sit at the baseline (|frac| < 0.4);
      - dip must not be longer than a planet on that orbit around that star could make;
      - period must not match the star's own spin period or its harmonics.
  * Optional injection of a fake planet into the raw data (for sensitivity tests).

Usage:
  .venv/bin/python search_v2.py RUN_NAME TARGETS_CSV [n_workers]
  INJECT=1 ... injects one fake planet per star (params recorded in each result).
"""
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares, LombScargle

warnings.filterwarnings("ignore")
import lightkurve as lk  # noqa: E402
from wotan import flatten  # noqa: E402

OUT = Path(__file__).parent / "out"
CACHE = OUT / "mast_cache"
MAX_SECTORS = 6
PMIN, PMAX = 0.5, 15.0
DURATIONS_H = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
N_SIGNALS = 3
INJECT = os.environ.get("INJECT") == "1"
R_EARTH_PER_RSUN = 109.1


# ---------------------------------------------------------------- data
FFI = os.environ.get("FFI") == "1"


def download(tic):
    """Raw normalised light curves, one per sector (cached on disk).
    Default: SPOC 2-min PDCSAP. FFI=1: TESS-SPOC FFI PDCSAP, else QLP, per sector."""
    if FFI:
        return download_ffi(tic)
    res = lk.search_lightcurve(f"TIC {tic}", author="SPOC", exptime=120)
    if len(res) == 0:
        return []
    sectors = sorted({int(str(m).split()[-1]) for m in res.mission})[-MAX_SECTORS:]
    raw = []
    for i in range(len(res)):
        s = int(str(res[i].mission[0]).split()[-1])
        if s not in sectors or any(p[0] == s for p in raw):
            continue
        try:
            lc = res[i].download(download_dir=str(CACHE), quality_bitmask="default")
        except Exception:
            continue
        lc = lc.remove_nans().normalize()
        raw.append((s, np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float)))
    return raw


def download_ffi(tic):
    res = lk.search_lightcurve(f"TIC {tic}", author=["TESS-SPOC", "QLP"])
    if len(res) == 0:
        return []
    by_sector = {}
    for i in range(len(res)):
        s = int(str(res[i].mission[0]).split()[-1])
        a = str(res[i].author[0])
        exp = float(res[i].exptime[0].value)
        if exp > 700:  # 30-min cadence (2018-20) is too coarse for ~1 h transits
            continue
        rank = 0 if a == "TESS-SPOC" else 1
        if s not in by_sector or rank < by_sector[s][0]:
            by_sector[s] = (rank, i, a)
    raw = []
    for s in sorted(by_sector)[-MAX_SECTORS:]:
        rank, i, a = by_sector[s]
        try:
            lc = res[i].download(download_dir=str(CACHE))
        except Exception:
            continue
        if a == "QLP":
            lc = lc[lc.quality == 0]
        lc = lc.remove_nans().normalize()
        if len(lc) < 100:
            continue
        raw.append((s, np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float)))
    return raw


def spin_period(raw):
    """Star's rotation period from a Lomb-Scargle periodogram of the raw flux (0.1-15 d)."""
    t = np.concatenate([r[1] for r in raw])
    f = np.concatenate([r[2] for r in raw])
    # 30-min bins to speed things up
    b = np.floor(t / (30 / 1440))
    _, idx = np.unique(b, return_inverse=True)
    tb = np.bincount(idx, t) / np.bincount(idx)
    fb = np.bincount(idx, f) / np.bincount(idx)
    ok = np.isfinite(fb)
    freq = np.linspace(1 / 15, 10, 30000)
    ls = LombScargle(tb[ok], fb[ok])
    p = ls.power(freq)
    j = int(np.argmax(p))
    amp = np.std(fb[ok]) * 1e6
    return float(1 / freq[j]), float(p[j]), float(amp)


def detrend(raw, prot, ppow):
    """Biweight flatten with a spin-aware window; returns 10-min binned t, f, sector."""
    if ppow > 0.02 and prot < 3:
        window = float(np.clip(prot / 4, 0.125, 0.5))
    else:
        window = 0.5
    T, F, S = [], [], []
    for s, t, f in raw:
        flat = flatten(t, f, method="biweight", window_length=window, edge_cutoff=0.0,
                       break_tolerance=0.3)
        ok = np.isfinite(flat)
        t, flat = t[ok], flat[ok]
        # clip flares (upward) only
        med, sd = np.median(flat), 1.4826 * np.median(np.abs(flat - np.median(flat)))
        keep = flat < med + 4 * sd
        t, flat = t[keep], flat[keep]
        b = np.floor(t / (10 / 1440))
        _, idx, cnt = np.unique(b, return_inverse=True, return_counts=True)
        tb = np.bincount(idx, t) / cnt
        fb = np.bincount(idx, flat) / cnt
        cad_min = np.median(np.diff(t)) * 1440
        good = cnt >= max(1, int(0.5 * 10 / cad_min))  # at least half the expected points per bin
        T.append(tb[good]); F.append(fb[good]); S.append(np.full(good.sum(), s))
    t, f, s = np.concatenate(T), np.concatenate(F), np.concatenate(S)
    o = np.argsort(t)
    return t[o], f[o], s[o], window


def inject(raw, rng, rstar, mstar):
    """Multiply a fake trapezoid transit into the raw flux. Returns new raw + params."""
    P = float(np.exp(rng.uniform(np.log(1.0), np.log(10.0))))
    rp = float(rng.uniform(1.0, 2.0))  # Earth radii
    depth = (rp / (rstar * R_EARTH_PER_RSUN)) ** 2
    dur = max_duration_h(P, rstar, mstar) * rng.uniform(0.5, 0.9) / 24
    t0 = raw[0][1][0] + rng.uniform(0, P)
    out = []
    for s, t, f in raw:
        ph = np.abs(((t - t0) / P + 0.5) % 1 - 0.5) * P
        ing = dur * 0.1
        prof = np.clip((dur / 2 - ph) / ing, 0, 1)
        out.append((s, t, f * (1 - depth * prof)))
    return out, dict(P=P, t0=t0, rp=rp, depth_ppm=depth * 1e6, dur_h=dur * 24)


def max_duration_h(P, rstar, mstar):
    """Longest transit (central, circular) a planet on period P could have, in hours."""
    if not (rstar == rstar and mstar == mstar):
        rstar, mstar = 0.3, 0.3
    a_rsun = 215.03 * (mstar * (P / 365.25) ** 2) ** (1 / 3)
    return P * 24 / np.pi * np.arcsin(min(1.0, rstar / a_rsun * 1.05))


# ---------------------------------------------------------------- search
def period_grid(base):
    freqs = [1 / PMAX]
    while freqs[-1] < 1 / PMIN:
        fq = freqs[-1]
        freqs.append(fq + (0.5 / 24) * (1 / fq) ** (1 / 3) * fq / (3 * base))
    return np.sort(1 / np.array(freqs))


def _sde(power):
    k = max(len(power) // 200, 1)
    trend = pd.Series(power).rolling(2 * k + 1, center=True, min_periods=1).median().values
    p2 = power - trend
    return (p2 - p2.mean()) / p2.std()


def bls_search(t, f, periods=None):
    """Returns P, t0, duration, SDE, SDE-spectrum, period grid.
    Single season: one coherent BLS on a grid fine enough for the full span.
    Several seasons (gaps > 100 d): stack-slide - coarse BLS per season on one shared grid
    (fine enough for the longest season), powers summed; then the top peaks are refined
    coherently on all the data."""
    gaps = np.where(np.diff(t) > 100)[0]
    edges = np.concatenate([[0], gaps + 1, [len(t)]])
    seasons = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] > 200]
    if len(seasons) <= 1 or (t.max() - t.min()) < 400 or os.environ.get("COHERENT") == "1":
        grid = period_grid(t.max() - t.min())
        bls = BoxLeastSquares(t, f)
        power = np.empty(len(grid))
        for i in range(0, len(grid), 50000):
            power[i:i + 50000] = bls.power(grid[i:i + 50000], DURATIONS_H / 24, objective="likelihood", oversample=5).power
        sde = _sde(power)
        j = int(np.argmax(sde))
        r = bls.power([grid[j]], DURATIONS_H / 24, objective="snr")
        return float(r.period[0]), float(r.transit_time[0]), float(r.duration[0]), float(sde[j]), sde, grid
    longest = max(t[b - 1] - t[a] for a, b in seasons)
    grid = period_grid(longest)
    power = np.zeros(len(grid))
    for a, b in seasons:
        bls = BoxLeastSquares(t[a:b], f[a:b])
        for i in range(0, len(grid), 50000):
            power[i:i + 50000] += bls.power(grid[i:i + 50000], DURATIONS_H / 24, objective="likelihood", oversample=5).power
    sde = _sde(power)
    # refine the top distinct peaks coherently on the full data
    order = np.argsort(sde)[::-1]
    peaks = []
    for j in order:
        if all(abs(grid[j] - grid[p]) / grid[p] > 0.002 for p in peaks):
            peaks.append(j)
        if len(peaks) == 5:
            break
    full = BoxLeastSquares(t, f)
    span = t.max() - t.min()
    best = None
    for j in peaks:
        P0 = grid[j]
        step = (0.5 / 24) * P0 ** (1 / 3) * P0 / (3 * span) * P0  # fine step in period
        fine = np.arange(P0 * 0.998, P0 * 1.002, max(step, 1e-7))
        rr = full.power(fine, DURATIONS_H / 24, objective="snr")
        k = int(np.argmax(rr.power))
        cand = (float(rr.power[k]), float(rr.period[k]), float(rr.transit_time[k]), float(rr.duration[k]), float(sde[j]))
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1], best[2], best[3], best[4], sde, grid


def score(t, f, sec, P, t0, dur):
    epoch = np.round((t - t0) / P).astype(int)
    dt = t - (t0 + epoch * P)
    intr = np.abs(dt) < dur / 2
    oot = np.abs(dt) > dur
    sd = 1.4826 * np.median(np.abs(f[oot] - np.median(f[oot])))
    base = np.median(f[oot])
    depth = base - np.mean(f[intr])
    snr = depth / (sd / np.sqrt(intr.sum()))
    # shoulders: just outside the dip, both sides, must be at baseline
    shl = (dt < -dur) & (dt > -2 * dur)
    shr = (dt > dur) & (dt < 2 * dur)
    sh = [(base - np.mean(f[m])) / depth if m.sum() and depth > 0 else 1.0 for m in (shl, shr)]
    # per-event contributions
    contrib = np.array([(base - np.mean(f[intr & (epoch == e)])) * (intr & (epoch == e)).sum()
                        for e in np.unique(epoch[intr])])
    # per-transit depth consistency: each event's depth vs the mean, using a local baseline
    ev_d, ev_e = [], []
    for e in np.unique(epoch[intr]):
        mi = intr & (epoch == e)
        mo = (epoch == e) & (np.abs(dt) > dur) & (np.abs(dt) < 4 * dur)
        if mi.sum() < 2 or mo.sum() < 6:
            continue
        ev_d.append(np.median(f[mo]) - np.mean(f[mi]))
        ev_e.append(np.std(f[mo]) * np.sqrt(1 / mi.sum() + 1 / mo.sum()))
    ev_d, ev_e = np.array(ev_d), np.array(ev_e)
    if len(ev_d) >= 4:
        w = 1 / ev_e ** 2
        mu = (ev_d * w).sum() / w.sum()
        ev_chi2_red = float((((ev_d - mu) / ev_e) ** 2).sum() / (len(ev_d) - 1))
    else:
        ev_chi2_red = np.nan
    max_frac = float(contrib.max() / contrib.sum()) if contrib.sum() > 0 else 1.0
    oe = []
    for parity in (0, 1):
        m = intr & (epoch % 2 == parity)
        oe.append((base - np.mean(f[m]) if m.sum() else np.nan, sd / np.sqrt(max(m.sum(), 1)), m.sum()))
    if min(oe[0][2], oe[1][2]) < 3:  # one parity unobserved: test not possible
        oe_sigma, oe_frac = 0.0, 0.0
    else:
        oe_sigma = abs(oe[0][0] - oe[1][0]) / np.hypot(oe[0][1], oe[1][1])
        oe_frac = abs(oe[0][0] - oe[1][0]) / depth if depth > 0 else 9.0
    dt2 = t - (t0 + P / 2 + np.round((t - t0 - P / 2) / P) * P)
    m2 = np.abs(dt2) < dur / 2
    sec_sigma = (base - np.mean(f[m2])) / (sd / np.sqrt(max(m2.sum(), 1))) if m2.sum() else 0.0
    sec_frac = (base - np.mean(f[m2])) / depth if (m2.sum() and depth > 0) else 0.0
    sec_cov = sec_hit = 0
    for s in np.unique(sec):
        m = intr & (sec == s)
        if m.sum() < 3:
            continue
        sec_cov += 1
        sec_hit += (base - np.mean(f[m])) / (sd / np.sqrt(m.sum())) > 2
    # split-half test: the dip must appear in both halves of the data at a similar depth
    tmid = np.median(t)
    halves = []
    for m0 in (t < tmid, t >= tmid):
        mi, mo = intr & m0, oot & m0
        if mi.sum() < 3 or mo.sum() < 10:
            halves.append((np.nan, np.nan)); continue
        halves.append((np.median(f[mo]) - np.mean(f[mi]), sd / np.sqrt(mi.sum())))
    if np.all(np.isfinite([h[0] for h in halves])):
        half_min_sigma = float(min(h[0] / h[1] for h in halves))
        half_diff_sigma = float(abs(halves[0][0] - halves[1][0]) / np.hypot(halves[0][1], halves[1][1]))
    else:
        half_min_sigma, half_diff_sigma = np.nan, np.nan
    # wave test: compare a box vs a 2-harmonic sinusoid, both fit by least squares
    ph = 2 * np.pi * ((t - t0) / P)
    X = np.column_stack([np.ones_like(t)] + [fn(k * ph) for k in (1, 2) for fn in (np.cos, np.sin)])
    resid_sine = f - X @ np.linalg.lstsq(X, f, rcond=None)[0]
    Xb = np.column_stack([np.ones_like(t), intr.astype(float)])
    resid_box = f - Xb @ np.linalg.lstsq(Xb, f, rcond=None)[0]
    chi0 = np.sum((f - f.mean()) ** 2)
    d_box = chi0 - np.sum(resid_box ** 2)
    d_sine = chi0 - np.sum(resid_sine ** 2)
    wave_ratio = float(d_sine / d_box) if d_box > 0 else 9.0
    return dict(depth_ppm=float(depth * 1e6), snr=float(snr), n_transits=len(contrib),
                max_event_frac=max_frac, odd_even_sigma=float(oe_sigma), odd_even_frac=float(oe_frac),
                secondary_sigma=float(sec_sigma), secondary_frac=float(sec_frac), sectors_covered=sec_cov,
                sectors_detected=int(sec_hit), scatter_ppm=float(sd * 1e6),
                event_chi2_red=ev_chi2_red, half_min_sigma=half_min_sigma, half_diff_sigma=half_diff_sigma, shoulder_left=float(sh[0]), shoulder_right=float(sh[1]), wave_ratio=wave_ratio)


def subtract_wave(t, f, sec, P, nharm=4):
    """Remove a smooth periodic wave (4 harmonics), fit separately per sector since spots evolve."""
    f = f.copy()
    for s in np.unique(sec):
        m = sec == s
        ph = 2 * np.pi * t[m] / P
        X = np.column_stack([np.ones(m.sum())] + [fn(k * ph) for k in range(1, nharm + 1) for fn in (np.cos, np.sin)])
        coef = np.linalg.lstsq(X, f[m], rcond=None)[0]
        f[m] = f[m] - X @ coef + 1.0
    return f


def near(P, base, mults=(0.5, 1, 2, 3, 1 / 3), tol=0.01):
    return any(abs(P - base * k) / (base * k) < tol for k in mults)


def verdict(c, prot, ppow, rstar, mstar):
    why = []
    sde_min = 7.0 if c.get("stacked") else 9.0
    if c["sde"] < sde_min: why.append(f"weak (SDE<{sde_min:.0f})")
    if c["snr"] < 7.1: why.append("low S/N")
    if c["n_transits"] < 3: why.append("<3 transits")
    if c["max_event_frac"] > 0.5: why.append("one event dominates")
    if c["odd_even_sigma"] > 3 and c["odd_even_frac"] > 0.25: why.append("odd/even differ")
    if (c["secondary_sigma"] > 3 and c["secondary_frac"] > 0.25) or (c["secondary_sigma"] > 5 and c["secondary_frac"] > 0.15):
        why.append("secondary eclipse")
    if c["wave_ratio"] > 0.6: why.append("wave-shaped (spots)")
    if max(abs(c["shoulder_left"]), abs(c["shoulder_right"])) > 0.4: why.append("shoulders not flat")
    if c["duration_h"] > 1.5 * max_duration_h(c["period"], rstar, mstar): why.append("too long for orbit")
    if ppow > 0.05 and near(c["period"], prot): why.append("matches star spin")
    if near(c["period"], 1.0) or near(c["period"], 13.7): why.append("instrumental period")
    if c["sectors_covered"] >= 2 and c["sectors_detected"] < 2: why.append("not seen per sector")
    if c.get("event_chi2_red", np.nan) > 2.5: why.append("transit depths inconsistent")
    if 12.5 <= c["period"] <= 15.0 and c["n_transits"] < 8: why.append("near TESS orbit period")
    if c["half_min_sigma"] == c["half_min_sigma"]:
        if c["half_min_sigma"] < 3: why.append("missing in one half of data")
        if c["half_diff_sigma"] > 3: why.append("depth differs between halves")
    return why


# ---------------------------------------------------------------- driver
def run_star(row, res_dir):
    tic = int(row["ID"])
    out = res_dir / f"{tic}.json"
    if out.exists():
        return json.loads(out.read_text())
    t_start = time.time()
    rstar, mstar = float(row.get("rad", np.nan)), float(row.get("mass", np.nan))
    rec = dict(tic=tic, d_pc=row["d"], Tmag=row["Tmag"], Teff=row["Teff"], rad=rstar,
               control=row.get("control", "") if isinstance(row.get("control"), str) else "")
    try:
        raw = download(tic)
        if not raw:
            rec["status"] = "no data"
            out.write_text(json.dumps(rec))
            return rec
        if INJECT:
            rng = np.random.default_rng(tic)
            raw, rec["injected"] = inject(raw, rng, rstar if rstar == rstar else 0.3,
                                          mstar if mstar == mstar else 0.3)
        prot, ppow, pamp = spin_period(raw)
        t, f, sec, window = detrend(raw, prot, ppow)
        noise = 1.4826 * np.median(np.abs(f - np.median(f))) * 1e6
        rec["noise_ppm_10min"] = float(noise)
        if noise > float(os.environ.get("MAX_NOISE", "5000")):
            rec["status"] = f"too noisy ({noise:.0f} ppm per 10 min)"
            rec["seconds"] = round(time.time() - t_start)
            out.write_text(json.dumps(rec, indent=1, default=float))
            print(f"TIC {tic} skipped: {rec['status']}", flush=True)
            return rec
        periods = period_grid(t.max() - t.min())
        rec.update(status="ok", sectors=[r[0] for r in raw], spin_period=prot, spin_power=ppow,
                   spin_amp_ppm=pamp, window_d=window, n_periods=len(periods))
        cands, mask = [], np.ones(len(t), bool)
        for k in range(N_SIGNALS):
            tm, fm, sm = t[mask], f[mask], sec[mask]
            P, t0, dur, sde, spec, grid = bls_search(tm, fm)
            c = dict(n=k + 1, period=P, t0_btjd=t0, duration_h=dur * 24, sde=sde,
                     stacked=bool((tm.max() - tm.min()) >= 400 and np.any(np.diff(tm) > 100)
                                  and os.environ.get("COHERENT") != "1"))
            c.update(score(tm, fm, sm, P, t0, dur))
            if rstar == rstar:
                c["radius_earth"] = float(np.sqrt(max(c["depth_ppm"], 0) / 1e6) * rstar * R_EARTH_PER_RSUN)
            c["fails"] = verdict(c, prot, ppow, rstar, mstar)
            c["pass"] = not c["fails"]
            cands.append(c)
            if c["pass"] or k == 0:
                plot(tic, k + 1, c, tm, fm, grid, spec, res_dir)
            if sde < 7:  # nothing left worth masking for
                break
            if "wave-shaped (spots)" in c["fails"]:
                f = subtract_wave(t, f, sec, P)  # starspot wave: remove it, keep the data
            else:
                ph = np.abs(((t - t0) / P + 0.5) % 1 - 0.5) * P
                mask &= ph > dur  # remove this signal's dips and look again
        rec["signals"] = cands
        rec["pass"] = any(c["pass"] for c in cands)
        if INJECT:
            inj = rec["injected"]
            rec["recovered"] = any(
                abs(c["period"] - inj["P"]) / inj["P"] < 0.005
                or abs(c["period"] - 2 * inj["P"]) / (2 * inj["P"]) < 0.005
                or abs(c["period"] - inj["P"] / 2) / (inj["P"] / 2) < 0.005
                for c in cands if c["sde"] >= 9)
            rec["recovered_pass"] = any(
                abs(c["period"] - inj["P"]) / inj["P"] < 0.005 and c["pass"] for c in cands)
    except Exception as e:
        rec["status"] = f"error: {e}"
        rec["trace"] = traceback.format_exc()[-1000:]
    rec["seconds"] = round(time.time() - t_start)
    out.write_text(json.dumps(rec, indent=1, default=float))
    s = rec.get("signals", [])
    summary = "; ".join(f"P={c['period']:.4f} SDE={c['sde']:.1f} {'PASS' if c['pass'] else ','.join(c['fails'][:3])}"
                        for c in s)
    extra = f" inj P={rec['injected']['P']:.3f} rec={rec['recovered']}/{rec['recovered_pass']}" if INJECT and "injected" in rec else ""
    print(f"TIC {tic} [{rec['seconds']}s] spin={rec.get('spin_period', 0):.2f}d "
          f"win={rec.get('window_d', 0):.2f} | {summary or rec['status']}{extra}", flush=True)
    return rec


def plot(tic, n, c, t, f, periods, sde, res_dir):
    P, t0 = c["period"], c["t0_btjd"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(periods, sde, lw=0.4, color="#35507a")
    ax[0].axvline(P, color="#c0392b", ls="--", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Period (d)"); ax[0].set_ylabel("SDE")
    ax[0].set_title(f"TIC {tic} signal {n}: P={P:.4f} d SDE={c['sde']:.1f}")
    lim = max(3 * abs(c["depth_ppm"]), 600)
    for a, off, title in ((ax[1], 0, "Transit"), (ax[2], 0.5, "Phase 0.5 (secondary)")):
        h = (((t - t0) / P - off + 0.5) % 1 - 0.5) * P * 24
        m = np.abs(h) < 5
        a.plot(h[m], (f[m] - 1) * 1e6, ".", ms=1.5, color="#b8c2d0")
        e = np.linspace(-5, 5, 41); mid = (e[1:] + e[:-1]) / 2
        idx = np.digitize(h[m], e) - 1
        b = [np.mean((f[m][idx == k] - 1) * 1e6) if (idx == k).any() else np.nan for k in range(40)]
        a.plot(mid, b, "o", ms=3, color="#1f2d44")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylim(-lim, lim * 0.6); a.set_xlabel("Hours"); a.set_title(title)
    ax[1].set_ylabel("ppm")
    verdict_txt = "PASS" if c["pass"] else ", ".join(c["fails"])
    fig.suptitle(verdict_txt, fontsize=9, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(res_dir / f"{tic}_s{n}.png", dpi=110); plt.close(fig)


def _job(args):
    return run_star(*args)


if __name__ == "__main__":
    run_name, targets = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    res_dir = OUT / run_name
    res_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(OUT / targets).to_dict("records")
    with Pool(n, maxtasksperchild=5) as pool:
        recs = pool.map(_job, [(r, res_dir) for r in rows], chunksize=1)
    flat = []
    for r in recs:
        for c in r.get("signals", [{}]):
            flat.append({**{k: v for k, v in r.items() if k not in ("signals", "trace", "injected")},
                         **{f"inj_{k}": v for k, v in r.get("injected", {}).items()}, **c})
    pd.DataFrame(flat).to_csv(res_dir / "summary.csv", index=False)
    print("DONE", flush=True)
