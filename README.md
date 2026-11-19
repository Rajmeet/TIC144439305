# TIC 144439305.01: a 1.1 Earth-radius, 22.6-hour transiting planet candidate around a nearby M dwarf

**Rajmeet Chandok** (2026). Search, vetting and analysis were carried out with Claude (Anthropic) at the
author's direction. Please cite this record (see `CITATION.cff`) if you use the candidate.

**Status: planet candidate (likely planet by TRICERATOPS criteria; not validated, not confirmed).** Found 2026-11-18 in an independent search
of public NASA TESS full-frame-image data. As of 2026-11-18 the star is not a TOI or CTOI, has no
ExoFOP follow-up, and has no planet in the NASA Exoplanet Archive; nothing is listed within 38′.

| | |
|---|---|
| Star | TIC 144439305 = Gaia DR3 6532568669359184000 = 2MASS J23552819-4407251 |
| Position | RA 358.86748°, Dec −44.12371° (J2000) |
| Brightness | Tmag 12.15, G 13.29, K 9.83 |
| Distance | 62.1 pc (Gaia parallax 16.07 ± 0.01 mas) |
| Star | M dwarf, Teff 3505 K, R = 0.465 ± 0.014 R☉, M = 0.462 ± 0.020 M☉ (TIC 8.2) |
| Gaia binarity indicators | RUWE 1.07, ipd_frac_multi_peak 0, RV χ² p-value 0.95 (no sign of a companion) |
| TIC contamination ratio | 0.026 |
| **False-positive probability** | **0.0092 ± 0.0023 with MOLUSC Gaia companion limits** (0.062 ± 0.004 without); NFPP 0.0044 |

## Planet parameters (MCMC transit fit, sectors 29, 69, 96, 103, 104)

| Parameter | Value |
|---|---|
| Period | 0.9398098 (+0.0000015 / −0.0000020) d |
| Mid-transit T0 | BTJD 3922.3400 ± 0.0016 (BJD_TDB 2460922.3400) |
| Depth | 489 (+68 / −65) ppm |
| Rp/R★ | 0.0221 ± 0.0015 |
| Radius | **1.12 ± 0.08 R⊕** |
| Impact parameter | 0.36 (+0.18 / −0.22) |
| Duration T14 | 1.02 (+0.06 / −0.08) h |
| a/R★ | 6.75 ± 0.21 |
| Insolation | 138 (+9 / −8) S⊕ |
| Equilibrium temperature | 873 ± 14 K (albedo 0.3) |

Fit: batman, quadratic limb darkening fixed (0.25, 0.35), circular orbit, Gaussian prior on stellar
density from the TIC, 10-min supersampling, emcee (28 walkers × 4000 steps). `data/fit_144439305.json`, `figures/transit_fit.png`.

## How it was found and vetted

1. **Search.** Box-least-squares search of 3,580 TESS-FFI-only M dwarfs at 40–70 pc (this star was in the
   second chunk of 500), on QLP / TESS-SPOC light curves, with per-season stacking. SDE 8.2, S/N 8.9,
   113 transits in sectors 29, 69, 96, 103, 104.
2. **Light-curve checks (all pass).** Odd/even depths agree (0.8σ); no secondary eclipse (0.9σ);
   transit depths consistent (χ²_red 1.19); seen in both halves of the data (≥6.0σ each, depths differ 0.4σ);
   seen in all 5 sectors (597, 594, 350, 435, 502 ppm); no single event > 4% of the signal; flat out-of-transit
   shoulders; not a starspot wave; duration consistent with a planet on this orbit around this star
   (maximum central-transit duration 1.07 h).
3. **Pixel check.** TESScut cutouts, per-transit per-pixel linear baselines, all stars propagated by proper
   motion to each sector's epoch. In-aperture dip in 4 of 5 sectors (272, 554, 466, 479 ppm; S96 −197 ± 105 ppm
   while the S96 light curve shows 350 ± 131 ppm). Joint fit of the difference images over all sectors prefers
   the target; the nearest stars that could host the signal are disfavoured: TIC 2055664316 (T = 18.9, 25″,
   Δχ² = +24), TIC 144439304 (T = 15.4, 48″, Δχ² = +55), TIC 144439308 (T = 14.2, 47″, Δχ² = +61).
   Free-position fits land 3–11″ from the target. `figures/difference_images.png`.
4. **Independent confirmation in older data.** With the period window set by 2025–26 data alone, the
   2020 + 2023 data reach 6.49σ at a consistent period and depth; 0 of 200 time-shifted null trials reach it
   (p < 0.005).
5. **Pre-registered test on unsearched data.** Predictions for TESS sectors 105 and 106 (June–July 2026,
   not used in the search or fit; no pipeline light curves existed) were frozen at 2026-11-18 20:55:03 UTC,
   SHA-256 `7c9ad78233b91b76aa40728cc33c18a4d0e4bcb316d20fec71a4c2f298bed688`
   (`data/PREDICTION_TIC144439305_S105_S106.json`), then tested:

   | | depth at predicted times | wrong-phase nulls (95th pct) |
   |---|---|---|
   | S105 (24 transits) | 505 ± 107 ppm (4.7σ) | 244 ppm |
   | S106 (25 transits) | 444 ± 103 ppm (4.3σ) | 221 ppm |
   | **Combined** | **473 ± 74 ppm (6.4σ)** | 149 ppm; 0 of 40 nulls reach it |

   **Verdict under the frozen rule: RECOVERED.** `data/PREREG_RESULT_TIC144439305.json`.
6. **False-positive probability.** TRICERATOPS 1.1 (5 runs × 2×10⁵ samples, 3×3 apertures in S69–S106,
   no high-resolution imaging, no Gaia companion limits): **FPP = 0.062 ± 0.004, NFPP = 0.0039**.
   Scenario probabilities: transit on target 79%, planet on primary of an unresolved bound binary 14%,
   on the secondary 6%, all nearby-star scenarios < 0.4%. `data/triceratops_144439305.json`.

7. **Rechecks (2026-11-19).**
   - *Independent light curves.* Depth at the fitted ephemeris, noise from 38 wrong phases:
     own background-subtracted 3×3 photometry from TESScut, 7 sectors: **504 ± 65 ppm (7.7σ)**, χ²_red 0.81;
     TESS-SPOC (S29, S69): **543 ± 86 ppm (6.3σ)**, χ²_red 0.14; TARS (S29, S69, S96): 290 ± 119 ppm (2.4σ),
     χ²_red 0.08 (TARS errors ~2× larger). None of these share code with the QLP light curves used in the
     search. `data/recheck_lc_144439305.json`.
   - *Source location with the real TESS PRF* (TESS_PRF package, per camera/CCD/sector, stars moved by
     proper motion): the target is the best source over 7 sectors; TIC 2055664316 (T = 18.9, 25″) is worse by
     Δχ² = 47 (~7σ), TIC 144439304 (48″) by 76, TIC 144439308 (47″) by 84. Free-position PRF fits land
     4.7–15″ from the target in 6 of 7 sectors (S96, the weakest, is unconstrained). `data/prf_144439305.json`.
   - *Gaia companion limits.* MOLUSC (commit b26efbf, RUWE + Gaia contrast, 10⁵ companions, age 5 Gyr;
     typo `star_jitter`→`added_jitter` patched locally, `--jitter -1`): 47% of simulated companions survive.
     TRICERATOPS with these limits, 5 runs × 2×10⁵: **FPP = 0.0092 ± 0.0023**, NFPP = 0.0044; transit on target
     92.8%, unresolved-binary primary 6.2%, secondary 0.5%. No neighbours removed. `data/triceratops_144439305_molusc.json`.

## What it is not (yet)

- Not formally validated. With Gaia companion limits the FPP (0.0092) is below the 0.015 threshold of
  Giacalone et al. (2021), but NFPP (0.0044) is above their 0.001 threshold, so the formal class is
  "likely planet". The NFPP comes almost entirely from TIC 144439304 (T = 15.4, 48″), which the PRF analysis
  disfavours by Δχ² = 76; **high-resolution imaging** would close both this and the bound-companion channel.
- Mass unknown. A 1.1 R⊕ rocky planet would induce ~1–2 m/s on this star (needs ESPRESSO/MAROON-X class RVs).
- The transit is shallow (~0.5 mmag) for ground-based photometry at V ≈ 14.5; a 1–2 m telescope with
  good red sensitivity is needed per event, or stacking many events.

## Upcoming transits

`data/ephemeris.csv`: 433 predicted transits from 2026-11-19 to 2027-12-30, propagated from every posterior
sample (1σ ≈ 3.0 min now, 3.9 min by end of 2027). UTC times are geocentric; ingress/egress = mid ± 0.51 h.

## Contents

- `data/` - fit results and posterior chain, pre-registered prediction and its result, independent
  light-curve and PRF rechecks, TRICERATOPS outputs (with and without MOLUSC limits), MOLUSC surviving
  companions (gzipped), predicted transit times, and the draft ExoFOP CTOI row.
- `figures/` - folded transit with model and per-sector folds; difference images; search periodogram.
- `code/` - the search and vetting pipeline (see below).

## Reproducing

Python 3.12 with numpy, scipy, pandas, matplotlib, astropy, astroquery, lightkurve, wotan, batman-package,
emcee, triceratops 1.1, TESS_PRF, tess-point, and MOLUSC (https://github.com/woodml/MOLUSC, commit b26efbf,
with the `star_jitter` -> `added_jitter` typo fixed and `--jitter -1`). Run from a folder containing `code/`'s
scripts and an `out/` folder; scripts download TESS data from MAST on first use.

| Step | Command |
|---|---|
| Target list (FFI-only M dwarfs 40-70 pc) | `python select_ffi2.py` |
| Search + light-curve checks | `FFI=1 python search_v2.py <run> <targets.csv> 3` (driver: `drive_ffi2b.sh`) |
| Automatic pixel verdict | `python auto_vet.py <run> FFI` then `python known_check.py <run> <targets.csv>` |
| Old-data confirmation | `FFI=1 python ephem_test.py 144439305 0.93981 1.0 69`; `FFI=1 python null_test.py 144439305 0.93981 1.0 69 200` |
| Transit fit | `FFI=1 python fit_candidate.py 144439305 0.9398114376736757 2088.767936914173 1.0` |
| Pre-registered S105/106 test | `python prereg_test.py` |
| Independent light curves | `python recheck_lc.py 144439305` |
| PRF source test | `python prf_check.py 144439305` |
| False-positive probability | `FFI=1 python tric_run.py 200000 5 [molusc_kept.csv]` |

## Credits

This work uses data from the TESS mission, funded by the NASA Science Mission Directorate, obtained from
MAST, including QLP (Huang et al. 2020), TESS-SPOC (Caldwell et al. 2020) and TARS light curves and TESScut
cutouts (Brasseur et al. 2019); ESA Gaia DR3; the TESS Input Catalog (Stassun et al. 2019). Tools:
lightkurve, wotan (Hippke et al. 2019), batman (Kreidberg 2015), emcee (Foreman-Mackey et al. 2013),
TRICERATOPS (Giacalone & Dressing 2020; Giacalone et al. 2021), MOLUSC (Wood, Mann & Kraus 2021),
TESS_PRF, tess-point.

## Licence

Code: MIT (`LICENSE`). Text, figures and data: CC BY 4.0 (`LICENSE-CC-BY-4.0`).
