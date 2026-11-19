"""Automatic pixel-level verdict for passing signals in a run directory.

For each passing signal: run pixel_check.py (TESScut cutouts, per-transit local baselines,
stars moved to the observing epoch), then require:
  * combined in-aperture dip > 4 sigma,
  * the dip seen (> 2 sigma) in at least 2/3 of sectors, and no sector brightening (< -2 sigma),
  * no other star (at its position at the observing epoch) fits the difference images better
    than the target by more than max(9, 5% of the target's total fit improvement).
Usage: .venv/bin/python auto_vet.py RUN_DIR [FFI]"""
import json, glob, os, subprocess, sys
import numpy as np

run = sys.argv[1]
env = dict(os.environ, FFI="1") if len(sys.argv) > 2 else dict(os.environ)
out = []
for f in sorted(glob.glob(f"out/{run}/*.json")):
    r = json.load(open(f))
    for c in r.get("signals", []):
        if not c.get("pass"):
            continue
        tic = r["tic"]
        args = [str(tic), str(c["period"]), str(c["t0_btjd"]), str(c["duration_h"]), ",".join(map(str, r["sectors"]))]
        pj = f"out/pixels_{tic}.json"
        if os.path.exists(pj):
            os.remove(pj)
        subprocess.run(["nice", "-n", "10", ".venv/bin/python", "pixel_check.py"] + args,
                       capture_output=True, text=True, env=env)
        why = []
        if not os.path.exists(pj):
            why.append("pixel check failed to run")
        else:
            p = json.load(open(pj))
            d = np.array([x["depth"] for x in p["sectors"]]); e = np.array([x["err"] for x in p["sectors"]])
            w = 1 / e ** 2; mu = (d * w).sum() / w.sum(); mu_err = w.sum() ** -0.5
            chi2r = float((((d - mu) / e) ** 2).sum() / max(len(d) - 1, 1))
            if mu / mu_err < 4: why.append(f"weak in pixels ({mu:.0f}+-{mu_err:.0f} ppm)")
            if len(d) >= 2 and (d / e > 2).mean() < 2 / 3:
                why.append(f"dip missing in pixels in {int(((d / e) <= 2).sum())}/{len(d)} sectors")
            if (d / e < -2).any(): why.append("brightens at transit in some sector")
            j = p["joint"]; tgt = j[str(tic)]
            for k, v in j.items():
                if k != str(tic) and v["delta"] < -max(9.0, 0.05 * tgt["improvement"]):
                    why.append(f"neighbour TIC {k} ({v['sep']:.0f} arcsec) fits better by {-v['delta']:.0f}")
            c["pixel"] = dict(depth=float(mu), err=float(mu_err), chi2r=chi2r, best_star=p["best_star"])
        c["pixel_fails"] = why
        c["pixel_pass"] = not why
        out.append((tic, c["period"], c["sde"], c.get("radius_earth"), why))
        print(f"TIC {tic} P={c['period']:.4f} SDE={c['sde']:.1f} -> {'PIXEL PASS' if not why else '; '.join(why)}", flush=True)
    json.dump(r, open(f, "w"), indent=1, default=float)
