#!/usr/bin/env python3
"""Sweeps KLT tracker settings on TartanAir and reports, per setting, the
track life-cycle metrics (the plan's main tracker goal: tracks as long as
physically possible, few drop-and-rediscover) next to the accuracy metrics
that guard against tracking wrong points.

Per setting, over one trajectory (P000) of each TartanAir set:
  completeness   tracked length / visible length per track (median, mean)
  premature      share of ended tracks that died while their point was visible,
                 and the main cause
  redetected     share of new tracks that re-detect a dropped, still visible one
  overtracked    share of tracks that followed their point out of view/occlusion
  gross          share of observations (ages 1-5) > 2 px beyond the oracle floor
  excess         median error minus median oracle error (ages 1-5)
  features       median live features per frame

    python python/tracker_sweep.py [--fb 0.5,1,2] [--ransac 1,2] [--extra "--window 31"]

Writes sweep_out/<setting>/... and sweep_out/summary.csv.
"""
import argparse
import collections
import csv
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from msckf_gates.common import REPO, fnum, large_data_dir, read_csv, run, tool

SETS = ["RetroOffice/Data_easy", "RetroOffice/Data_hard", "Office/Data_hard", "OldTownFall/Data_easy",
        "OldTownFall/Data_hard", "OldTownNight/Data_hard", "AmericanDiner/Data_hard"]


def summarize(dirs):
    comp, ended, prem, causes, new, red, over, gross, err, orc, feats = [], 0, 0, collections.Counter(), 0, 0, 0, [], [], [], []
    for d in dirs:
        for r in read_csv(d / "tracks.csv"):
            if r["gt_valid"] != "1":
                continue
            new += 1
            comp.append(min(1.0, int(r["tracked_len"]) / int(r["visible_len"])))
            red += r["redetection_of"] != "-1"
            over += r["overtracked"] == "1"
            if r["end"] != "alive":
                ended += 1
                if r["premature"] == "1":
                    prem += 1
                    causes[r["end"]] += 1
        for r in read_csv(d / "observations.csv"):
            if r["status"] == "ok" and 1 <= int(r["age"]) <= 5:
                o = fnum(r["oracle_px"])
                if np.isfinite(o):
                    err.append(fnum(r["error_px"]))
                    orc.append(o)
        feats += [fnum(r["features"]) for r in read_csv(d / "frames.csv")]
    err, orc = np.array(err), np.array(orc)
    return {
        "completeness_median": float(np.median(comp)),
        "completeness_mean": float(np.mean(comp)),
        "premature_pct": 100.0 * prem / max(ended, 1),
        "main_cause": causes.most_common(1)[0][0] if causes else "",
        "redetected_pct": 100.0 * red / max(new, 1),
        "overtracked_pct": 100.0 * over / max(new, 1),
        "gross_pct": 100.0 * float(np.mean(err > orc + 2.0)),
        "excess_px": float(np.median(err) - np.median(orc)),
        "features": float(np.median(feats)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tartanair", type=Path, default=large_data_dir() / "tartanair")
    ap.add_argument("--fb", default="0.5,1,2,4", help="forward-backward thresholds (px)")
    ap.add_argument("--ransac", default="1,2,3", help="RANSAC thresholds (px at the image centre)")
    ap.add_argument("--extra", default="", help="extra tracker_eval options for every setting")
    ap.add_argument("--out", type=Path, default=REPO / "sweep_out")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    trajs = [args.tartanair / s / "P000" for s in SETS]
    settings = list(itertools.product(args.fb.split(","), args.ransac.split(",")))
    jobs = []
    for fb, rs in settings:
        name = f"fb{fb}_ransac{rs}" + (("_" + args.extra.replace("--", "").replace(" ", "")) if args.extra else "")
        for t in trajs:
            d = args.out / name / "_".join(t.parts[-3:])
            cmd = [tool("tracker_eval"), "tartanair", "--traj", t, "--out", d, "--fb-px", fb, "--ransac-px", rs]
            cmd += args.extra.split()
            jobs.append((name, d, cmd))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in [ex.submit(run, cmd, d.with_suffix(".log")) for _, d, cmd in jobs]:
            f.result()

    rows = []
    for name in dict.fromkeys(n for n, _, _ in jobs):
        s = summarize([d for n, d, _ in jobs if n == name])
        rows.append({"setting": name, **s})
    keys = list(rows[0].keys())
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(" ".join(f"{k:>20s}" for k in keys))
    for r in rows:
        print(" ".join(f"{r[k]:>20.3f}" if isinstance(r[k], float) else f"{r[k]:>20s}" for k in keys))
    print(f"summary: {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
