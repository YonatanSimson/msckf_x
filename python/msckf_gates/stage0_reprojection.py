"""Stage 0 gate: reprojection error on TUM-VI with the published calibrations.

Runs tools/calib_reproj on the calib-cam1 sequence for each published Kalibr
calibration (pinhole-equi -> KB4, omni-radtan -> unified + radtan), with the
intrinsics held fixed and the board pose fitted per image, then checks per
camera, on the corners kept by Kalibr's 4-sigma outlier rule:

- RMSE <= the published reference RMSE, sqrt(mean^2 + std^2) of the camera's
  error norms in TUM-VI's results-imucam-imucalib.txt (~0.105 px);
- at most 5 % of the corners removed as outliers;
- no radial trend: in every radius band with enough corners, the RMSE of the
  kept corners also stays under the reference.

The RMSE is recomputed here from corners.csv, independently of the tool's own
printout. Also reported (not gated): tags detected per frame with and without
the detector's junction-opening step, which is why that step exists.
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import yaml

from msckf_gates.common import GateReport, data_dir, fnum, gate_out, read_csv, run, tool

MODELS = ["pinhole-equi-512", "omni-radtan-512"]
MAX_REMOVED = 0.05
RADIUS_BANDS = [0, 100, 150, 200, 230, 300]  # px from the principal point
MIN_BAND_CORNERS = 200


def published_reference(results_txt):
    """{cam: (mean, std, rmse)} from Kalibr's results-imucam-imucalib.txt."""
    ref = {}
    pat = re.compile(r"Reprojection error \((cam\d)\) \[px\]:\s+mean ([\d.]+), median [\d.]+, std: ([\d.]+)")
    for m in pat.finditer(Path(results_txt).read_text()):
        mean, std = float(m.group(2)), float(m.group(3))
        ref[m.group(1)] = (mean, std, math.hypot(mean, std))
    return ref


def principal_point(camchain, cam):
    c = yaml.safe_load(Path(camchain).read_text())[cam]
    k = c["intrinsics"]
    return (k[3], k[4]) if c["camera_model"] == "omni" else (k[2], k[3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--sequence", default="dataset-calib-cam1_512_16")
    args = ap.parse_args()

    out = gate_out("stage0_reprojection")
    target = args.tumvi / "kalibr_results" / "april_6x6_80x80cm.yaml"
    report = GateReport("Stage 0: TUM-VI reprojection with published calibrations")
    for model in MODELS:
        kdir = args.tumvi / "kalibr_results" / model
        camchain = kdir / "camchain-imucam-imucalib.yaml"
        ref = published_reference(kdir / "results-imucam-imucalib.txt")
        mdir = out / model
        run([tool("calib_reproj"), "--euroc", args.tumvi / args.sequence, "--camchain", camchain,
             "--target", target, "--out", mdir], log=out / f"{model}.log")
        rows = read_csv(mdir / "corners.csv")
        for cam in sorted({r["cam"] for r in rows}):
            rc = [r for r in rows if r["cam"] == cam]
            e = np.array([[fnum(r["du"]), fnum(r["dv"])] for r in rc])
            uv = np.array([[fnum(r["u"]), fnum(r["v"])] for r in rc])
            keep = np.array([r["inlier"] == "1" for r in rc])
            norm = np.linalg.norm(e, axis=1)
            rmse = float(np.sqrt(np.mean(norm[keep] ** 2)))
            removed = 1.0 - keep.mean()
            ref_mean, ref_std, ref_rmse = ref[cam]
            label = f"{model} {cam}"
            report.check(f"{label}: RMSE after 4σ filter", rmse, "<=", ref_rmse, "px")
            report.check(f"{label}: corners removed", 100 * removed, "<=", 100 * MAX_REMOVED, "%")
            cx, cy = principal_point(camchain, cam)
            radius = np.hypot(uv[:, 0] - cx, uv[:, 1] - cy)
            worst, bands = 0.0, []
            for lo, hi in zip(RADIUS_BANDS[:-1], RADIUS_BANDS[1:]):
                sel = keep & (radius >= lo) & (radius < hi)
                if sel.sum() < MIN_BAND_CORNERS:
                    continue
                band_rmse = float(np.sqrt(np.mean(norm[sel] ** 2)))
                out_frac = 1.0 - keep[(radius >= lo) & (radius < hi)].mean()
                worst = max(worst, band_rmse)
                bands.append(f"{lo}-{hi} px: {sel.sum()} kept, RMSE {band_rmse:.3f}, removed {100 * out_frac:.1f}%")
            report.check(f"{label}: worst radius-band RMSE", worst, "<=", ref_rmse, "px")
            report.note(f"{label}: {len(norm)} corners; all-corner RMSE {np.sqrt(np.mean(norm ** 2)):.3f} px; "
                        f"kept |e| mean {norm[keep].mean():.3f}, median {np.median(norm[keep]):.3f} px; "
                        f"published mean {ref_mean:.4f}, std {ref_std:.4f}, RMSE {ref_rmse:.4f} px")
            report.note(f"{label} radius bands: " + "; ".join(bands))
    # Detector ablation: tags per frame without the junction opening.
    model = MODELS[0]
    kdir = args.tumvi / "kalibr_results" / model
    nodil = out / "no-open-junctions"
    run([tool("calib_reproj"), "--euroc", args.tumvi / args.sequence, "--camchain",
         kdir / "camchain-imucam-imucalib.yaml", "--target", target, "--out", nodil,
         "--no-open-junctions"], log=out / "no-open-junctions.log")
    for label, d in (("with junction opening", out / model), ("without", nodil)):
        fr = [r for r in read_csv(d / "frames.csv") if r["cam"] == "cam0"]
        tags = np.array([fnum(r["tags"]) for r in fr])
        report.note(f"detector {label}: cam0 frames with the grid {int((tags > 0).sum())}/{len(tags)}, "
                    f"tags per frame mean {tags.mean():.1f}, median {np.median(tags):.0f} (of 36)")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
