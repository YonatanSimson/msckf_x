#!/usr/bin/env python3
"""Checks that our vendored Basalt optical flow (BasaltFlowTracker, run by
`tracker_eval tracks --tracker basalt`) reproduces the unmodified Basalt
(its Python module, as in tracker_compare.py) on real sequences.

For each sequence and flow type (frame_to_frame, patch), both run with
Basalt's shipped config for the dataset on the first --frames images.
Points are matched by position within each frame, not by id: a single
point dropped in one run and not the other shifts every later id. Gated per
run: at most 0.1 % of either run's observations without a point of the other
run within 0.01 px in the same frame. Not
bit-identical: float evaluation order differs from Basalt's build (its
-O3 -march=native flags alone do not reproduce it), and an occasional point
then converges to a neighbouring minimum (seen: one observation in ~30,000
off by 1.7-6 px in patch mode).

Only frame_to_frame (the adopted mode) is gated. Patch mode is reported:
its tracks keep the detection frame's template for life, so a single
borderline forward-backward decision persists and changes later detections
in that cell (EuRoC MH_01, 500 frames: 0.6 % of points unmatched, median
14 px away, first at frame 15). Consistent with float sensitivity but not
proven; to be resolved if patch mode is ever adopted.

    python python/basalt_flow_check.py [--sets calib,room,euroc,tartan] [--frames 500]

Writes basalt_flow_check_out/<type>/<tracker>/<sequence>/tracks.csv and
basalt_flow_check_out/report.md (with metrics.json).
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from msckf_gates.common import REPO, GateReport, data_dir, large_data_dir, run, tool
from tracker_compare import Sequence, Variant, basalt_config, run_basalt, sequences, vendored_flow_args

MAX_PX = 1e-2  # pixel agreement
MAX_DIFFERING_PCT = 0.1  # share of observations allowed beyond MAX_PX


def run_vendored(seq: Sequence, cfg: dict, d: Path, frames: int) -> None:
    run(
        [
            tool("tracker_eval"),
            "tracks",
            *seq.tracks_args,
            *vendored_flow_args(cfg),
            "--max-frames",
            frames,
            "--out",
            d,
        ],
        d.with_suffix(".log"),
    )


def by_frame(path: Path) -> dict[int, dict[int, np.ndarray]]:
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    out: dict[int, dict[int, np.ndarray]] = {}
    for row in a:
        out.setdefault(int(row[0]), {})[int(row[2])] = row[3:5]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="calib,room,euroc,tartan")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tartanair", type=Path, default=large_data_dir() / "tartanair")
    ap.add_argument("--out", type=Path, default=REPO / "basalt_flow_check_out")
    args = ap.parse_args()

    # One sequence per data kind is enough to exercise every code path.
    seqs, seen = [], set()
    for s in sequences(args.sets.split(","), args.tumvi, args.euroc, args.tartanair):
        if s.kind not in seen:
            seen.add(s.kind)
            seqs.append(dataclasses.replace(s, images=s.images[: args.frames]))

    report = GateReport("Vendored Basalt flow vs unmodified Basalt")
    for flow_type in ("frame_to_frame", "patch"):
        for seq in seqs:
            v = Variant(f"basalt_{flow_type}", "basalt", config={"optical_flow_type": flow_type})
            cfg = basalt_config(seq, v)
            name = seq.name.replace("/", "_")
            ref_dir = args.out / flow_type / "basalt" / name
            our_dir = args.out / flow_type / "vendored" / name
            run_basalt(seq, v, ref_dir)
            run_vendored(seq, cfg, our_dir, len(seq.images))
            ref, ours = by_frame(ref_dir / "tracks.csv"), by_frame(our_dir / "tracks.csv")
            frames = sorted(set(ref) | set(ours))
            n_ref = n_ours = miss_ref = miss_ours = 0
            first = None
            for k in frames:
                a = np.array(list(ref.get(k, {}).values())).reshape(-1, 2)
                b = np.array(list(ours.get(k, {}).values())).reshape(-1, 2)
                if len(a) and len(b):
                    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
                    ma, mb = int(np.sum(dist.min(axis=1) > MAX_PX)), int(np.sum(dist.min(axis=0) > MAX_PX))
                else:
                    ma, mb = len(a), len(b)
                n_ref, n_ours, miss_ref, miss_ours = n_ref + len(a), n_ours + len(b), miss_ref + ma, miss_ours + mb
                if (ma or mb) and first is None:
                    first = k
            label = f"{flow_type} {seq.name}"
            unmatched = (
                (f"Basalt points without a vendored one within {MAX_PX:g} px", 100.0 * miss_ref / max(n_ref, 1)),
                (f"vendored points without a Basalt one within {MAX_PX:g} px", 100.0 * miss_ours / max(n_ours, 1)),
            )
            for what, pct in unmatched:
                if flow_type == "frame_to_frame":
                    report.check(f"{label}: {what}", pct, "<=", MAX_DIFFERING_PCT, "%")
                else:
                    report.note(f"{label}: {what}: {pct:.3g} % (not gated)")
            report.note(
                f"{label}: {len(frames)} frames, {n_ref} / {n_ours} observations (Basalt / vendored), "
                f"first frame with an unmatched point: {first if first is not None else '-'}"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    raise SystemExit(report.finish(args.out))


if __name__ == "__main__":
    main()
