#!/usr/bin/env python3
"""Head-to-head of our KLT tracker and Basalt's optical flow (plan, Stage 1,
"Head-to-head against Basalt"). Both trackers run on the same images and
write the same tracks.csv (frame, t_ns, track, u, v); one scorer answers
three questions for each:

spread    How well spread are the features? Per frame, over an 8x8 grid of
          the valid region (a cell counts if >= 50 % valid): share of cells
          holding a feature (all, and stable ones of age >= 5), evenness
          (normalized entropy of features per cell, 1 = even), and the share
          of new detections landing within 8 px of a live feature (wasted).
survival  How long do tracks survive? Kaplan-Meier survival of track age
          (tracks alive at the end are censored): median, and survival to
          5 / 10 / 20 frames; the observation-weighted median length; and
          the share of tracks dying within 2 frames.
drift     How many drift off target? TUM-VI calib-imu sequences: a static
          AprilGrid filmed at 20 Hz under fast motion. Basalt's AprilGrid
          detector finds the corners in every frame; with the published
          Kalibr calibration (pinhole-equi = Basalt's kb4) their bearings
          give the board-to-bearing homography of each frame. A feature that
          lies inside the detected part of the board is mapped to board
          coordinates in its first such frame (the anchor) and predicted in
          every later frame through that frame's homography and the camera
          model. Drift is the distance between track and prediction. Two
          failure modes are separated: a track is "aliased" when it jumps to
          the same pattern on a neighbouring tag (its board offset is within
          0.2 pitch of a non-zero multiple of the tag pitch; the grid repeats,
          and motion beyond half a pitch per frame lets an unpredicted tracker
          lock onto the wrong copy). Otherwise, up to its first jump, it has
          drifted if it ever exceeds 1 px (and, separately, 2 px).
          Reported for all on-board tracks and for those anchored within
          1.5 px of a grid corner. The floor is the same scoring applied to
          the detected corners themselves, chained per corner id.

Variants: ours (KltTrackerConfig defaults), ours_tuned (best tracker_sweep
setting), ours_tuned_cv (plus constant-velocity prediction), basalt (the
dataset's shipped Basalt config) and basalt_dense (Basalt with a smaller
grid, for a feature budget close to our 200), basalt_patch (Basalt's
"patch" flow: each track is aligned against the patch from its detection
frame instead of the previous frame, so error does not accumulate), and
vendored (our copy of Basalt's flow, BasaltFlowTracker, the adopted front
end, with the shipped config; it should score like basalt).

    python python/tracker_compare.py [--sets calib,room,euroc,tartan] [--variants ...]

Writes tracker_compare_out/<variant>/<sequence>_<cam>/tracks.csv,
tracker_compare_out/corners/<sequence>_<cam>.npz, and
tracker_compare_out/summary.csv / report.md. Basalt's Python module comes
from $BASALT_PYTHON (default ~/work/basalt/build/relwithdebinfo/python;
build with -DBASALT_BUILD_PYTHON_BINDINGS=ON).
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from msckf_gates.common import REPO, data_dir, euroc_images, large_data_dir, run, tool, write_metrics

BASALT_REPO = Path(os.environ.get("BASALT_REPO", Path.home() / "work" / "basalt"))
sys.path.insert(0, os.environ.get("BASALT_PYTHON", str(BASALT_REPO / "build" / "relwithdebinfo" / "python")))
import basalt  # noqa: E402  (path set above)

EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
TARTAN = [
    "RetroOffice/Data_easy",
    "RetroOffice/Data_hard",
    "Office/Data_hard",
    "OldTownFall/Data_easy",
    "OldTownFall/Data_hard",
    "OldTownNight/Data_hard",
    "AmericanDiner/Data_hard",
]
CALIB = ["calib-imu1", "calib-imu2", "calib-imu3", "calib-imu4"]
CAMS = ["cam0", "cam1"]

OURS_TUNED = ["--levels", "4", "--block-size", "11", "--quality", "0.05", "--fb-px", "1"]
GRID = 8  # coverage grid, cells per side
STABLE_AGE = 5
WASTED_PX = 8.0
CORNER_PX = 1.5  # anchor this close to a grid corner: "corner" subset
DRIFT_PX = (1.0, 2.0)
MIN_CORNERS = 16  # board homography needs >= 4 tags
MAX_FIT_PX = 0.5  # frames whose corner fit RMS exceeds this are not used
AGES = (1, 5, 10, 20)


@dataclass
class Sequence:
    name: str  # e.g. tumvi/calib-imu1/cam0
    kind: str  # calib, room, euroc, tartan
    images: list[tuple[int, Path]]  # (t_ns, path)
    tracks_args: list[str]  # tracker_eval tracks options selecting the images and camera
    basalt_config: Path  # Basalt's shipped config for this dataset
    kb4: np.ndarray | None = None  # fx fy cx cy k1..k4 (calib: drift scoring)
    width: int = 0
    height: int = 0


@dataclass
class Variant:
    name: str
    tracker: str  # ours, basalt (Basalt's own module) or vendored (our copy, BasaltFlowTracker)
    args: list[str] = field(default_factory=list)  # ours: tracker_eval options
    config: dict[str, Any] = field(default_factory=dict)  # basalt: overrides of the shipped config


def variants() -> list[Variant]:
    return [
        Variant("ours", "ours"),
        Variant("ours_tuned", "ours", OURS_TUNED),
        Variant("ours_tuned_cv", "ours", OURS_TUNED + ["--constant-velocity"]),
        Variant("basalt", "basalt"),
        Variant("basalt_dense", "basalt", config={"dense": True}),
        Variant("basalt_patch", "basalt", config={"optical_flow_type": "patch"}),
        Variant("vendored", "vendored"),
    ]


# ------------------------------------------------------------- sequences ----


def tumvi_kb4(camchain: Path, cam: str) -> np.ndarray:
    c = yaml.safe_load(camchain.read_text())[cam]
    assert c["camera_model"] == "pinhole" and c["distortion_model"] == "equidistant"
    return np.array(c["intrinsics"] + c["distortion_coeffs"], dtype=np.float64)


def sequences(sets: list[str], tumvi: Path, euroc: Path, tartan: Path) -> list[Sequence]:
    camchain = tumvi / "kalibr_results" / "pinhole-equi-512" / "camchain-imucam-imucalib.yaml"
    out = []
    for kind, names in (("calib", CALIB), ("room", ["room1"])):
        if kind not in sets:
            continue
        for s in names:
            root = tumvi / f"dataset-{s}_512_16"
            for cam in CAMS:
                out.append(
                    Sequence(
                        f"tumvi/{s}/{cam}",
                        kind,
                        euroc_images(root / "mav0" / cam),
                        ["--euroc", str(root), "--cam", cam, "--camchain", str(camchain)],
                        BASALT_REPO / "data" / "tumvi_512_config.json",
                        tumvi_kb4(camchain, cam),
                    )
                )
    if "euroc" in sets:
        for s in EUROC:
            root = euroc / s
            for cam in CAMS:
                out.append(
                    Sequence(
                        f"euroc/{s}/{cam}",
                        "euroc",
                        euroc_images(root / "mav0" / cam),
                        ["--euroc", str(root), "--cam", cam],
                        BASALT_REPO / "data" / "euroc_config.json",
                    )
                )
    if "tartan" in sets:
        for s in TARTAN:
            traj = tartan / s / "P000"
            img_dir = traj / "image_lcam_front"
            imgs = sorted(img_dir.iterdir()) if img_dir.exists() else []
            out.append(
                Sequence(
                    f"tartanair/{s}/P000",
                    "tartan",
                    [(100_000_000 * i, p) for i, p in enumerate(imgs)],
                    ["--traj", str(traj)],
                    BASALT_REPO / "data" / "euroc_config.json",
                )
            )
    missing = [q for q in out if not q.images or not q.images[0][1].exists()]
    for q in missing:
        print(f"skipping {q.name}: no data (scripts/data/fetch_*.sh)")
    out = [q for q in out if q not in missing]
    for q in out:
        q.height, q.width = read_image(q.images[0][1]).shape[:2]
    return out


def run_dir(out: Path, variant: str, seq: Sequence) -> Path:
    return out / variant / seq.name.replace("/", "_")


# --------------------------------------------------------------- running ----


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_gray(path: Path) -> np.ndarray:
    img = read_image(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def shipped_flow_config(path: Path) -> dict[str, Any]:
    """The optical-flow settings of one of Basalt's shipped configs (data/*.json)."""
    shipped = json.loads(path.read_text())["value0"]
    keys = [
        "optical_flow_type",
        "optical_flow_detection_grid_size",
        "optical_flow_max_recovered_dist2",
        "optical_flow_pattern",
        "optical_flow_max_iterations",
        "optical_flow_levels",
        "optical_flow_epipolar_error",
    ]
    return {k: shipped["config." + k] for k in keys}


def vendored_flow_args(cfg: dict[str, Any]) -> list[str]:
    """tracker_eval options running our vendored Basalt flow with these settings."""
    return [
        "--tracker", "basalt",
        "--basalt-type", str(cfg["optical_flow_type"]),
        "--grid", str(cfg["optical_flow_detection_grid_size"]),
        "--fb-dist2", str(cfg["optical_flow_max_recovered_dist2"]),
        "--pattern", str(cfg["optical_flow_pattern"]),
        "--iterations", str(cfg["optical_flow_max_iterations"]),
        "--levels", str(cfg["optical_flow_levels"]),
    ]  # fmt: skip


def basalt_config(seq: Sequence, v: Variant) -> dict[str, Any]:
    """The shipped config's optical-flow settings, plus the variant's changes.
    basalt_dense shrinks the detection grid (one feature per empty cell) so the
    number of cells is close to our 200-feature budget."""
    cfg = shipped_flow_config(seq.basalt_config)
    if "optical_flow_type" in v.config:
        cfg["optical_flow_type"] = v.config["optical_flow_type"]
    if v.config.get("dense"):
        cfg["optical_flow_detection_grid_size"] = int(round(np.sqrt(seq.width * seq.height / 200.0)))
    return cfg


def run_basalt(seq: Sequence, v: Variant, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cfg = basalt_config(seq, v)
    # The camera model only enters Basalt's stereo epipolar check, unused with one camera.
    w, h = seq.width, seq.height
    flow = basalt.OpticalFlow("pinhole", np.array([w / 2, w / 2, w / 2, h / 2]), w, h, cfg)
    times = []
    with open(d / "tracks.csv", "w") as f:
        f.write("frame,t_ns,track,u,v\n")
        for k, (t_ns, path) in enumerate(seq.images):
            img = read_gray(path)
            t0 = time.perf_counter()
            ids, uv = flow.process(t_ns, img)
            times.append(1e3 * (time.perf_counter() - t0))
            for i, (u, vv) in zip(ids, uv):
                f.write(f"{k},{t_ns},{i},{u:.4f},{vv:.4f}\n")
    with open(d / "frames.csv", "w") as f:
        f.write("frame,time_ms\n" + "".join(f"{k},{t:.3f}\n" for k, t in enumerate(times)))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))


def run_ours(seq: Sequence, v: Variant, d: Path) -> None:
    args = vendored_flow_args(basalt_config(seq, v)) if v.tracker == "vendored" else v.args
    run([tool("tracker_eval"), "tracks", *seq.tracks_args, *args, "--out", d], d.with_suffix(".log"))


def detect_corners(seq: Sequence, _v: Variant | None, path: Path) -> None:
    """Basalt AprilGrid corners of every frame: frame index, corner id, pixel."""
    frames, ids, uvs = [], [], []
    for k, (_, p) in enumerate(seq.images):
        i, uv, _ = basalt.detect_aprilgrid(read_gray(p))
        frames.append(np.full(len(i), k))
        ids.append(i)
        uvs.append(uv)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frame=np.concatenate(frames), id=np.concatenate(ids), uv=np.concatenate(uvs))


# --------------------------------------------------------------- scoring ----


@dataclass
class Tracks:
    frame: np.ndarray
    track: np.ndarray
    uv: np.ndarray
    age: np.ndarray  # frames since the track's first observation
    num_frames: int


def load_tracks(path: Path, num_frames: int) -> Tracks:
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    frame, track, uv = a[:, 0].astype(int), a[:, 2].astype(np.int64), a[:, 3:5]
    _, first_idx, inv = np.unique(track, return_index=True, return_inverse=True)
    age = frame - frame[first_idx][inv]
    return Tracks(frame, track, uv, age, num_frames)


def score_spread(t: Tracks, valid: np.ndarray) -> dict[str, float]:
    h, w = valid.shape
    size = np.array([w, h], dtype=np.float64)
    # Cells (GRID x GRID) that are >= 50 % valid.
    cells_ok = cv2.resize((valid > 0).astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA) >= 0.5
    n_ok = int(cells_ok.sum())
    cov_all, cov_stable, even, feats, stable, outside = [], [], [], [], [], []
    order = np.argsort(t.frame, kind="stable")
    bounds = np.searchsorted(t.frame[order], np.arange(t.num_frames + 1))
    wasted = new = 0
    for k in range(t.num_frames):
        idx = order[bounds[k] : bounds[k + 1]]
        uv, age = t.uv[idx], t.age[idx]
        px = np.clip(np.round(uv).astype(int), 0, [w - 1, h - 1])
        inside = valid[px[:, 1], px[:, 0]] > 0
        outside.append(float(np.mean(~inside)) if len(idx) else 0.0)
        cell = np.clip((uv / size * GRID).astype(int), 0, GRID - 1)
        counts = np.zeros((GRID, GRID))
        np.add.at(counts, (cell[inside, 1], cell[inside, 0]), 1)
        counts_s = np.zeros((GRID, GRID))
        s = inside & (age >= STABLE_AGE)
        np.add.at(counts_s, (cell[s, 1], cell[s, 0]), 1)
        cov_all.append(float(np.mean(counts[cells_ok] > 0)))
        cov_stable.append(float(np.mean(counts_s[cells_ok] > 0)))
        c = counts[cells_ok]
        p = c[c > 0] / max(c.sum(), 1)
        even.append(float(-(p * np.log(p)).sum() / np.log(n_ok)) if len(p) else 0.0)
        feats.append(len(idx))
        stable.append(int(np.sum(age >= STABLE_AGE)))
        born, old = uv[age == 0], uv[age > 0]
        if k > 0 and len(born):
            new += len(born)
            if len(old):
                dist = np.linalg.norm(born[:, None, :] - old[None, :, :], axis=2).min(axis=1)
                wasted += int(np.sum(dist <= WASTED_PX))
    return {
        "features": float(np.median(feats)),
        "stable_features": float(np.median(stable)),
        "outside_valid_pct": 100.0 * float(np.mean(outside)),
        "coverage_pct": 100.0 * float(np.median(cov_all)),
        "coverage_stable_pct": 100.0 * float(np.median(cov_stable)),
        "evenness": float(np.median(even)),
        "wasted_detections_pct": 100.0 * wasted / max(new, 1),
    }


def kaplan_meier(lengths: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """S[k] = P(track length > k) for k = 0..max; censored where alive."""
    n = int(lengths.max()) + 2
    deaths = np.bincount(lengths[~alive], minlength=n)
    exits = np.bincount(lengths, minlength=n)
    at_risk = len(lengths) - np.concatenate([[0], np.cumsum(exits)[:-1]])
    hazard = np.where(at_risk > 0, deaths / np.maximum(at_risk, 1), 0.0)
    return np.cumprod(1.0 - hazard)


def score_survival(t: Tracks) -> dict[str, float]:
    ids, inv, counts = np.unique(t.track, return_inverse=True, return_counts=True)
    last = np.zeros(len(ids), dtype=int)
    np.maximum.at(last, inv, t.frame)
    lengths = counts  # observations per track (tracks are contiguous)
    alive = last == t.num_frames - 1
    s = kaplan_meier(lengths, alive)  # s[k] = P(length > k)

    def surv(k: int) -> float:  # P(length >= k)
        return float(s[k - 1]) if k - 1 < len(s) else 0.0

    below = np.nonzero(s < 0.5)[0]
    return {
        "tracks": float(len(ids)),
        "km_median_len": float(below[0]) if len(below) else float("inf"),
        "survive5_pct": 100.0 * surv(5),
        "survive10_pct": 100.0 * surv(10),
        "survive20_pct": 100.0 * surv(20),
        "obs_weighted_len": float(np.median(lengths[inv])),
        "die_within2_pct": 100.0 * float(np.mean((lengths <= 2) & ~alive)),
    }


@dataclass
class Board:
    """Per-frame homography H (board xy -> bearing) and the convex hull of the
    detected corners in board coordinates."""

    H: dict[int, np.ndarray]
    hull: dict[int, np.ndarray]
    fit_rms: list[float]
    pitch: float  # tag size plus spacing, m


def fit_homography(xy: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """DLT on unit bearings (valid past 90 deg): rays x (H [x y 1]) = 0, with
    the board coordinates normalized."""
    m, s = xy.mean(axis=0), xy.std() * np.sqrt(2)
    T = np.array([[1 / s, 0, -m[0] / s], [0, 1 / s, -m[1] / s], [0, 0, 1]])
    X = np.c_[(xy - m) / s, np.ones(len(xy))]
    A = []
    for x, r in zip(X, rays):
        rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        A.append(np.kron(rx, x[None, :]))
    _, _, vt = np.linalg.svd(np.vstack(A))
    return vt[-1].reshape(3, 3) @ T


def project_board(H: np.ndarray, xy: np.ndarray, kb4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = (H @ np.c_[xy, np.ones(len(xy))].T).T
    uv, ok = basalt.project_points("kb4", kb4, p)
    return uv, ok


def board_frames(corners: Path, kb4: np.ndarray, board_xy: np.ndarray) -> Board:
    c = np.load(corners)
    b = Board({}, {}, [], float(board_xy[4, 0] - board_xy[0, 0]))
    for k in np.unique(c["frame"]):
        sel = c["frame"] == k
        ids, uv = c["id"][sel], c["uv"][sel]
        if len(ids) < MIN_CORNERS:
            continue
        rays, ok = basalt.unproject_points("kb4", kb4, uv)
        ids, uv, rays = ids[ok], uv[ok], rays[ok]
        xy = board_xy[ids]
        H = None
        for _ in range(2):  # fit, drop corners > 1 px, refit
            if len(ids) < MIN_CORNERS:
                break
            H = fit_homography(xy, rays)
            # H is defined up to sign; orient it so it maps onto the rays, not their opposites.
            if np.median(np.einsum("ij,ij->i", (H @ np.c_[xy, np.ones(len(xy))].T).T, rays)) < 0:
                H = -H
            pred, okp = project_board(H, xy, kb4)
            err = np.where(okp, np.linalg.norm(pred - uv, axis=1), np.inf)
            keep = err <= 1.0
            if keep.all():
                break
            ids, uv, rays, xy = ids[keep], uv[keep], rays[keep], xy[keep]
        if H is None or len(ids) < MIN_CORNERS:
            continue
        rms = float(np.sqrt(np.mean(err[err <= 1.0] ** 2)))
        b.fit_rms.append(rms)
        if rms > MAX_FIT_PX:
            continue
        b.H[int(k)] = H
        b.hull[int(k)] = cv2.convexHull(xy.astype(np.float32))
    return b


def to_board(H: np.ndarray, uv: np.ndarray, kb4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rays, ok = basalt.unproject_points("kb4", kb4, uv)
    q = (np.linalg.inv(H) @ rays.T).T
    ok &= q[:, 2] > 1e-9  # in front of the board plane's projection
    return q[:, :2] / q[:, 2:3], ok


def inside(hull: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in xy], dtype=bool)


Errors = dict[int, list[tuple[int, float, bool]]]


def drift_errors(
    t: Tracks, board: Board, kb4: np.ndarray, corners_uv: dict[int, np.ndarray]
) -> tuple[Errors, dict[int, bool]]:
    """Per track: [(age since anchor, error px, aliased)] over evaluable
    frames, and whether its anchor is within CORNER_PX of a detected corner."""
    anchor: dict[int, tuple[int, np.ndarray]] = {}  # track -> (anchor frame, board xy)
    errors: Errors = {}
    at_corner: dict[int, bool] = {}
    order = np.argsort(t.frame, kind="stable")
    bounds = np.searchsorted(t.frame[order], np.arange(t.num_frames + 1))
    for k in range(t.num_frames):
        if k not in board.H:
            continue
        idx = order[bounds[k] : bounds[k + 1]]
        if not len(idx):
            continue
        H, hull = board.H[k], board.hull[k]
        tr, uv = t.track[idx], t.uv[idx]
        known = np.array([x in anchor for x in tr])
        # Evaluate tracks already anchored.
        if known.any():
            xy = np.array([anchor[x][1] for x in tr[known]])
            vis = inside(hull, xy)
            pred, ok = project_board(H, xy, kb4)
            # Where the track actually is on the board, relative to its anchor, in tag pitches.
            q, okq = to_board(H, uv[known], kb4)
            off = (q - xy) / board.pitch
            n = np.round(off)
            aliased = okq & np.any(n != 0, axis=1) & np.all(np.abs(off - n) < 0.2, axis=1)
            for x, u, p, good, k0, al in zip(
                tr[known], uv[known], pred, ok & vis, [anchor[x][0] for x in tr[known]], aliased
            ):
                if good:
                    err = float(np.linalg.norm(u - p))
                    errors[x].append((k - k0, err, bool(al) and err > max(DRIFT_PX)))
        # Anchor the others that lie on the board.
        new = ~known
        if new.any():
            xy, ok = to_board(H, uv[new], kb4)
            ok &= inside(hull, xy)
            cuv = corners_uv.get(k, np.zeros((0, 2)))
            for x, u, q, good in zip(tr[new], uv[new], xy, ok):
                if good:
                    anchor[x] = (k, q)
                    errors[x] = []
                    at_corner[x] = bool(len(cuv)) and float(np.linalg.norm(cuv - u, axis=1).min()) <= CORNER_PX
    return errors, at_corner


def summarize_drift(errors: Errors, subset: dict[int, bool] | None, prefix: str) -> dict[str, float]:
    tracks = [e for x, e in errors.items() if e and (subset is None or subset.get(x, False))]
    out = {f"{prefix}tracks": float(len(tracks))}
    aliased = [any(al for _, _, al in e) for e in tracks]
    out[f"{prefix}aliased_pct"] = 100.0 * float(np.mean(aliased)) if tracks else float("nan")
    # Gradual drift: each track up to its first aliased jump.
    pre = [e[: next((i for i, (_, _, al) in enumerate(e) if al), len(e))] for e in tracks]
    for thr in DRIFT_PX:
        drifted = [bool(e) and max(err for _, err, _ in e) > thr for e in pre]
        out[f"{prefix}drift{thr:g}px_pct"] = 100.0 * float(np.mean(drifted)) if tracks else float("nan")
    by_age = [(a, err) for e in pre for a, err, _ in e]
    if by_age:
        ages, errs = np.array(by_age).T
        for a in AGES:
            sel = ages == a
            out[f"{prefix}err_age{a}_median"] = float(np.median(errs[sel])) if sel.any() else float("nan")
            out[f"{prefix}err_age{a}_p90"] = float(np.percentile(errs[sel], 90)) if sel.any() else float("nan")
    return out


def corner_tracks(corners: Path, num_frames: int) -> Tracks:
    """The detected corners as tracks: consecutive detections of one corner id."""
    c = np.load(corners)
    frame, cid, uv = c["frame"], c["id"].astype(np.int64), c["uv"]
    order = np.lexsort((frame, cid))
    frame, cid, uv = frame[order], cid[order], uv[order]
    brk = np.r_[True, (cid[1:] != cid[:-1]) | (frame[1:] != frame[:-1] + 1)]
    track = np.cumsum(brk)
    _, first_idx, inv = np.unique(track, return_index=True, return_inverse=True)
    return Tracks(frame, track, uv, frame - frame[first_idx][inv], num_frames)


def score_drift(
    t: Tracks, corners: Path, kb4: np.ndarray, board_xy: np.ndarray, cache: dict[Path, Board]
) -> dict[str, float]:
    if corners not in cache:
        cache[corners] = board_frames(corners, kb4, board_xy)
    board = cache[corners]
    c = np.load(corners)
    cuv = {int(k): c["uv"][c["frame"] == k] for k in np.unique(c["frame"])}
    errors, at_corner = drift_errors(t, board, kb4, cuv)
    out = {"board_frames": float(len(board.H)), "board_fit_rms_px": float(np.median(board.fit_rms))}
    out.update(summarize_drift(errors, None, "board_"))
    out.update(summarize_drift(errors, at_corner, "corner_"))
    return out


def floor_drift(
    corners: Path, num_frames: int, kb4: np.ndarray, board_xy: np.ndarray, cache: dict[Path, Board]
) -> dict[str, float]:
    """The drift scoring applied to the corners themselves: the floor set by
    the corner detector, the calibration and the homography fit."""
    return score_drift(corner_tracks(corners, num_frames), corners, kb4, board_xy, cache)


# ---------------------------------------------------------------- report ----

SPREAD_COLS = [
    "features",
    "stable_features",
    "coverage_pct",
    "coverage_stable_pct",
    "evenness",
    "wasted_detections_pct",
    "outside_valid_pct",
]
SURVIVAL_COLS = [
    "km_median_len",
    "survive5_pct",
    "survive10_pct",
    "survive20_pct",
    "obs_weighted_len",
    "die_within2_pct",
]
DRIFT_COLS = [
    "board_tracks",
    "board_aliased_pct",
    "board_drift1px_pct",
    "board_drift2px_pct",
    "board_err_age5_median",
    "board_err_age20_median",
    "corner_tracks",
    "corner_aliased_pct",
    "corner_drift1px_pct",
    "corner_drift2px_pct",
    "corner_err_age5_median",
    "corner_err_age20_median",
]


# Which direction is better, for the benchmark's degradation check
# (python/benchmark_compare.py). Counts and diagnostics without a better
# direction (features, tracks, board frames, fit RMS) are recorded, not judged.
HIGHER_BETTER = {
    "stable_features",
    "coverage_pct",
    "coverage_stable_pct",
    "evenness",
    "km_median_len",
    "survive5_pct",
    "survive10_pct",
    "survive20_pct",
    "obs_weighted_len",
}
LOWER_BETTER = {
    "wasted_detections_pct",
    "outside_valid_pct",
    "die_within2_pct",
    "ms_per_frame",
    "board_aliased_pct",
    "board_drift1px_pct",
    "board_drift2px_pct",
    "board_err_age5_median",
    "board_err_age20_median",
    "corner_aliased_pct",
    "corner_drift1px_pct",
    "corner_drift2px_pct",
    "corner_err_age5_median",
    "corner_err_age20_median",
}


def aggregate(rows: list[dict[str, Any]], variant: str, kind: str, col: str) -> float:
    """Median over the sequences of one variant and data kind (track counts summed)."""
    x = np.array([r.get(col, np.nan) for r in rows if r["variant"] == variant and r["kind"] == kind], dtype=float)
    if not np.isfinite(x).any():
        return float("nan")
    return float(np.nansum(x) if col.endswith("tracks") else np.nanmedian(x))


def table(rows: list[dict[str, Any]], cols: list[str], kinds: list[str]) -> list[str]:
    """Markdown table of per-(variant, kind) medians over sequences (track
    counts summed)."""
    lines = ["| Variant | Data | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
    variants_seen = list(dict.fromkeys(r["variant"] for r in rows))
    for kind in kinds:
        for v in variants_seen:
            if not any(r["variant"] == v and r["kind"] == kind for r in rows):
                continue
            vals = [f"{aggregate(rows, v, kind, c):.3g}" for c in cols]
            lines.append(f"| {v} | {kind} | " + " | ".join(vals) + " |")
    return lines


def metrics(rows: list[dict[str, Any]], kinds: list[str]) -> list[dict[str, Any]]:
    """Every table cell as a named metric (<variant>/<kind>/<column>)."""
    cols = SPREAD_COLS + ["ms_per_frame"] + SURVIVAL_COLS + ["board_frames", "board_fit_rms_px"] + DRIFT_COLS
    out = []
    for v in dict.fromkeys(r["variant"] for r in rows):
        for kind in kinds:
            for c in cols:
                val = aggregate(rows, v, kind, c)
                if np.isfinite(val):
                    better = "higher" if c in HIGHER_BETTER else "lower" if c in LOWER_BETTER else None
                    out.append({"name": f"{v}/{kind}/{c}", "value": val, "better": better})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="calib,room,euroc,tartan")
    ap.add_argument("--variants", default=",".join(v.name for v in variants()))
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tartanair", type=Path, default=large_data_dir() / "tartanair")
    ap.add_argument("--out", type=Path, default=REPO / "tracker_compare_out")
    ap.add_argument(
        "--corners",
        type=Path,
        help="AprilGrid corner cache (default <out>/corners); "
        "detections depend only on the images, so runs may share it",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rerun", action="store_true", help="re-run trackers even if tracks.csv exists")
    args = ap.parse_args()

    corners_dir = args.corners or args.out / "corners"
    seqs = sequences(args.sets.split(","), args.tumvi, args.euroc, args.tartanair)
    vs = [v for v in variants() if v.name in args.variants.split(",")]
    if not any(v.name == "ours" for v in vs):
        vs.insert(0, variants()[0])  # its valid-region mask is used for every variant
    board_xy = basalt.aprilgrid_corners_3d(str(BASALT_REPO / "data" / "aprilgrid_6x6.json"))[:, :2]

    jobs: list[tuple[Callable[[Sequence, Any, Path], None], Sequence, Variant | None, Path]] = []
    for seq in seqs:
        for v in vs:
            d = run_dir(args.out, v.name, seq)
            if args.rerun or not (d / "tracks.csv").exists():
                jobs.append((run_basalt if v.tracker == "basalt" else run_ours, seq, v, d))
        corners = corners_dir / (seq.name.replace("/", "_") + ".npz")
        if seq.kind == "calib" and not corners.exists():
            jobs.append((detect_corners, seq, None, corners))
    print(f"{len(seqs)} sequences, {len(vs)} variants, {len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in [ex.submit(fn, s, v, d) for fn, s, v, d in jobs]:
            fut.result()

    rows: list[dict[str, Any]] = []
    cache: dict[Path, Board] = {}
    for seq in seqs:
        valid = read_image(run_dir(args.out, "ours", seq) / "valid_mask.png", cv2.IMREAD_GRAYSCALE)
        corners = corners_dir / (seq.name.replace("/", "_") + ".npz")
        n = len(seq.images)
        if seq.kind == "calib":
            assert seq.kb4 is not None
            rows.append(
                {
                    "variant": "corners (floor)",
                    "kind": seq.kind,
                    "sequence": seq.name,
                    **floor_drift(corners, n, seq.kb4, board_xy, cache),
                }
            )
        for v in vs:
            d = run_dir(args.out, v.name, seq)
            t = load_tracks(d / "tracks.csv", n)
            ms = np.loadtxt(d / "frames.csv", delimiter=",", skiprows=1, usecols=-1, ndmin=1)
            row: dict[str, Any] = {
                "variant": v.name,
                "kind": seq.kind,
                "sequence": seq.name,
                "ms_per_frame": float(np.median(ms)),
            }
            row.update(score_spread(t, valid))
            row.update(score_survival(t))
            if seq.kind == "calib":
                assert seq.kb4 is not None
                row.update(score_drift(t, corners, seq.kb4, board_xy, cache))
            rows.append(row)
            print(
                f"{v.name:14s} {seq.name:40s} feats {row['features']:.0f} cov {row['coverage_pct']:.0f}% "
                f"km-median {row['km_median_len']:.0f} "
                + (f"drift>1px {row.get('board_drift1px_pct', float('nan')):.1f}%" if seq.kind == "calib" else ""),
                flush=True,
            )

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    kinds = [k for k in ("calib", "room", "euroc", "tartan") if k in args.sets.split(",")]
    md = [
        "# Tracker comparison: ours vs Basalt",
        "",
        "Medians over sequences (cam0 and cam1 each a sequence); track counts summed. "
        "Produced by `python/tracker_compare.py`.",
        "",
        "## How well spread are the features?",
        "",
        *table([r for r in rows if "features" in r], SPREAD_COLS + ["ms_per_frame"], kinds),
        "",
        "## How long do tracks survive?",
        "",
        *table([r for r in rows if "features" in r], SURVIVAL_COLS, kinds),
        "",
    ]
    if "calib" in kinds:
        md += [
            "## How many drift off target? (TUM-VI calib-imu, AprilGrid truth)",
            "",
            *table(
                [r for r in rows if r["kind"] == "calib"], ["board_frames", "board_fit_rms_px"] + DRIFT_COLS, ["calib"]
            ),
            "",
        ]
    (args.out / "report.md").write_text("\n".join(md))
    write_metrics(args.out, metrics(rows, kinds))
    print("\n".join(md))
    print(f"summary: {args.out / 'summary.csv'}\nreport: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
