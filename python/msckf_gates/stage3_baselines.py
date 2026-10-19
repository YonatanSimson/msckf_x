"""Stage 3 baselines: OpenVINS and sqrtVINS run as unmodified binaries
(clean-room rule: their code is never read; scripts/baselines/run_baseline.sh)
monocular on the same real-data sequences as the Stage 3 gate
(msckf_gates.sequences), scored with the same tools/msckf_eval, so the plan's
"ATE within 20 % of OpenVINS mono, run as a binary" is measured here and read
by the Stage 3 real-data gate (msckf_gates.baselines.measured).

Per sequence: a ROS1 bag written once from the EuRoC layout
(python/euroc_to_rosbag.py, cached in <MSCKF_DATA_LARGE>/rosbags with the
topics the system's config expects), the system's own config for EuRoC,
TUM-VI and UZH-FPV, a generated one for LaMAria's pinhole export (the
system's EuRoC estimator settings with LaMAria's calibration, IMU noise and
gravity, and its dynamic initializer, since LaMAria never rests);
the raw fisheye export is skipped (neither system has Aria's Fisheye624).
Records ATE RMSE, rotation ATE RMSE and relative error per system and
sequence; failures are noted. Gated: every OpenVINS run (the plan's reference)
produces a scored trajectory; sqrtVINS failures with its own configs are
reported only. Runs take about real time per
sequence, so --workers runs are parallel on separate ROS masters.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import yaml

from euroc_layout import rigid
from euroc_to_rosbag import write_bag
from msckf_gates import sequences
from msckf_gates.common import REPO, GateReport, data_dir, fnum, gate_out, large_data_dir, read_csv, run, tool
from msckf_gates.sequences import Sequence
from msckf_gates.stage3_real import on_disk, present

SYSTEMS = {"openvins": "open_vins", "sqrtvins": "sqrtVINS"}  # system -> source folder under non_commercial
EUROC_TOPICS = (["/cam0/image_raw"], "/imu0")
UZH_TOPICS = (["/snappy_cam/stereo_l"], "/snappy_imu")
# Sequence name prefix -> (config folder of the system, camera topics, IMU topic).
CONFIGS: list[tuple[str, str, tuple[list[str], str]]] = [
    ("euroc_", "euroc_mav", EUROC_TOPICS),
    ("tumvi_", "tum_vi", EUROC_TOPICS),
    ("uzh_indoor_forward_", "uzhfpv_indoor", UZH_TOPICS),
    ("uzh_indoor_45_", "uzhfpv_indoor_45", UZH_TOPICS),
    ("uzh_outdoor_forward_", "uzhfpv_outdoor", UZH_TOPICS),
    ("uzh_outdoor_45_", "uzhfpv_outdoor_45", UZH_TOPICS),
    ("lamaria_R_", "lamaria", EUROC_TOPICS),
]


# Seconds of bag skipped before the systems start, from the comment in their
# serial.launch ("v1-2: 0, mh1: 40, mh2: 35, mh3: 17.5, mh4-5: 15"): the
# Machine Hall sequences open with a handheld IMU excitation and a rest on
# the ground that their static initializer does not handle. Ours starts at
# 0.5 s everywhere, which is a (small) advantage for them.
BAG_START_S: dict[str, float] = {
    "euroc_MH_01_easy": 40.0,
    "euroc_MH_02_easy": 35.0,
    "euroc_MH_03_medium": 17.5,
    "euroc_MH_04_difficult": 15.0,
    "euroc_MH_05_difficult": 15.0,
}


def non_commercial() -> Path:
    return Path(os.environ.get("MSCKF_NON_COMMERCIAL", Path.home() / "work" / "non_commercial"))


def config_of(seq: Sequence) -> tuple[str, tuple[list[str], str]]:
    for prefix, config, topics in CONFIGS:
        if seq.name.startswith(prefix):
            return config, topics
    raise ValueError(f"{seq.name}: no baseline config")


def euroc_dir(seq: Sequence) -> Path:
    return Path(seq.cam[seq.cam.index("--euroc") + 1])


def bag_path(seq: Sequence) -> Path:
    return large_data_dir() / "rosbags" / f"{seq.name}.bag"


def ensure_bag(seq: Sequence) -> Path:
    bag = bag_path(seq)
    if not bag.exists():
        _, (cam_topics, imu_topic) = config_of(seq)
        write_bag(euroc_dir(seq), bag, [seq.cam[seq.cam.index("--cam") + 1]], cam_topics, imu_topic)
    return bag


class PlainDumper(yaml.SafeDumper):
    """YAML as OpenCV's FileStorage reads it: no anchors or aliases, and
    sequences indented under their key."""

    def ignore_aliases(self, data):
        return True

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def opencv_yaml(doc: dict) -> str:
    """OpenCV FileStorage YAML, as the baseline configs are written."""
    return "%YAML:1.0\n\n" + yaml.dump(doc, Dumper=PlainDumper, sort_keys=False, default_flow_style=None)


def lamaria_camchain(calib: dict, seq: Sequence) -> dict:
    c = calib["cam0"]
    T_imu_cam = rigid(c["T_b_s"]["qvec"], c["T_b_s"]["tvec"])
    return {"cam0": {
        "T_imu_cam": [[float(x) for x in row] for row in T_imu_cam],
        "cam_overlaps": [],
        "camera_model": "pinhole",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
        "distortion_model": "radtan",
        "intrinsics": [float(x) for x in c["params"]],
        "resolution": [c["resolution"]["width"], c["resolution"]["height"]],
        "rostopic": config_of(seq)[1][0][0],
    }}  # fmt: skip


def lamaria_imuchain(calib: dict, seq: Sequence) -> dict:
    imu = calib["imu0"]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {"imu0": {
        "T_i_b": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "accelerometer_noise_density": float(imu["acc_noise_density"]),
        "accelerometer_random_walk": float(imu["acc_bias_random_walk_sigma"]),
        "gyroscope_noise_density": float(imu["gyro_noise_density"]),
        "gyroscope_random_walk": float(imu["gyro_bias_random_walk_sigma"]),
        "rostopic": config_of(seq)[1][1],
        "time_offset": 0.0,
        "update_rate": float(imu["imu_rate"]),
        "model": "kalibr",
        "Tw": identity, "R_IMUtoGYRO": identity, "Ta": identity, "R_IMUtoACC": identity,
        "Tg": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }}  # fmt: skip


def write_lamaria_config(system: str, seq: Sequence, out: Path) -> Path:
    """The system's EuRoC estimator settings with the sequence's own calibration and gravity."""
    calib_json = euroc_dir(seq) / "pinhole_calibrations" / f"{euroc_dir(seq).name}.json"
    calib = json.loads(calib_json.read_text())
    d = out / "config" / system / seq.name
    d.mkdir(parents=True, exist_ok=True)
    template = (non_commercial() / SYSTEMS[system] / "config" / "euroc_mav" / "estimator_config.yaml").read_text()
    # LaMAria is handheld from the first frame, so the static initializer never
    # sees a rest ("platform moving too much"): use the dynamic one.
    settings = {"gravity_mag": calib["imu0"]["gravity_magnitude"], "max_cameras": 1, "use_stereo": "false",
                "init_dyn_use": "true"}  # fmt: skip
    estimator = template
    for key, value in settings.items():
        estimator = re.sub(rf"^{key}: \S+", f"{key}: {value}", estimator, flags=re.M)
    (d / "estimator_config.yaml").write_text(estimator)
    (d / "kalibr_imucam_chain.yaml").write_text(opencv_yaml(lamaria_camchain(calib, seq)))
    (d / "kalibr_imu_chain.yaml").write_text(opencv_yaml(lamaria_imuchain(calib, seq)))
    return d


def config_arg(system: str, seq: Sequence, out: Path) -> str:
    config, _ = config_of(seq)
    return str(write_lamaria_config(system, seq, out)) if config == "lamaria" else config


def run_and_score(system: str, seq: Sequence, out: Path, ports: "Queue[int]", reuse: bool) -> str:
    """Runs the system on the sequence's bag and msckf_eval; returns the failure, empty on success.
    With reuse, a sequence already scored in out is kept (development; benchmarks rerun all).
    ports holds the free ROS master ports; one is taken for the run and given back."""
    d = out / system / seq.name
    if reuse and (d / "eval" / "metrics.csv").exists():
        return ""
    try:
        bag = ensure_bag(seq)
        port = ports.get()
        try:
            run([REPO / "scripts" / "baselines" / "run_baseline.sh", system, config_arg(system, seq, out), bag, d,
                 BAG_START_S.get(seq.name, 0.0)],
                log=d.parent / f"{seq.name}.log", env={"ROS_MASTER_URI": f"http://localhost:{port}"})  # fmt: skip
        finally:
            ports.put(port)
        run([tool("msckf_eval"), "--gt", seq.gt, "--gt-format", "euroc", "--est", d / "est.txt", "--name", seq.name,
             "--out", d / "eval"], log=d.parent / f"{seq.name}_eval.log")  # fmt: skip
    except RuntimeError as e:
        return str(e)
    return ""


def record(report: GateReport, system: str, seq: Sequence, out: Path, error: str) -> None:
    label = f"{system} {seq.name}"
    if error:
        report.note(f"{label}: FAILED ({error})")
        return
    m = read_csv(out / system / seq.name / "eval" / "metrics.csv")[0]
    ate, rot, re_ = fnum(m["ate_rmse_m"]), fnum(m["ate_rot_rmse_deg"]), fnum(m["re_trans_mean_pct"])
    report.record(f"{label}: ATE RMSE", ate, "lower", "m")
    report.record(f"{label}: rotation ATE RMSE", rot, "lower", "deg")
    report.record(f"{label}: relative translation error, mean", re_, "lower", "%")
    report.note(f"{label}: ATE {ate:.3f} m over {fnum(m['length_m']):.0f} m, rotation {rot:.2f} deg, RE {re_:.2f} %")


def supported(seqs: list[Sequence], report: GateReport) -> list[Sequence]:
    skipped = [s.name for s in seqs if s.name.startswith("lamaria_fisheye_")]
    if skipped:
        report.note(f"skipped (no Fisheye624 in the baselines): {', '.join(skipped)}")
    return [s for s in seqs if s.name not in skipped]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", type=Path, default=large_data_dir() / "euroc")
    ap.add_argument("--tumvi", type=Path, default=data_dir() / "tumvi")
    ap.add_argument("--uzh-fpv", type=Path, default=large_data_dir() / "uzh_fpv")
    ap.add_argument("--lamaria", type=Path, default=large_data_dir() / "lamaria" / "training")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reuse", action="store_true", help="keep sequences already scored in --out")
    ap.add_argument("--out", default="stage3_baselines", help="gate_out subdirectory")
    args = ap.parse_args()

    out = gate_out(args.out)
    report = GateReport("Stage 3 baselines: OpenVINS and sqrtVINS mono, run as binaries")
    lamaria = present([args.lamaria / s for s in sequences.LAMARIA], report, "LaMAria")
    uzh = present([args.uzh_fpv / f"{n}_snapdragon_with_gt" for n in sequences.UZH_FPV], report, "UZH-FPV")
    seqs = supported([*sequences.euroc(args.euroc), *on_disk(sequences.tumvi(args.tumvi), report, "TUM-VI"),
                      *sequences.converted("uzh", uzh),
                      *sequences.converted("lamaria", lamaria)], report)  # fmt: skip
    jobs = [(system, seq) for system in args.systems.split(",") for seq in seqs]
    ports: "Queue[int]" = Queue()
    for port in range(11411, 11411 + args.workers):
        ports.put(port)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        errors = list(ex.map(lambda j: run_and_score(j[0], j[1], out, ports, args.reuse), jobs))
    for (system, seq), error in zip(jobs, errors):
        record(report, system, seq, out, error)
    openvins_failed = sum(1 for (system, _), e in zip(jobs, errors) if e and system == "openvins")
    report.check("OpenVINS runs failed (the plan's reference baseline)", openvins_failed, "<=", 0)
    report.record("baseline runs failed, all systems", float(sum(1 for e in errors if e)), "lower")
    return report.finish(out)


if __name__ == "__main__":
    raise SystemExit(main())
