# Python scripts

Analysis and display helpers. They are not part of the C++ build or its tests.

Environment: `/home/user/.venv/msckf` (Python 3.12, GUI OpenCV). Dependencies are declared in `pyproject.toml`; set up a venv with `pip install -e python/`. This also installs the `play-euroc` command.

Windows open through WSLg (`DISPLAY=:0`). If `DISPLAY` points at a Windows host X server that isn't running, `viz.py` cannot open a window.

Example: `DISPLAY=:0 play-euroc ~/data/tumvi/dataset-calib-cam1_512_16 --aprilgrid`.

| Script | Purpose |
|---|---|
| `viz.py` | `FrameSink`: shows frames in a window when OpenCV has a GUI. Otherwise it writes `viz_out/<name>.mp4` and opens it in the Windows player (WSL). |
| `play_euroc.py` | Plays a EuRoC/TUM-VI sequence with all cameras side by side. `--aprilgrid` overlays Kalibr AprilGrid tag detections. |
| `lamaria_to_euroc.py`, `lamaria_fisheye_to_euroc.py`, `uzh_fpv_to_euroc.py`, `kitti_raw_to_euroc.py` | Converters to the EuRoC layout the tools read (`mav0/`, `gt_imu.csv`); shared writers in `euroc_layout.py`. The LaMAria fisheye export reads the VRS with projectaria_tools. |
| `aria_fisheye_fixture.py` | Writes `tests/data/fisheye624_aria.csv`, the projectaria_tools reference the Fisheye624 cross-check test compares against. |
| `euroc_to_rosbag.py` | ROS1 bag (mono camera + IMU) from a EuRoC layout, for the baseline binaries; needs no ROS. |
| `msckf_gates/` | Gate checkers (`gate-stage0-reprojection`, `gate-stage1a-static-init`, or `python -m msckf_gates.<gate>`). Each one runs the C++ tools, recomputes the gate metrics from their CSV output, and writes `gate_out/<gate>/report.md`. Run them all with `scripts/run_gates.sh`. |

`play_euroc.py --aprilgrid` uses OpenCV aruco (`DICT_APRILTAG_36h11`, `markerBorderBits = 2`) only as a quick visual. It misses many tags, because each tag's border touches the grid's corner squares. The measurement detector is the C++ one in `msckf_calib`.
