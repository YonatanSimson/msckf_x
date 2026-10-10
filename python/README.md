# Python scripts

Analysis and display helpers. They are not part of the C++ build or its tests.

Environment: `/home/user/.venv/msckf` (Python 3.12, GUI OpenCV). Dependencies are declared in `pyproject.toml`; set up a venv with `pip install -e python/`. This also installs the `play-euroc` command.

Windows open through WSLg (`DISPLAY=:0`). If `DISPLAY` points at a Windows host X server that isn't running, `viz.py` cannot open a window.

Example: `DISPLAY=:0 play-euroc ~/data/tumvi/dataset-calib-cam1_512_16 --aprilgrid`.

| Script | Purpose |
|---|---|
| `viz.py` | `FrameSink`: shows frames in a window when OpenCV has a GUI. Otherwise it writes `viz_out/<name>.mp4` and opens it in the Windows player (WSL). |
| `play_euroc.py` | Plays a EuRoC/TUM-VI sequence with all cameras side by side. `--aprilgrid` overlays Kalibr AprilGrid tag detections. |

AprilGrid detection uses OpenCV aruco's `DICT_APRILTAG_36h11` with `markerBorderBits = 2`, because Kalibr grids print a 2-bit black border.
