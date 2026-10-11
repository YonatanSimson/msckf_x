"""Frame display that works with or without an OpenCV GUI backend.

Usage:
    with FrameSink("name", fps=20, out_dir="viz_out") as sink:
        for frame in frames:
            if not sink.show(frame):
                break

With a GUI build of OpenCV, frames go to a window (space or a left click =
pause / resume, n = step, q/Esc = quit). With a headless build (pip opencv-python-headless, as in
indoor3d_venv), frames are written to <out_dir>/<name>.mp4, which is opened
in the Windows default player under WSL.
"""
import os
import shutil
import subprocess

import cv2

# The opencv-python wheel bundles Qt without fonts, and its import points
# QT_QPA_FONTDIR at that empty cv2/qt/fonts, so Qt warns on every window
# ("QFontDatabase: Cannot find font directory"). Qt reads the variable when the
# first window opens, so overriding it after importing cv2 is enough.
SYSTEM_FONTS = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(SYSTEM_FONTS) and not os.path.isdir(os.environ.get("QT_QPA_FONTDIR", "")):
    os.environ["QT_QPA_FONTDIR"] = SYSTEM_FONTS


def has_gui():
    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def open_external(path):
    """Open a file with the desktop's default application (WSL: Windows)."""
    path = os.path.abspath(path)
    if shutil.which("explorer.exe") and shutil.which("wslpath"):
        win = subprocess.run(["wslpath", "-w", path], capture_output=True,
                             text=True, check=True).stdout.strip()
        subprocess.run(["explorer.exe", win], check=False)  # returns 1 on success
    elif shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", path])
    else:
        print(f"open {path} manually")


class FrameSink:
    def __init__(self, name, fps=20.0, out_dir="viz_out", force_video=False, open_video=True):
        self.name, self.fps = name, fps
        self.gui = not force_video and has_gui()
        self.path = None if self.gui else os.path.join(out_dir, name + ".mp4")
        self.open_video = open_video
        self.writer = None
        self.paused = False
        self.placed = False
        self.clicked = False
        if self.gui:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(name, self._on_mouse)
        else:
            os.makedirs(out_dir, exist_ok=True)

    def show(self, frame):
        """Display or record one BGR frame. Returns False when the user quits."""
        if not self.gui:
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              self.fps, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError(f"cannot write {self.path}")
            self.writer.write(frame)
            return True
        cv2.imshow(self.name, frame)
        if not self.placed:
            # A resizable window can open off-screen or tiny (seen under WSLg):
            # size it to the frame and put it on screen.
            h, w = frame.shape[:2]
            cv2.resizeWindow(self.name, w, h)
            cv2.moveWindow(self.name, 40, 40)
            self.placed = True
        while True:
            # Poll while paused too, so a mouse click (which waitKey does not
            # return) can resume.
            key = cv2.waitKey(50 if self.paused else max(1, int(1000 / self.fps))) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key == ord(" ") or self.clicked:
                self.paused = not self.paused
                self.clicked = False
            if not self.paused or key == ord("n"):
                return True

    def _on_mouse(self, event, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True

    def close(self):
        if self.gui:
            cv2.destroyAllWindows()
        elif self.writer is not None:
            self.writer.release()
            print(f"wrote {self.path}")
            if self.open_video:
                open_external(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
