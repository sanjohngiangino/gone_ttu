#!/usr/bin/env python3
"""G1|T2 retarget viewer with mouse orbit.

EGL render runs in a subprocess; parent only does GLFW blit + input
(avoids EGL/GLX BadAccess in one process).

Mouse: LMB orbit, wheel zoom, RMB pan
Keys: 1-6 / [ ] motion, Space pause, R restart, C reset cam, Esc quit
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = Path(__file__).resolve()


def _run_worker(args: argparse.Namespace) -> None:
    """EGL-only render loop: JSON request lines on stdin → raw RGB on stdout."""
    # Keep this process free of glfw/GLX imports.
    os.environ["MUJOCO_GL"] = "egl"
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_SCRIPT.parent))

    import cv2

    from g1_to_t2.adapter import JointAdapter
    from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES
    from retarget_motion_core import (
        DEFAULT_G1_XML,
        DEFAULT_MAP,
        DEFAULT_MOTION_DIR,
        DEFAULT_T2_XML,
        MotionBank,
        RobotPlayer,
    )

    adapter = JointAdapter.from_yaml(DEFAULT_MAP)
    bank = MotionBank(DEFAULT_MOTION_DIR, adapter)
    g1 = RobotPlayer(
        DEFAULT_G1_XML,
        G1_JOINT_NAMES,
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        "G1",
        args.width,
        args.height,
    )
    t2 = RobotPlayer(
        DEFAULT_T2_XML,
        T2_JOINT_NAMES,
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        "T2",
        args.width,
        args.height,
    )
    ground = not args.no_ground
    z_g1 = [
        bank.compute_z_shift(g1, c, "g1") if ground else 0.0 for c in bank.clips
    ]
    z_t2 = [
        bank.compute_z_shift(t2, c, "t2") if ground else 0.0 for c in bank.clips
    ]

    meta = {
        "ok": True,
        "names": bank.names,
        "lengths": [c["T"] for c in bank.clips],
        "width": args.width * 2,
        "height": args.height,
    }
    sys.stdout.buffer.write((json.dumps(meta) + "\n").encode())
    sys.stdout.buffer.flush()

    for line in sys.stdin.buffer:
        req = json.loads(line.decode())
        if req.get("cmd") == "quit":
            break
        idx = int(req["idx"]) % len(bank)
        clip = bank.clips[idx]
        t = int(req["frame"]) % clip["T"]
        look_offset = np.asarray(req.get("look_offset", [0, 0, 0]), dtype=np.float64)
        lookat = (
            np.array(
                [
                    float(clip["base_pos"][t, 0]),
                    float(clip["base_pos"][t, 1]),
                    0.55,
                ],
                dtype=np.float64,
            )
            + look_offset
        )
        for p in (g1, t2):
            p.cam.azimuth = float(req["azimuth"])
            p.cam.elevation = float(req["elevation"])
            p.cam.distance = float(req["distance"])
            p.cam.lookat[:] = lookat

        g1.set_pose(
            clip["base_pos"][t], clip["base_quat"][t], clip["q_g1"][t], z_g1[idx]
        )
        left = g1.render().copy()
        t2.set_pose(
            clip["base_pos"][t], clip["base_quat"][t], clip["q_t2"][t], z_t2[idx]
        )
        right = t2.render().copy()
        canvas = np.concatenate([left, right], axis=1)
        name = bank.names[idx]
        cv2.putText(
            canvas,
            f"G1 | {name}  t={t}/{clip['T']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "T2 remapped",
            (args.width + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "LMB orbit | wheel zoom | RMB pan | C reset",
            (12, canvas.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        payload = np.ascontiguousarray(canvas, dtype=np.uint8).tobytes()
        sys.stdout.buffer.write(struct.pack("<I", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


class Viewer:
    def __init__(
        self,
        proc: subprocess.Popen,
        names: list[str],
        lengths: list[int],
        width: int,
        height: int,
        fps: float,
        start: int,
    ) -> None:
        import glfw
        from OpenGL.GL import (
            GL_COLOR_BUFFER_BIT,
            glClear,
            glClearColor,
        )

        self.glfw = glfw
        self.GL_COLOR_BUFFER_BIT = GL_COLOR_BUFFER_BIT
        self.glClear = glClear

        self.proc = proc
        self.names = names
        self.lengths = lengths
        self.width = width
        self.height = height
        self.dt = 1.0 / fps
        self.idx = start % len(names)
        self.frame = 0
        self.paused = False
        self.running = True
        self._dirty = True

        self.azimuth = 140.0
        self.elevation = -15.0
        self.distance = 2.7
        self.look_offset = np.zeros(3, dtype=np.float64)
        self._drag = None
        self._rgb = np.zeros((height, width, 3), dtype=np.uint8)

        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, "G1 | T2 retarget", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("create_window failed")
        glfw.make_context_current(self.window)
        glfw.set_window_pos(self.window, 160, 120)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glClearColor(0.05, 0.05, 0.07, 1.0)
        self._title()
        print(
            f"Window up on DISPLAY={os.environ.get('DISPLAY')} — "
            "LMB orbit, wheel zoom, RMB pan",
            flush=True,
        )

    def _title(self) -> None:
        st = "PAUSE" if self.paused else "PLAY"
        self.glfw.set_window_title(
            self.window,
            f"G1 | T2  [{self.idx+1}/{len(self.names)}] {self.names[self.idx]}  {st}",
        )

    def _request_frame(self) -> None:
        req = {
            "cmd": "frame",
            "idx": self.idx,
            "frame": self.frame,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "look_offset": self.look_offset.tolist(),
        }
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write((json.dumps(req) + "\n").encode())
        self.proc.stdin.flush()
        hdr = self.proc.stdout.read(4)
        if len(hdr) < 4:
            raise RuntimeError("render worker died")
        (nbytes,) = struct.unpack("<I", hdr)
        payload = self.proc.stdout.read(nbytes)
        if len(payload) != nbytes:
            raise RuntimeError("short frame from worker")
        self._rgb = (
            np.frombuffer(payload, dtype=np.uint8)
            .reshape(self.height, self.width, 3)
            .copy()
        )

    def _on_key(self, window, key, scancode, action, mods):  # noqa: ARG002
        glfw = self.glfw
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            self.running = False
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
            self._title()
        elif key == glfw.KEY_R:
            self.frame = 0
            self._dirty = True
        elif key == glfw.KEY_C:
            self.azimuth, self.elevation, self.distance = 140.0, -15.0, 2.7
            self.look_offset[:] = 0.0
            self._dirty = True
        elif key in (glfw.KEY_LEFT, glfw.KEY_LEFT_BRACKET):
            self.idx = (self.idx - 1) % len(self.names)
            self.frame = 0
            self._title()
            self._dirty = True
        elif key in (glfw.KEY_RIGHT, glfw.KEY_RIGHT_BRACKET):
            self.idx = (self.idx + 1) % len(self.names)
            self.frame = 0
            self._title()
            self._dirty = True
        elif glfw.KEY_1 <= key <= glfw.KEY_6:
            self.idx = (key - glfw.KEY_1) % len(self.names)
            self.frame = 0
            self._title()
            self._dirty = True

    def _on_mouse_button(self, window, button, action, mods):  # noqa: ARG002
        glfw = self.glfw
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._drag = ("orbit", x, y)
            elif button in (glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE):
                self._drag = ("pan", x, y)
        elif action == glfw.RELEASE:
            self._drag = None

    def _on_cursor(self, window, x, y):  # noqa: ARG002
        if self._drag is None:
            return
        mode, px, py = self._drag
        dx, dy = x - px, y - py
        self._drag = (mode, x, y)
        if mode == "orbit":
            self.azimuth = (self.azimuth + dx * 0.35) % 360.0
            self.elevation = float(np.clip(self.elevation - dy * 0.25, -89.0, 89.0))
        else:
            scale = self.distance * 0.0018
            az = np.deg2rad(self.azimuth)
            right = np.array([np.cos(az), np.sin(az), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            self.look_offset = self.look_offset - right * dx * scale + up * dy * scale
        self._dirty = True

    def _on_scroll(self, window, _xoff, yoff):  # noqa: ARG002
        self.distance = float(np.clip(self.distance * (0.9 ** yoff), 0.6, 12.0))
        self._dirty = True

    def run(self) -> None:
        from OpenGL.GL import (
            GL_RGB,
            GL_UNSIGNED_BYTE,
            glDrawPixels,
            glViewport,
            glWindowPos2i,
        )

        glfw = self.glfw
        next_t = time.perf_counter()
        self._request_frame()
        while self.running and not glfw.window_should_close(self.window):
            glfw.poll_events()
            now = time.perf_counter()
            if not self.paused and now >= next_t:
                self.frame = (self.frame + 1) % self.lengths[self.idx]
                next_t = now + self.dt
                self._dirty = True
            if self._dirty:
                self._request_frame()
                self._dirty = False
            self.glClear(self.GL_COLOR_BUFFER_BIT)
            glViewport(0, 0, self.width, self.height)
            flipped = np.ascontiguousarray(np.flipud(self._rgb))
            glWindowPos2i(0, 0)
            glDrawPixels(
                flipped.shape[1], flipped.shape[0], GL_RGB, GL_UNSIGNED_BYTE, flipped
            )
            glfw.swap_buffers(self.window)
        try:
            if self.proc.stdin:
                self.proc.stdin.write(b'{"cmd":"quit"}\n')
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            glfw.destroy_window(self.window)
        finally:
            glfw.terminate()


def _run_viewer(args: argparse.Namespace) -> None:
    if not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY vuoto. export DISPLAY=:1")

    cmd = [
        sys.executable,
        "-u",
        str(_SCRIPT),
        "--worker",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.no_ground:
        cmd.append("--no-ground")
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    # Avoid inheriting a GLX-oriented PyOpenGL platform from the parent shell.
    env.pop("PYOPENGL_PLATFORM", None)

    print("Starting EGL render worker…", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        cwd=str(_REPO),
        env=env,
    )
    assert proc.stdout is not None
    # Worker may print diagnostics on stderr; first stdout line must be JSON meta.
    meta = None
    while True:
        meta_line = proc.stdout.readline()
        if not meta_line:
            raise RuntimeError("render worker failed to start")
        text = meta_line.decode().strip()
        if not text:
            continue
        try:
            meta = json.loads(text)
            break
        except json.JSONDecodeError:
            print(f"[worker] {text}", flush=True)
            continue
    if not meta.get("ok"):
        raise RuntimeError(f"worker error: {meta}")
    names = list(meta["names"])
    lengths = list(meta["lengths"])
    print("Motions:", ", ".join(f"{i+1}:{n}" for i, n in enumerate(names)), flush=True)
    Viewer(
        proc,
        names,
        lengths,
        width=int(meta["width"]),
        height=int(meta["height"]),
        fps=args.fps,
        start=args.start,
    ).run()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-ground", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        _run_worker(args)
    else:
        _run_viewer(args)


if __name__ == "__main__":
    main()
