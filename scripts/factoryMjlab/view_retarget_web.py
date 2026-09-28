#!/usr/bin/env python3
"""Browser GUI: G1 | T2 retarget motions (RDP-safe, EGL).

Pre-renders all clips (one EGL context at a time), then serves them in the browser.

  http://127.0.0.1:8090/

Keys (click the page first): 1-6 motion, [ ] prev/next, Space pause, R restart
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>G1 | T2 retarget motions</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
 #wrap{display:flex;flex-direction:column;align-items:center;gap:8px;padding:12px}
 img{max-width:98vw;background:#000;border:1px solid #333}
 kbd{background:#222;border:1px solid #555;border-radius:4px;padding:1px 6px}
 .bar{opacity:.9;font-size:14px}
</style></head><body>
<div id="wrap">
 <div class="bar" id="status">loading…</div>
 <img id="frame" alt="G1 | T2"/>
 <div class="bar">
  <kbd>1</kbd>–<kbd>6</kbd> motion · <kbd>[</kbd>/<kbd>]</kbd> ·
  <kbd>Space</kbd> pause · <kbd>R</kbd> restart · click page for keys
 </div>
</div>
<script>
const img=document.getElementById('frame');
const status=document.getElementById('status');
function refresh(){
  img.src='/frame.jpg?t='+Date.now();
  fetch('/state').then(r=>r.json()).then(s=>{
    status.textContent=`[${s.idx+1}/${s.n}] ${s.name}  t=${s.frame}/${s.T}  ${s.paused?'PAUSE':'PLAY'}`;
  }).catch(()=>{});
}
setInterval(refresh,40); refresh();
window.addEventListener('keydown',e=>{
  const map={' ':'space','r':'r','R':'r','ArrowLeft':'prev','ArrowRight':'next',
             '[':'prev',']':'next','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6'};
  const k=map[e.key]; if(!k) return; e.preventDefault();
  fetch('/key?k='+encodeURIComponent(k));
});
</script></body></html>
"""


def _render_pair(g1: RobotPlayer, t2: RobotPlayer) -> np.ndarray:
    """Render G1 then T2 with exclusive EGL (close G1 context before T2)."""
    left = g1.renderer.render().copy()
    # Tear down G1 GL so T2 can make_current.
    g1.renderer.close()
    right = t2.renderer.render().copy()
    t2.renderer.close()
    # Recreate for next frame / next clip.
    g1.renderer = mujoco.Renderer(g1.model, height=g1.renderer.height if False else left.shape[0], width=left.shape[1])
    # height/width from image
    h, w = left.shape[:2]
    g1.renderer = mujoco.Renderer(g1.model, height=h, width=w)
    t2.renderer = mujoco.Renderer(t2.model, height=h, width=w)
    return np.concatenate([left, right], axis=1)


def prerender_all(
    bank: MotionBank,
    g1: RobotPlayer,
    t2: RobotPlayer,
    ground: bool,
) -> list[list[bytes]]:
    clips_jpeg: list[list[bytes]] = []
    h, w = g1.renderer.height, g1.renderer.width
    for i, name in enumerate(bank.names):
        clip = bank.clips[i]
        print(f"  prerender [{i+1}/{len(bank)}] {name} ({clip['T']} frames)…", flush=True)
        zg = bank.compute_z_shift(g1, clip, "g1") if ground else 0.0
        zt = bank.compute_z_shift(t2, clip, "t2") if ground else 0.0

        # Pass 1: all G1 frames (single EGL context)
        lefts: list[np.ndarray] = []
        for t in range(clip["T"]):
            g1.set_pose(clip["base_pos"][t], clip["base_quat"][t], clip["q_g1"][t], zg)
            g1.renderer.update_scene(g1.data, camera=g1.cam)
            lefts.append(g1.renderer.render().copy())
        g1.renderer.close()

        # Pass 2: all T2 frames
        rights: list[np.ndarray] = []
        # ensure T2 renderer alive (may have been ok); recreate if closed
        if getattr(t2.renderer, "_gl_context", None) is None:
            t2.renderer = mujoco.Renderer(t2.model, height=h, width=w)
        for t in range(clip["T"]):
            t2.set_pose(clip["base_pos"][t], clip["base_quat"][t], clip["q_t2"][t], zt)
            t2.renderer.update_scene(t2.data, camera=t2.cam)
            rights.append(t2.renderer.render().copy())
        t2.renderer.close()

        # Recreate both for next clip
        g1.renderer = mujoco.Renderer(g1.model, height=h, width=w)
        t2.renderer = mujoco.Renderer(t2.model, height=h, width=w)

        frames: list[bytes] = []
        for t, (left, right) in enumerate(zip(lefts, rights)):
            canvas = np.concatenate([left, right], axis=1)
            cv2.putText(
                canvas,
                f"G1  |  {name}  t={t}/{clip['T']}",
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
                (w + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )
            ok, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 75],
            )
            frames.append(buf.tobytes() if ok else b"")
        clips_jpeg.append(frames)
        print(f"    done {name}", flush=True)
    return clips_jpeg


class Player:
    def __init__(self, bank: MotionBank, frames: list[list[bytes]], fps: float):
        self.bank = bank
        self.frames = frames
        self.dt = 1.0 / fps
        self.idx = 0
        self.frame = 0
        self.paused = False
        self.lock = threading.Lock()

    def handle_key(self, k: str) -> None:
        with self.lock:
            if k == "space":
                self.paused = not self.paused
            elif k == "r":
                self.frame = 0
            elif k == "prev":
                self.idx = (self.idx - 1) % len(self.frames)
                self.frame = 0
            elif k == "next":
                self.idx = (self.idx + 1) % len(self.frames)
                self.frame = 0
            elif k in "123456":
                self.idx = (int(k) - 1) % len(self.frames)
                self.frame = 0

    def jpeg(self) -> bytes:
        with self.lock:
            seq = self.frames[self.idx]
            return seq[self.frame % len(seq)]

    def state(self) -> dict:
        with self.lock:
            return {
                "idx": self.idx,
                "n": len(self.frames),
                "name": self.bank.names[self.idx],
                "frame": self.frame,
                "T": len(self.frames[self.idx]),
                "paused": self.paused,
            }

    def tick(self) -> None:
        with self.lock:
            if not self.paused:
                self.frame = (self.frame + 1) % len(self.frames[self.idx])


def make_handler(player: Player):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path)
            if path.path in ("/", "/index.html"):
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path.path == "/frame.jpg":
                body = player.jpeg()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path.path == "/state":
                body = json.dumps(player.state()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path.path == "/key":
                k = (parse_qs(path.query).get("k") or [""])[0]
                if k:
                    player.handle_key(k)
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-ground", action="store_true")
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()

    adapter = JointAdapter.from_yaml(DEFAULT_MAP)
    bank = MotionBank(DEFAULT_MOTION_DIR, adapter)
    print(f"Loaded {len(bank)} motions — prerendering for browser…", flush=True)

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

    frames = prerender_all(bank, g1, t2, ground=not args.no_ground)
    player = Player(bank, frames, fps=args.fps)
    player.idx = args.start % len(frames)

    def loop():
        nxt = time.perf_counter()
        while True:
            player.tick()
            nxt += player.dt
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    threading.Thread(target=loop, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(player))
    url = f"http://{args.host}:{args.port}/"
    print(f"\n>>> Open in YOUR browser: {url}\n", flush=True)
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()


if __name__ == "__main__":
    main()
