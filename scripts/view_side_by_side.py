#!/usr/bin/env python3
"""Side-by-side viewer: G1 runs native Goalkeeper policy | T2 runs remapped policy.

Modes:
  --mode window   live OpenCV window (G1 left, T2 right)
  --mode video    write MP4 and exit
  --mode dual     two MuJoCo passive viewers (tile the windows yourself)

Example:
  python scripts/view_side_by_side.py --seconds 8 --mode window
  python scripts/view_side_by_side.py --mode video --out refs/g1_t2_compare.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from g1_to_t2.adapter import JointAdapter
from g1_to_t2.g1_mujoco_env import G1MujocoEnv
from g1_to_t2.loader import load_rsl_actor
from g1_to_t2.mujoco_env import T2MujocoEnv
from g1_to_t2.policy import G1PolicyWrapper, PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G1 = ROOT / "refs" / "G1_29dof" / "scene.xml"
DEFAULT_T2 = ROOT / "refs" / "T2_31dof" / "scene.xml"
DEFAULT_CKPT = ROOT / "refs" / "goalkeeper.pt"
DEFAULT_MAP = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"
DEFAULT_POLICY = ROOT / "configs" / "policies" / "goalkeeper_g1.yaml"
DEFAULT_STAND = ROOT / "configs" / "policies" / "t2_native_stand.yaml"


def load_stand(path: Path) -> tuple[np.ndarray, float, float, float]:
    data = yaml.safe_load(path.read_text())
    return (
        np.asarray(data["t2_default_joint_angles"], dtype=np.float64),
        float(data.get("height", 0.95)),
        float(data.get("kp", 250.0)),
        float(data.get("kd", 6.0)),
    )


class G1NativePolicy:
    """Run Goalkeeper actor directly on G1 (no remapping)."""

    def __init__(self, ckpt: Path, policy_yaml: Path, device: str = "cpu") -> None:
        self.cfg = PolicyConfig.from_yaml(policy_yaml)
        self.actor, _ = load_rsl_actor(
            ckpt,
            device=device,
            num_one_step_obs=self.cfg.one_step_obs_dim,
            actor_history_length=self.cfg.history_length,
        )
        self.device = torch.device(device)
        self.default = np.asarray(self.cfg.g1_default_joint_angles, dtype=np.float64)
        self._history: list[np.ndarray] = []
        self._last = np.zeros(29, dtype=np.float64)

    def reset(self) -> None:
        self._history.clear()
        self._last[:] = 0.0

    def act(
        self,
        env: G1MujocoEnv,
        ball_pos_local: np.ndarray,
    ) -> dict:
        dof_pos = env.get_q() - self.default
        step = np.concatenate(
            [
                env.get_base_ang_vel(),
                env.get_projected_gravity(),
                ball_pos_local,
                dof_pos,
                env.get_qd(),
                self._last,
            ]
        ).astype(np.float32)
        while len(self._history) < self.cfg.history_length:
            self._history.append(step.copy())
        self._history.append(step)
        if len(self._history) > self.cfg.history_length:
            self._history = self._history[-self.cfg.history_length :]
        hist = np.concatenate(self._history)[None, :]
        with torch.no_grad():
            action = (
                self.actor.act_inference(
                    torch.as_tensor(hist, device=self.device, dtype=torch.float32)
                )
                .cpu()
                .numpy()[0]
            )
        self._last = action.astype(np.float64)
        target = self.default + self.cfg.action_scale * action
        return {"action": action, "target_q": target}


def render_pair(
    r_g1: mujoco.Renderer,
    r_t2: mujoco.Renderer,
    env_g1: G1MujocoEnv,
    env_t2: T2MujocoEnv,
    cam_distance: float = 2.4,
) -> np.ndarray:
    def _cam(model: mujoco.MjModel) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = np.array([0.0, 0.0, 0.75])
        cam.distance = cam_distance
        cam.azimuth = 145.0
        cam.elevation = -18.0
        return cam

    r_g1.update_scene(env_g1.data, camera=_cam(env_g1.model))
    img_g1 = r_g1.render()
    r_t2.update_scene(env_t2.data, camera=_cam(env_t2.model))
    img_t2 = r_t2.render()

    try:
        import cv2

        img_g1 = img_g1.copy()
        img_t2 = img_t2.copy()
        cv2.putText(
            img_g1,
            "G1 native policy",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 220, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_t2,
            "T2 remapped policy",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_g1,
            f"h={env_g1.base_height():.2f}",
            (16, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_t2,
            f"h={env_t2.base_height():.2f}",
            (16, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        return np.concatenate([img_g1, img_t2], axis=1)
    except ImportError:
        return np.concatenate([img_g1, img_t2], axis=1)


def run_dual_viewers(
    env_g1: G1MujocoEnv,
    env_t2: T2MujocoEnv,
    step_fn,
    seconds: float,
    decimation: int,
) -> None:
    import mujoco.viewer

    dt = float(env_g1.model.opt.timestep)
    n = int(seconds / (dt * decimation))
    with mujoco.viewer.launch_passive(env_g1.model, env_g1.data) as v1, mujoco.viewer.launch_passive(
        env_t2.model, env_t2.data
    ) as v2:
        v1.cam.distance = 2.4
        v2.cam.distance = 2.4
        for _ in range(n):
            if not v1.is_running() or not v2.is_running():
                break
            step_fn()
            v1.sync()
            v2.sync()
            time.sleep(dt * decimation)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g1-xml", type=Path, default=DEFAULT_G1)
    p.add_argument("--t2-xml", type=Path, default=DEFAULT_T2)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--stand", type=Path, default=DEFAULT_STAND)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--mode", choices=("window", "video", "dual"), default="window")
    p.add_argument("--out", type=Path, default=ROOT / "refs" / "g1_t2_compare.mp4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--ball", type=float, nargs=3, default=(1.5, 0.0, 0.8))
    args = p.parse_args()

    if not args.g1_xml.exists():
        raise SystemExit(f"Missing G1 scene: {args.g1_xml}")
    if not args.t2_xml.exists():
        raise SystemExit(f"Missing T2 scene: {args.t2_xml}")
    if not args.ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")

    stand_q, stand_h, kp, kd = load_stand(args.stand)
    cfg = PolicyConfig.from_yaml(args.policy)
    g1_default = np.asarray(cfg.g1_default_joint_angles, dtype=np.float64)
    ball = np.asarray(args.ball, dtype=np.float64)

    env_g1 = G1MujocoEnv.load(args.g1_xml, kp=180.0, kd=4.0)
    env_t2 = T2MujocoEnv.load(args.t2_xml, kp=kp, kd=kd)

    env_g1.reset(q_g1=g1_default, height=0.85)
    env_g1.settle(g1_default, steps=250)
    env_t2.reset(q_t2=stand_q, height=stand_h)
    env_t2.settle(stand_q, steps=250)

    g1_policy = G1NativePolicy(args.ckpt, args.policy)
    g1_policy.reset()

    t2_wrapper = G1PolicyWrapper.from_checkpoint(
        args.ckpt, args.map, args.policy, device="cpu"
    )
    t2_wrapper.config.t2_default_joint_angles = stand_q.tolist()
    t2_wrapper.reset()

    # validate map loads
    _ = JointAdapter.from_yaml(args.map)

    def step_once() -> None:
        out_g1 = g1_policy.act(env_g1, ball)
        env_g1.step_pd(out_g1["target_q"], n_substeps=args.decimation)
        out_t2 = t2_wrapper.act(
            q_t2=env_t2.get_q(),
            qd_t2=env_t2.get_qd(),
            base_ang_vel=env_t2.get_base_ang_vel(),
            projected_gravity=env_t2.get_projected_gravity(),
            ball_pos_local=ball,
        )
        env_t2.step_pd(out_t2["target_q_t2"], n_substeps=args.decimation)

    if args.mode == "dual":
        print("Opening two MuJoCo viewers: left/first = G1, second = T2")
        run_dual_viewers(env_g1, env_t2, step_once, args.seconds, args.decimation)
        return

    # Offscreen render path
    r_g1 = mujoco.Renderer(env_g1.model, height=args.height, width=args.width)
    r_t2 = mujoco.Renderer(env_t2.model, height=args.height, width=args.width)

    dt = float(env_g1.model.opt.timestep)
    n_steps = int(args.seconds / (dt * args.decimation))
    frames: list[np.ndarray] = []

    use_cv = True
    try:
        import cv2
    except ImportError:
        use_cv = False
        if args.mode == "window":
            print("opencv not installed — falling back to --mode video")
            args.mode = "video"

    writer = None
    if args.mode == "video" or not use_cv:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if use_cv:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(args.out), fourcc, 1.0 / (dt * args.decimation), (args.width * 2, args.height)
            )
        else:
            # imageio fallback
            try:
                import imageio.v2 as imageio
            except ImportError:
                raise SystemExit("Install opencv-python or imageio to save video") from None

    print(f"Running {n_steps} synced steps ({args.seconds}s) mode={args.mode}")
    for i in range(n_steps):
        step_once()
        frame = render_pair(r_g1, r_t2, env_g1, env_t2)
        # MuJoCo returns RGB; OpenCV wants BGR for imshow/write
        if use_cv:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if args.mode == "window":
                cv2.imshow("G1 native | T2 remapped", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer is not None:
                writer.write(bgr)
        else:
            frames.append(frame)
        if i % 50 == 0:
            print(
                f"t={i * dt * args.decimation:.2f}s  "
                f"G1_h={env_g1.base_height():.2f}  T2_h={env_t2.base_height():.2f}"
            )

    if writer is not None:
        writer.release()
        print(f"wrote {args.out}")
    elif frames:
        import imageio.v2 as imageio

        imageio.mimsave(args.out, frames, fps=max(1, int(1.0 / (dt * args.decimation))))
        print(f"wrote {args.out}")

    if use_cv and args.mode == "window":
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
