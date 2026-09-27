#!/usr/bin/env python3
"""Smoke-test: run transferred G1 policy on T2 in MuJoCo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from g1_to_t2.adapter import JointAdapter
from g1_to_t2.mujoco_env import T2MujocoEnv
from g1_to_t2.policy import G1PolicyWrapper, PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "refs" / "T2_31dof" / "T2_31dof.xml"
DEFAULT_STAND = ROOT / "configs" / "policies" / "t2_native_stand.yaml"
DEFAULT_CKPT = ROOT / "refs" / "goalkeeper.pt"
DEFAULT_MAP = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"
DEFAULT_POLICY = ROOT / "configs" / "policies" / "goalkeeper_g1.yaml"


def load_stand(path: Path) -> tuple[np.ndarray, float, float, float]:
    import yaml

    data = yaml.safe_load(path.read_text())
    q = np.asarray(data["t2_default_joint_angles"], dtype=np.float64)
    return (
        q,
        float(data.get("height", 0.92)),
        float(data.get("kp", 180.0)),
        float(data.get("kd", 4.0)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--stand", type=Path, default=DEFAULT_STAND)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--kd", type=float, default=None)
    p.add_argument("--hold-default", action="store_true",
                   help="Ignore policy; hold native T2 stand pose")
    p.add_argument("--require-stand", action="store_true",
                   help="Exit non-zero if robot falls (default: warn only)")
    p.add_argument("--log", type=Path, default=ROOT / "refs" / "smoke_log.csv")
    args = p.parse_args()

    if not args.xml.exists():
        raise SystemExit(f"Missing MJCF: {args.xml}")
    if not args.ckpt.exists() and not args.hold_default:
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")

    stand_q, stand_h, stand_kp, stand_kd = load_stand(args.stand)
    kp = args.kp if args.kp is not None else stand_kp
    kd = args.kd if args.kd is not None else stand_kd

    env = T2MujocoEnv.load(args.xml, kp=kp, kd=kd)
    adapter = JointAdapter.from_yaml(args.map)
    cfg = PolicyConfig.from_yaml(args.policy)

    # Init from stable T2 stand; G1 defaults still used inside wrapper obs.
    env.reset(q_t2=stand_q, height=stand_h)
    env.settle(stand_q, steps=400)

    wrapper = None
    if not args.hold_default:
        # Prefer T2 stand as action default offsets when present
        cfg.t2_default_joint_angles = stand_q.tolist()
        wrapper = G1PolicyWrapper.from_checkpoint(
            args.ckpt, args.map, args.policy, device="cpu"
        )
        # override defaults after load
        wrapper.config.t2_default_joint_angles = stand_q.tolist()
        wrapper.reset()

    dt = float(env.model.opt.timestep)
    n_steps = int(args.seconds / (dt * args.decimation))
    fell = False
    rows: list[list[float]] = []

    for i in range(n_steps):
        if wrapper is None:
            target = stand_q
            action_norm = 0.0
        else:
            out = wrapper.act(
                q_t2=env.get_q(),
                qd_t2=env.get_qd(),
                base_ang_vel=env.get_base_ang_vel(),
                projected_gravity=env.get_projected_gravity(),
                ball_pos_local=np.array([1.5, 0.0, 0.8]),
            )
            target = out["target_q_t2"]
            action_norm = float(np.linalg.norm(out["action_t2"]))

        env.step_pd(target, n_substeps=args.decimation)
        h = env.base_height()
        up = env.upright()
        rows.append([i * dt * args.decimation, h, up, action_norm])
        if h < 0.35 or up < 0.2:
            fell = True
            break

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "base_height", "upright", "action_norm"])
        w.writerows(rows)

    status = "FELL" if fell else "ok"
    print(
        f"status={status} steps={len(rows)} "
        f"final_height={rows[-1][1]:.3f} upright={rows[-1][2]:.3f}"
    )
    print(f"log -> {args.log}")
    # unused but keeps adapter import meaningful for map validation
    _ = adapter.summary()
    if fell and args.require_stand:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
