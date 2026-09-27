#!/usr/bin/env python3
"""Calibrate T2 stand pose + optional joint sign flips.

Uses configs/policies/t2_native_stand.yaml by default (G1-mapped crouch is not
statically stable on T2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from g1_to_t2.joints import T2_JOINT_NAMES
from g1_to_t2.mujoco_env import T2MujocoEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "refs" / "T2_31dof" / "T2_31dof.xml"
DEFAULT_STAND = ROOT / "configs" / "policies" / "t2_native_stand.yaml"
DEFAULT_MAP = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"
OUT_DEFAULTS = ROOT / "configs" / "policies" / "t2_stand_defaults.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--stand", type=Path, default=DEFAULT_STAND)
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--kd", type=float, default=None)
    p.add_argument(
        "--flip",
        nargs="*",
        default=[],
        help="T2 joint names whose map sign should be flipped",
    )
    p.add_argument("--write-map", type=Path, default=None)
    args = p.parse_args()

    stand = yaml.safe_load(args.stand.read_text())
    default_t2 = np.asarray(stand["t2_default_joint_angles"], dtype=np.float64)
    if default_t2.shape[0] != 31:
        raise SystemExit("t2_default_joint_angles must have length 31")
    kp = float(args.kp if args.kp is not None else stand.get("kp", 180.0))
    kd = float(args.kd if args.kd is not None else stand.get("kd", 4.0))
    height = float(stand.get("height", 0.92))

    env = T2MujocoEnv.load(args.xml, kp=kp, kd=kd)
    env.reset(q_t2=default_t2, height=height)
    env.settle(default_t2, steps=400)

    dt = float(env.model.opt.timestep)
    n = int(args.seconds / dt)
    heights = []
    uprights = []
    for _ in range(n):
        env.step_pd(default_t2, n_substeps=1)
        heights.append(env.base_height())
        uprights.append(env.upright())

    mean_h = float(np.mean(heights[-min(100, len(heights)) :]))
    mean_up = float(np.mean(uprights[-min(100, len(uprights)) :]))
    final_q = env.get_q()
    err = final_q - default_t2

    payload = {
        "source": "calibrate_signs.py native stand hold",
        "stand_config": str(args.stand),
        "height_init": height,
        "kp": kp,
        "kd": kd,
        "mean_height_last": mean_h,
        "mean_upright_last": mean_up,
        "final_upright": float(env.upright()),
        "t2_default_joint_angles": default_t2.tolist(),
        "t2_joint_names": T2_JOINT_NAMES,
        "tracking_abs_err_mean": float(np.mean(np.abs(err))),
        "stable": bool(mean_h > 0.6 and mean_up > 0.8),
    }
    OUT_DEFAULTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DEFAULTS.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(
        f"stand mean_height={mean_h:.3f} mean_upright={mean_up:.3f} "
        f"stable={payload['stable']}"
    )
    print(f"tracking_abs_err_mean={payload['tracking_abs_err_mean']:.4f}")
    print(f"wrote {OUT_DEFAULTS}")

    if args.flip:
        data = yaml.safe_load(args.map.read_text())
        flip_set = set(args.flip)
        for e in data["entries"]:
            if e["t2"] in flip_set:
                e["sign"] = -float(e.get("sign", 1.0))
                print(f"flipped sign for {e['t2']} -> {e['sign']}")
        dest = args.write_map or args.map
        dest.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"wrote map {dest}")

    if not payload["stable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
