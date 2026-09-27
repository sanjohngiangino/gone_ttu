#!/usr/bin/env python3
"""Dump rsl_rl checkpoint shapes / compatibility flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_to_t2.loader import inspect_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    info = inspect_checkpoint(args.checkpoint)
    payload = {
        "path": info.path,
        "iter": info.iter,
        "num_actions": info.num_actions,
        "actor_in_dim": info.actor_in_dim,
        "history_in_dim": info.history_in_dim,
        "has_history_encoder": info.has_history_encoder,
        "has_ball_estimator": info.has_ball_estimator,
        "has_region_estimator": info.has_region_estimator,
        "has_critic": info.has_critic,
        "compatible_g1_joint_pd": info.num_actions == 29,
        "keys": info.keys,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"checkpoint: {info.path}")
        print(f"iter: {info.iter}")
        print(f"num_actions: {info.num_actions}")
        print(f"actor_in_dim: {info.actor_in_dim}")
        print(f"history_in_dim: {info.history_in_dim}")
        print(f"history_encoder: {info.has_history_encoder}")
        print(f"ball_estimator: {info.has_ball_estimator}")
        print(f"region_estimator: {info.has_region_estimator}")
        print(f"critic: {info.has_critic}")
        print(f"compatible_g1_29_joint_pd: {info.num_actions == 29}")
        if info.num_actions != 29:
            print("WARNING: adapter assumes 29-DoF G1 action dim")


if __name__ == "__main__":
    main()
