"""Reset / domain events for Goalkeeper (ball launch by region)."""

from __future__ import annotations

from typing import Any

import torch

# Isaac G129Cfg.commands.ranges_* → (h_min, h_max, w_min, w_max)
REGION_RANGES = [
    (0.4, 1.2, 0.2, 1.2),
    (0.4, 1.2, -1.2, -0.2),
    (1.2, 1.6, 0.0, 1.0),
    (1.2, 1.6, -1.0, 0.0),
    (0.1, 0.3, 0.2, 1.2),
    (0.1, 0.3, -1.2, -0.2),
]


def reset_ball_toward_region(
    env: Any,
    env_ids: torch.Tensor | None = None,
    approach_dist: float = 3.0,
    flight_time: float = 0.6,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    regions = torch.randint(0, len(REGION_RANGES), (n,), device=env.device)
    if "goal_region" not in env.extras:
        env.extras["goal_region"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    env.extras["goal_region"][env_ids] = regions

    root = robot.data.root_link_pos_w[env_ids]
    ball_pos = root.clone()
    ball_pos[:, 0] += approach_dist
    ball_pos[:, 2] = 1.0

    land = torch.zeros(n, 3, device=env.device)
    for i, r in enumerate(regions.tolist()):
        h0, h1, w0, w1 = REGION_RANGES[r]
        land[i, 1] = (w0 + w1) * 0.5
        land[i, 2] = (h0 + h1) * 0.5
        land[i, 0] = root[i, 0] + 0.3
    vel = (land - ball_pos) / flight_time

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :3] = ball_pos
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    twist = torch.zeros(n, 6, device=env.device)
    twist[:, :3] = vel
    ball.write_root_link_velocity_to_sim(twist, env_ids=env_ids)
