"""Goalkeeper MDP observation terms (mjlab).

Target actor one-step layout (Isaac G129Cfg parity):
  ang_vel(3) + projected_gravity(3) + ball_pos_b(3)
  + dof_pos(29) + dof_vel(29) + last_action(29)  = 96
"""

from __future__ import annotations

from typing import Any

import torch


def ball_pos_b(env: Any, robot_cfg: Any = None, ball_cfg: Any = None) -> torch.Tensor:
    """Ball position in robot root frame."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    root_pos = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    ball_pos = ball.data.root_link_pos_w
    from mjlab.utils.math import quat_apply_inverse

    return quat_apply_inverse(root_quat, ball_pos - root_pos)


def ball_vel_b(env: Any, robot_cfg: Any = None, ball_cfg: Any = None) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    root_quat = robot.data.root_link_quat_w
    ball_vel = ball.data.root_link_lin_vel_w
    from mjlab.utils.math import quat_apply_inverse

    return quat_apply_inverse(root_quat, ball_vel)


def goal_region_id(env: Any) -> torch.Tensor:
    return (
        env.extras.get(
            "goal_region",
            torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        )
        .float()
        .unsqueeze(-1)
    )


def reach_distance(env: Any) -> torch.Tensor:
    if "reach_distance" in env.extras:
        return env.extras["reach_distance"].unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, device=env.device)
