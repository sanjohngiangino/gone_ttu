"""Goalkeeper reward terms — scales from Isaac G129Cfg.rewards.scales."""

from __future__ import annotations

from typing import Any

import torch


def reward_ee_reach(env: Any, sigma: float = 5.0) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    body_names = list(getattr(robot.data, "body_names", []))
    hand_candidates = [
        n for n in body_names if "rubber_hand" in n or "wrist_yaw" in n or "hand" in n
    ]
    if not hand_candidates:
        return torch.zeros(env.num_envs, device=env.device)
    ball_pos = ball.data.root_link_pos_w
    dists = []
    for name in hand_candidates[:4]:
        try:
            idx = body_names.index(name)
            pos = robot.data.body_link_pos_w[:, idx]
            dists.append(torch.linalg.norm(pos - ball_pos, dim=-1))
        except Exception:
            continue
    if not dists:
        return torch.zeros(env.num_envs, device=env.device)
    d = torch.stack(dists, dim=-1).min(dim=-1).values
    env.extras["reach_distance"] = d
    return torch.exp(-sigma * d)


def reward_stop_ball(env: Any, speed_th: float = 0.5) -> torch.Tensor:
    ball = env.scene["ball"]
    speed = torch.linalg.norm(ball.data.root_link_lin_vel_w, dim=-1)
    return (speed < speed_th).float()


def reward_success_intercept(env: Any, reach_th: float = 0.2) -> torch.Tensor:
    d = env.extras.get("reach_distance")
    if d is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (d < reach_th).float()


def reward_post_upright(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    grav = robot.data.projected_gravity_b
    return torch.clamp(-grav[:, 2], 0.0, 1.0)


def penalty_ang_vel_xy(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    w = robot.data.root_link_ang_vel_w[:, :2]
    return torch.sum(torch.square(w), dim=-1)


def penalty_action_rate(env: Any) -> torch.Tensor:
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1
    )


def penalty_dof_limits(env: Any) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    limits = robot.data.joint_pos_limits
    lower = limits[..., 0]
    upper = limits[..., 1]
    out = torch.clamp(lower - q, min=0.0) + torch.clamp(q - upper, min=0.0)
    return torch.sum(out, dim=-1)
