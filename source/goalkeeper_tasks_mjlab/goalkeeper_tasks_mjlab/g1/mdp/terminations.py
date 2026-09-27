"""Terminations for Goalkeeper."""

from __future__ import annotations

from typing import Any

import torch


def root_height_below(env: Any, minimum_height: float = 0.35) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_link_pos_w[:, 2] < minimum_height


def bad_orientation(env: Any, limit_angle: float = 1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    grav = robot.data.projected_gravity_b
    return torch.linalg.norm(grav[:, :2], dim=-1) > limit_angle
