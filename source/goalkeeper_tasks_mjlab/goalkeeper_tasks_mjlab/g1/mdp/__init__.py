"""MDP exports for Goalkeeper G1."""

from goalkeeper_tasks_mjlab.g1.mdp.events import REGION_RANGES, reset_ball_toward_region
from goalkeeper_tasks_mjlab.g1.mdp.observations import (
    ball_pos_b,
    ball_vel_b,
    goal_region_id,
    reach_distance,
)
from goalkeeper_tasks_mjlab.g1.mdp.rewards import (
    penalty_action_rate,
    penalty_ang_vel_xy,
    penalty_dof_limits,
    reward_ee_reach,
    reward_post_upright,
    reward_stop_ball,
    reward_success_intercept,
)
from goalkeeper_tasks_mjlab.g1.mdp.terminations import bad_orientation, root_height_below

__all__ = [
    "REGION_RANGES",
    "ball_pos_b",
    "ball_vel_b",
    "goal_region_id",
    "reach_distance",
    "reward_ee_reach",
    "reward_stop_ball",
    "reward_success_intercept",
    "reward_post_upright",
    "penalty_ang_vel_xy",
    "penalty_action_rate",
    "penalty_dof_limits",
    "reset_ball_toward_region",
    "root_height_below",
    "bad_orientation",
]
