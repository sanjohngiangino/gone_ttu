"""G1 Goalkeeper environment config for mjlab.

Port target: InternRobotics Humanoid-Goalkeeper (IsaacGym G129Cfg).
Follows SoccerLab mjlab patterns (register + ManagerBasedRlEnvCfg).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from goalkeeper_tasks_mjlab.g1.mdp import (
    bad_orientation,
    ball_pos_b,
    ball_vel_b,
    goal_region_id,
    penalty_action_rate,
    penalty_ang_vel_xy,
    penalty_dof_limits,
    reach_distance,
    reset_ball_toward_region,
    reward_ee_reach,
    reward_post_upright,
    reward_stop_ball,
    reward_success_intercept,
    root_height_below,
)

_ROOT = pathlib.Path(__file__).resolve().parents[4]  # spqr_zoff/
_BALL_MJCF = _ROOT / "data" / "assets" / "ball" / "soccer_ball.xml"


def _ball_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_BALL_MJCF)),
        init_state=EntityCfg.InitialStateCfg(pos=(2.0, 0.0, 0.115)),
    )


def g1_goalkeeper_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build Goalkeeper env. `play=True` reduces envs / noise for visualization."""

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": get_g1_robot_cfg(),
            "ball": _ball_cfg(),
        },
    )

    # --- observations (actor ≈ Isaac one-step; history via policy / stacked frames) ---
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func="mjlab.envs.mdp:base_ang_vel",
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2) if not play else None,
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func="mjlab.envs.mdp:projected_gravity",
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05) if not play else None,
        ),
        "ball_pos": ObservationTermCfg(
            func=ball_pos_b,
            noise=Unoise(n_min=-0.08, n_max=0.08) if not play else None,
            scale=0.3,
        ),
        "joint_pos": ObservationTermCfg(
            func="mjlab.envs.mdp:joint_pos_rel",
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01) if not play else None,
        ),
        "joint_vel": ObservationTermCfg(
            func="mjlab.envs.mdp:joint_vel_rel",
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-1.5, n_max=1.5) if not play else None,
            scale=0.05,
        ),
        "actions": ObservationTermCfg(func="mjlab.envs.mdp:last_action"),
    }

    critic_terms = {
        **actor_terms,
        "ball_vel": ObservationTermCfg(func=ball_vel_b, scale=0.2),
        "goal_region": ObservationTermCfg(func=goal_region_id),
        "reach": ObservationTermCfg(func=reach_distance),
        "base_lin_vel": ObservationTermCfg(
            func="mjlab.envs.mdp:base_lin_vel",
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=2.0,
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            scale=G1_ACTION_SCALE if isinstance(G1_ACTION_SCALE, float) else 0.25,
            use_default_offset=True,
        ),
    }
    # Force Goalkeeper action_scale parity if zoo scale is a dict
    if not isinstance(getattr(actions["joint_pos"], "scale", 0.25), (int, float)):
        actions["joint_pos"].scale = 0.25

    rewards = {
        "eereach": RewardTermCfg(func=reward_ee_reach, weight=10.0),
        "success": RewardTermCfg(func=reward_success_intercept, weight=5.0),
        "stopball": RewardTermCfg(func=reward_stop_ball, weight=100.0),
        "post_upright": RewardTermCfg(func=reward_post_upright, weight=3.0),
        "ang_vel_xy": RewardTermCfg(func=penalty_ang_vel_xy, weight=-0.1),
        "action_rate": RewardTermCfg(func=penalty_action_rate, weight=-0.1),
        "dof_limits": RewardTermCfg(func=penalty_dof_limits, weight=-3.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func="mjlab.envs.mdp:time_out", time_out=True),
        "fallen": TerminationTermCfg(
            func=root_height_below,
            params={"minimum_height": 0.35},
        ),
        "bad_ori": TerminationTermCfg(func=bad_orientation, params={"limit_angle": 1.2}),
    }

    events = {
        "reset_ball": EventTermCfg(
            func=reset_ball_toward_region,
            mode="reset",
        ),
    }

    sim = SimulationCfg(
        nconmax=128,
        njmax=700,
        mujoco=MujocoCfg(timestep=0.005, iterations=10),
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=sim,
        episode_length_s=3.0,
        decimation=4,
        viewer=ViewerConfig(lookat=(0.0, 0.0, 0.8), distance=3.0),
    )
    # num envs set by train/play CLI in mjlab; play hint via attribute if supported
    if play and hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
        cfg.scene.num_envs = 1
    return cfg
