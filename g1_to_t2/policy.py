"""Build G1-layout observations from T2 state and run transferred policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from g1_to_t2.adapter import JointAdapter
from g1_to_t2.loader import load_rsl_actor


@dataclass
class PolicyConfig:
    """Layout for one-step G1 actor observation (Goalkeeper-compatible default)."""

    num_actions: int = 29
    num_dofs: int = 29
    history_length: int = 10
    action_scale: float = 0.25
    # one-step: ang_vel(3) + gravity(3) + ball(3) + dof_pos(29) + dof_vel(29) + actions(29) = 96
    ball_obs_dim: int = 3
    include_base_ang_vel: bool = True
    include_projected_gravity: bool = True
    g1_default_joint_angles: list[float] = field(default_factory=list)
    t2_default_joint_angles: list[float] = field(default_factory=list)

    @property
    def one_step_obs_dim(self) -> int:
        n = 0
        if self.include_base_ang_vel:
            n += 3
        if self.include_projected_gravity:
            n += 3
        n += self.ball_obs_dim
        n += self.num_dofs * 2
        n += self.num_actions
        return n

    @property
    def history_obs_dim(self) -> int:
        return self.one_step_obs_dim * self.history_length

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PolicyConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class G1PolicyWrapper:
    """T2 low-state -> pack to G1 obs history -> G1 actor -> unpack action to T2."""

    def __init__(
        self,
        actor: torch.nn.Module,
        adapter: JointAdapter,
        config: PolicyConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor = actor
        self.adapter = adapter
        self.config = config
        self.device = torch.device(device)
        self._history: deque[np.ndarray] = deque(maxlen=config.history_length)
        self._last_action_g1 = np.zeros(config.num_actions, dtype=np.float64)
        self.g1_default = np.array(
            config.g1_default_joint_angles
            if config.g1_default_joint_angles
            else [0.0] * config.num_dofs,
            dtype=np.float64,
        )
        if self.g1_default.shape[0] != config.num_dofs:
            raise ValueError("g1_default_joint_angles length mismatch")

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        map_yaml: str | Path,
        policy_yaml: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> "G1PolicyWrapper":
        adapter = JointAdapter.from_yaml(map_yaml)
        config = (
            PolicyConfig.from_yaml(policy_yaml)
            if policy_yaml
            else PolicyConfig()
        )
        actor, _info = load_rsl_actor(
            ckpt_path,
            device=device,
            num_one_step_obs=config.one_step_obs_dim,
            actor_history_length=config.history_length,
        )
        return cls(actor=actor, adapter=adapter, config=config, device=device)

    def reset(self) -> None:
        self._history.clear()
        self._last_action_g1[:] = 0.0

    def build_one_step_obs(
        self,
        *,
        q_t2: np.ndarray,
        qd_t2: np.ndarray,
        base_ang_vel: np.ndarray | None = None,
        projected_gravity: np.ndarray | None = None,
        ball_pos_local: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.config
        q_g1 = self.adapter.pack_q_t2_to_g1(np.asarray(q_t2, dtype=np.float64))
        qd_g1 = self.adapter.pack_q_t2_to_g1(np.asarray(qd_t2, dtype=np.float64))
        # Goalkeeper-style: dof_pos relative to default
        dof_pos = q_g1 - self.g1_default
        parts: list[np.ndarray] = []
        if cfg.include_base_ang_vel:
            parts.append(np.asarray(base_ang_vel if base_ang_vel is not None else np.zeros(3)))
        if cfg.include_projected_gravity:
            parts.append(
                np.asarray(
                    projected_gravity
                    if projected_gravity is not None
                    else np.array([0.0, 0.0, -1.0])
                )
            )
        ball = (
            np.asarray(ball_pos_local, dtype=np.float64)
            if ball_pos_local is not None
            else np.zeros(cfg.ball_obs_dim)
        )
        if ball.shape[0] != cfg.ball_obs_dim:
            raise ValueError(f"ball_pos_local must have dim {cfg.ball_obs_dim}")
        parts.append(ball)
        parts.append(dof_pos)
        parts.append(qd_g1)
        parts.append(self._last_action_g1.copy())
        obs = np.concatenate(parts, axis=0)
        if obs.shape[0] != cfg.one_step_obs_dim:
            raise RuntimeError(
                f"one-step obs dim {obs.shape[0]} != expected {cfg.one_step_obs_dim}"
            )
        return obs.astype(np.float32)

    def act(
        self,
        *,
        q_t2: np.ndarray,
        qd_t2: np.ndarray,
        base_ang_vel: np.ndarray | None = None,
        projected_gravity: np.ndarray | None = None,
        ball_pos_local: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Return T2 action (normalized) and PD targets."""
        step = self.build_one_step_obs(
            q_t2=q_t2,
            qd_t2=qd_t2,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            ball_pos_local=ball_pos_local,
        )
        while len(self._history) < self.config.history_length:
            self._history.append(step.copy())
        self._history.append(step)

        hist = np.concatenate(list(self._history), axis=0)[None, :]  # (1, H*obs)
        with torch.no_grad():
            t = torch.as_tensor(hist, device=self.device, dtype=torch.float32)
            action_g1 = self.actor.act_inference(t).cpu().numpy()[0]

        self._last_action_g1 = action_g1.astype(np.float64)
        action_t2 = self.adapter.unpack_q_g1_to_t2(action_g1)
        # PD target = default_t2 + scale * action  (if defaults provided)
        if self.config.t2_default_joint_angles:
            default_t2 = np.asarray(self.config.t2_default_joint_angles, dtype=np.float64)
            target_t2 = default_t2 + self.config.action_scale * action_t2
        else:
            # map G1 defaults to T2 then apply
            default_t2 = self.adapter.unpack_q_g1_to_t2(self.g1_default)
            target_t2 = default_t2 + self.config.action_scale * action_t2

        return {
            "action_g1": action_g1,
            "action_t2": action_t2,
            "target_q_t2": target_t2,
            "obs_step": step,
        }
