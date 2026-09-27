"""Load rsl_rl-style ActorCritic checkpoints (Goalkeeper / generic MLP)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    table = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "selu": nn.SELU(),
        "lrelu": nn.LeakyReLU(),
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]


def _mlp(in_dim: int, hidden: list[int], out_dim: int, act: str) -> nn.Sequential:
    """Linear-Act stacks for hidden layers, final Linear without activation."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden[0]), get_activation(act)]
    for i in range(len(hidden) - 1):
        layers.append(nn.Linear(hidden[i], hidden[i + 1]))
        layers.append(get_activation(act))
    layers.append(nn.Linear(hidden[-1], out_dim))
    return nn.Sequential(*layers)


@dataclass
class CheckpointInfo:
    path: str
    keys: list[str]
    num_actions: int
    has_history_encoder: bool
    has_ball_estimator: bool
    has_region_estimator: bool
    has_critic: bool
    actor_in_dim: int | None
    history_in_dim: int | None
    iter: Any = None


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Expected dict with model_state_dict, got {type(ckpt)}")
    msd = ckpt["model_state_dict"]
    keys = list(msd.keys())
    num_actions = int(msd["std"].shape[0]) if "std" in msd else int(msd["actor.6.weight"].shape[0])
    actor_in = int(msd["actor.0.weight"].shape[1]) if "actor.0.weight" in msd else None
    hist_in = (
        int(msd["history_encoder.0.weight"].shape[1])
        if "history_encoder.0.weight" in msd
        else None
    )
    return CheckpointInfo(
        path=str(path),
        keys=keys,
        num_actions=num_actions,
        has_history_encoder="history_encoder.0.weight" in msd,
        has_ball_estimator="ball_estimator.0.weight" in msd,
        has_region_estimator="region_estimator.0.weight" in msd,
        has_critic="critic.0.weight" in msd,
        actor_in_dim=actor_in,
        history_in_dim=hist_in,
        iter=ckpt.get("iter"),
    )


class GoalkeeperActor(nn.Module):
    """ActorCritic subset matching Humanoid-Goalkeeper act_inference."""

    def __init__(
        self,
        num_one_step_obs: int = 96,
        actor_history_length: int = 10,
        num_actions: int = 29,
        actor_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        history_latent_dim: int = 16,
        estimate_ball_dim: int = 6,
        num_regions: int = 6,
    ) -> None:
        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [512, 256, 256]
        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.num_actions = num_actions
        self.history_latent_dim = history_latent_dim
        self.estimate_ball_dim = estimate_ball_dim
        self.num_regions = num_regions

        hist_in = num_one_step_obs * actor_history_length
        self.history_encoder = nn.Sequential(
            nn.Linear(hist_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, history_latent_dim),
        )
        self.ball_estimator = nn.Sequential(
            nn.Linear(hist_in, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, estimate_ball_dim),
        )
        self.region_estimator = nn.Sequential(
            nn.Linear(hist_in, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_regions),
        )
        actor_in = num_one_step_obs + history_latent_dim + estimate_ball_dim + 1
        self.actor = _mlp(actor_in, actor_hidden_dims, num_actions, activation)
        self.std = nn.Parameter(torch.ones(num_actions))

    def act_inference(self, obs_history: torch.Tensor) -> torch.Tensor:
        history_latent = self.history_encoder(obs_history)
        estimate_ball = self.ball_estimator(obs_history)
        estimate_region = self.region_estimator(obs_history)
        actor_input = torch.cat(
            (
                obs_history[:, -self.num_one_step_obs :],
                history_latent,
                estimate_ball,
                torch.argmax(estimate_region, dim=-1, keepdim=True).float(),
            ),
            dim=-1,
        )
        return self.actor(actor_input)


class GenericActor(nn.Module):
    """Simple MLP actor: obs -> actions (no history encoder)."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 29,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 256]
        self.actor = _mlp(obs_dim, hidden_dims, num_actions, activation)
        self.std = nn.Parameter(torch.ones(num_actions))
        self.num_actions = num_actions

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def _infer_hidden_dims(msd: dict[str, torch.Tensor], prefix: str = "actor") -> list[int]:
    dims: list[int] = []
    i = 0
    while f"{prefix}.{i}.weight" in msd:
        w = msd[f"{prefix}.{i}.weight"]
        dims.append(int(w.shape[0]))
        i += 2  # weight, activation, weight, ...
    # last dim is num_actions — hidden are all but last
    if len(dims) < 2:
        raise ValueError(f"Could not infer hidden dims from {prefix}.*")
    return dims[:-1]


def load_rsl_actor(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    num_one_step_obs: int | None = None,
    actor_history_length: int | None = None,
) -> tuple[nn.Module, CheckpointInfo]:
    """Load a G1 policy module ready for act_inference."""
    path = Path(path)
    info = inspect_checkpoint(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    msd = ckpt["model_state_dict"]

    if info.has_history_encoder:
        hist_in = info.history_in_dim or 960
        hist_len = actor_history_length or 10
        one_step = num_one_step_obs or (hist_in // hist_len)
        hidden = _infer_hidden_dims(msd, "actor")
        model = GoalkeeperActor(
            num_one_step_obs=one_step,
            actor_history_length=hist_len,
            num_actions=info.num_actions,
            actor_hidden_dims=hidden,
        )
    else:
        if info.actor_in_dim is None:
            raise ValueError("Cannot load actor without actor.0.weight")
        hidden = _infer_hidden_dims(msd, "actor")
        model = GenericActor(
            obs_dim=info.actor_in_dim,
            num_actions=info.num_actions,
            hidden_dims=hidden,
        )

    # Load matching keys only (skip critic if present unused)
    own = model.state_dict()
    filtered = {k: v for k, v in msd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in filtered]
    if missing:
        raise RuntimeError(f"Missing keys when loading actor: {missing}")
    model.load_state_dict(filtered, strict=True)
    model.to(device)
    model.eval()
    return model, info
