#!/usr/bin/env python3
"""Fine-tune a native T2 31-DoF standing policy.

1) BC toward near-zero actions around the native stand pose (and optional
   soft mix with remapped G1 teacher actions)
2) Short PPO-style fine-tune with height/upright reward
3) Save refs/t2_stability.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from g1_to_t2.mujoco_env import T2MujocoEnv
from g1_to_t2.policy import G1PolicyWrapper, PolicyConfig
from g1_to_t2.t2_actor import T2StabilityActor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "refs" / "T2_31dof" / "T2_31dof.xml"
DEFAULT_CKPT = ROOT / "refs" / "goalkeeper.pt"
DEFAULT_MAP = ROOT / "configs" / "maps" / "g1_29_to_t2_31.yaml"
DEFAULT_POLICY = ROOT / "configs" / "policies" / "goalkeeper_g1.yaml"
DEFAULT_STAND = ROOT / "configs" / "policies" / "t2_native_stand.yaml"
OUT_CKPT = ROOT / "refs" / "t2_stability.pt"

OBS_DIM = 3 + 3 + 31 + 31 + 31


def load_stand(path: Path) -> tuple[np.ndarray, float, float, float]:
    data = yaml.safe_load(path.read_text())
    return (
        np.asarray(data["t2_default_joint_angles"], dtype=np.float64),
        float(data.get("height", 0.95)),
        float(data.get("kp", 250.0)),
        float(data.get("kd", 6.0)),
    )


def build_obs(env: T2MujocoEnv, last_action: np.ndarray, default_q: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            env.get_base_ang_vel(),
            env.get_projected_gravity(),
            env.get_q() - default_q,
            env.get_qd(),
            last_action,
        ]
    ).astype(np.float32)


def reward_fn(env: T2MujocoEnv, action: np.ndarray) -> float:
    h = env.base_height()
    up = env.upright()
    alive = 0.0 if (h > 0.5 and up > 0.7) else -5.0
    return float(
        1.0 * up
        + 0.5 * np.clip(h, 0.0, 1.2)
        + alive
        - 0.01 * np.sum(np.square(env.get_qd()))
        - 0.01 * np.sum(np.square(action))
    )


def collect_bc(
    env: T2MujocoEnv,
    default_t2: np.ndarray,
    height: float,
    episodes: int,
    horizon: int,
    decimation: int,
    teacher: G1PolicyWrapper | None,
    teacher_mix: float,
) -> tuple[np.ndarray, np.ndarray]:
    obs_list, act_list = [], []
    for _ in range(episodes):
        noise = np.random.uniform(-0.04, 0.04, size=31)
        q0 = default_t2 + noise
        env.reset(q_t2=q0, height=height)
        env.settle(default_t2, steps=150)
        if teacher is not None:
            teacher.reset()
        last = np.zeros(31, dtype=np.float32)
        for _t in range(horizon):
            o = build_obs(env, last, default_t2)
            # Primary target: hold stand (zero residual action)
            a = np.zeros(31, dtype=np.float32)
            if teacher is not None and teacher_mix > 0:
                out = teacher.act(
                    q_t2=env.get_q(),
                    qd_t2=env.get_qd(),
                    base_ang_vel=env.get_base_ang_vel(),
                    projected_gravity=env.get_projected_gravity(),
                    ball_pos_local=np.array([1.5, 0.0, 0.8]),
                )
                a = (1.0 - teacher_mix) * a + teacher_mix * np.clip(
                    out["action_t2"], -2.0, 2.0
                ).astype(np.float32)
            obs_list.append(o)
            act_list.append(a)
            target = default_t2 + 0.25 * a
            env.step_pd(target, n_substeps=decimation)
            last = a
            if env.base_height() < 0.4:
                break
    return np.stack(obs_list), np.stack(act_list)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--stand", type=Path, default=DEFAULT_STAND)
    p.add_argument("--out", type=Path, default=OUT_CKPT)
    p.add_argument("--bc-epochs", type=int, default=40)
    p.add_argument("--bc-episodes", type=int, default=15)
    p.add_argument("--rl-iters", type=int, default=50)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--teacher-mix", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    default_t2, stand_h, kp, kd = load_stand(args.stand)
    cfg = PolicyConfig.from_yaml(args.policy)
    cfg.t2_default_joint_angles = default_t2.tolist()
    action_scale = float(cfg.action_scale)

    env = T2MujocoEnv.load(args.xml, kp=kp, kd=kd)
    teacher = None
    if args.teacher_mix > 0 and args.ckpt.exists():
        teacher = G1PolicyWrapper.from_checkpoint(
            args.ckpt, args.map, args.policy, device=args.device
        )
        teacher.config.t2_default_joint_angles = default_t2.tolist()

    print("Collecting BC data (stand hold + optional teacher mix)...")
    obs_np, act_np = collect_bc(
        env,
        default_t2,
        stand_h,
        args.bc_episodes,
        args.horizon,
        args.decimation,
        teacher,
        args.teacher_mix,
    )
    print(f"BC samples: {len(obs_np)}")

    student = T2StabilityActor(OBS_DIM, num_actions=31).to(device)
    with torch.no_grad():
        student.log_std.fill_(-1.0)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    obs_t = torch.as_tensor(obs_np, device=device)
    act_t = torch.as_tensor(act_np, device=device)
    for ep in range(args.bc_epochs):
        pred = student.act_inference(obs_t)
        loss = F.mse_loss(pred, act_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"BC epoch {ep+1}/{args.bc_epochs} loss={loss.item():.6f}")

    print("On-policy stability fine-tune...")
    for it in range(args.rl_iters):
        noise = np.random.uniform(-0.03, 0.03, 31)
        env.reset(q_t2=default_t2 + noise, height=stand_h)
        env.settle(default_t2, steps=100)
        last = np.zeros(31, dtype=np.float32)
        traj_obs, traj_act, traj_logp, traj_rew, traj_val = [], [], [], [], []
        for _t in range(args.horizon):
            o = build_obs(env, last, default_t2)
            ot = torch.as_tensor(o[None, :], device=device)
            with torch.no_grad():
                a, logp, v = student.act(ot)
            a_np = np.clip(a.cpu().numpy()[0], -3.0, 3.0)
            target = default_t2 + action_scale * a_np
            env.step_pd(target, n_substeps=args.decimation)
            r = reward_fn(env, a_np)
            traj_obs.append(o)
            traj_act.append(a_np)
            traj_logp.append(float(logp.cpu()))
            traj_rew.append(r)
            traj_val.append(float(v.cpu()))
            last = a_np.astype(np.float32)
            if env.base_height() < 0.4 or env.upright() < 0.4:
                traj_rew[-1] -= 5.0
                break

        rewards = np.array(traj_rew, dtype=np.float32)
        values = np.array(traj_val, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        adv = 0.0
        for t in reversed(range(len(rewards))):
            nxt = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + 0.99 * nxt - values[t]
            adv = delta + 0.99 * 0.95 * adv
            advantages[t] = adv
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ot = torch.as_tensor(np.stack(traj_obs), device=device)
        at = torch.as_tensor(np.stack(traj_act), device=device)
        adv_t = torch.as_tensor(advantages, device=device)
        ret_t = torch.as_tensor(returns, device=device)
        old_logp = torch.as_tensor(traj_logp, device=device)

        logp, entropy, value = student.evaluate(ot, at)
        ratio = torch.exp(torch.clamp(logp - old_logp, -20, 20))
        clip = torch.clamp(ratio, 0.8, 1.2)
        pg_loss = -torch.min(ratio * adv_t, clip * adv_t).mean()
        v_loss = F.mse_loss(value, ret_t)
        loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()
        opt.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        opt.step()
        if (it + 1) % 10 == 0:
            print(
                f"RL {it+1}/{args.rl_iters} ret={float(rewards.sum()):.2f} "
                f"len={len(rewards)} loss={loss.item():.4f} grad={float(grad):.3f}"
            )

    # quick eval hold
    env.reset(q_t2=default_t2, height=stand_h)
    env.settle(default_t2, steps=100)
    last = np.zeros(31, dtype=np.float32)
    ok_steps = 0
    for _ in range(500):
        o = build_obs(env, last, default_t2)
        with torch.no_grad():
            a = student.act_inference(torch.as_tensor(o[None, :], device=device))
        a_np = np.clip(a.cpu().numpy()[0], -3.0, 3.0)
        env.step_pd(default_t2 + action_scale * a_np, n_substeps=args.decimation)
        last = a_np.astype(np.float32)
        if env.base_height() < 0.45 or env.upright() < 0.5:
            break
        ok_steps += 1
    print(f"eval_hold_steps={ok_steps}/500 height={env.base_height():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "obs_dim": OBS_DIM,
            "num_actions": 31,
            "action_scale": action_scale,
            "default_q_t2": default_t2.tolist(),
            "source_g1_ckpt": str(args.ckpt),
            "eval_hold_steps": ok_steps,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
