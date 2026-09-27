"""PPO / runner config for G1 Goalkeeper (mjlab + rsl-rl)."""

from __future__ import annotations


def g1_goalkeeper_runner_cfg():
    """Return mjlab RSL-RL runner cfg.

    Uses mjlab's stock PPO runner. AMP region-conditioned (Isaac amp_coef=0.4)
    can be swapped to beyondAMP AMPRunnerCfg later — see docs/MJLAB_GOALKEEPER_PORT.md.
    """
    try:
        from mjlab.tasks.velocity.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
    except ImportError:
        # Fallback import paths vary by mjlab version
        from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # type: ignore

    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 256],
            critic_hidden_dims=[512, 256, 256],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="spqr_g1_goalkeeper",
        max_iterations=200000,
        save_interval=200,
        num_steps_per_env=100,
    )
