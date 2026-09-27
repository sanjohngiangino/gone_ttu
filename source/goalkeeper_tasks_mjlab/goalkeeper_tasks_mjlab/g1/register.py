"""Register SPQR-Mjlab-Goalkeeper-G1 (requires mjlab)."""

from __future__ import annotations


def register() -> None:
    try:
        from mjlab.tasks.registry import register_mjlab_task
    except ImportError as exc:
        raise ImportError(
            "mjlab is required to register SPQR-Mjlab-Goalkeeper-G1. "
            "See docs/MJLAB_GOALKEEPER_PORT.md"
        ) from exc

    from goalkeeper_tasks_mjlab.g1.agents.ppo_cfg import g1_goalkeeper_runner_cfg
    from goalkeeper_tasks_mjlab.g1.env_cfg import g1_goalkeeper_env_cfg

    register_mjlab_task(
        task_id="SPQR-Mjlab-Goalkeeper-G1",
        env_cfg=g1_goalkeeper_env_cfg(),
        play_env_cfg=g1_goalkeeper_env_cfg(play=True),
        rl_cfg=g1_goalkeeper_runner_cfg(),
        runner_cls=None,
    )
