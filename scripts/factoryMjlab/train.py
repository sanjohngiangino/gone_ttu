#!/usr/bin/env python3
"""Train SPQR Goalkeeper on mjlab (G1).

Requires: pip install mjlab && pip install -e source/goalkeeper_tasks_mjlab

Example:
  python scripts/factoryMjlab/train.py SPQR-Mjlab-Goalkeeper-G1 --num_envs 64 --max_iterations 50
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="SPQR-Mjlab-Goalkeeper-G1")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args, unknown = parser.parse_known_args()

    # Register task
    import goalkeeper_tasks_mjlab

    goalkeeper_tasks_mjlab.register()


    try:
        from mjlab.scripts.train import main as mjlab_train
    except ImportError:
        try:
            # newer / alternate entrypoints
            from mjlab.tasks.train import train as mjlab_train  # type: ignore
        except ImportError as exc:
            print(
                "mjlab train entrypoint not found. Install mjlab and see "
                "docs/MJLAB_GOALKEEPER_PORT.md\n"
                f"Import error: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    # Prefer CLI passthrough to mjlab's train if it uses tyro/hydra
    sys.argv = [
        sys.argv[0],
        args.task,
        f"--env.scene.num-envs={args.num_envs}",
        f"--agent.max-iterations={args.max_iterations}",
        *unknown,
    ]
    if args.headless:
        sys.argv.append("--headless")
    mjlab_train()


if __name__ == "__main__":
    main()
