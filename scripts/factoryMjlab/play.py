#!/usr/bin/env python3
"""Play / visualize SPQR Goalkeeper on mjlab (G1).

Examples:
  python scripts/factoryMjlab/play.py SPQR-Mjlab-Goalkeeper-G1 --agent zero
  python scripts/factoryMjlab/play.py SPQR-Mjlab-Goalkeeper-G1 --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="SPQR-Mjlab-Goalkeeper-G1")
    parser.add_argument("--agent", default="zero", help="zero | random | checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    args, unknown = parser.parse_known_args()

    import goalkeeper_tasks_mjlab

    goalkeeper_tasks_mjlab.register()


    try:
        from mjlab.scripts.play import main as mjlab_play
    except ImportError:
        try:
            from mjlab.tasks.play import play as mjlab_play  # type: ignore
        except ImportError as exc:
            print(
                "mjlab play entrypoint not found. Install mjlab (GPU Linux recommended).\n"
                "Meanwhile you can still use the IsaacGym official play.py from "
                "InternRobotics/Humanoid-Goalkeeper.\n"
                f"Import error: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    sys.argv = [
        sys.argv[0],
        args.task,
        f"--agent={args.agent}",
        f"--env.scene.num-envs={args.num_envs}",
        *unknown,
    ]
    if args.checkpoint:
        sys.argv.append(f"--checkpoint={args.checkpoint}")
    mjlab_play()


if __name__ == "__main__":
    main()
