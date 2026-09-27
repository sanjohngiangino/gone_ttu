"""G1 Goalkeeper mjlab subpackage. MDP is importable without mjlab."""

# Registration is explicit via register_task() to keep mdp importable on macOS.
from goalkeeper_tasks_mjlab.g1 import mdp as mdp

__all__ = ["mdp", "register_task"]


def register_task() -> None:
    from goalkeeper_tasks_mjlab.g1.register import register

    register()
