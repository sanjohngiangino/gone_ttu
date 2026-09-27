"""SPQR Goalkeeper tasks for mjlab (G1 first)."""

__all__ = ["register"]


def register() -> None:
    """Register mjlab tasks (requires mjlab installed)."""
    from goalkeeper_tasks_mjlab.g1.register import register as _reg

    _reg()
