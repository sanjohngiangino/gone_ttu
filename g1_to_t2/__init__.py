"""G1 → T2 policy transfer: joint remap, checkpoint load, wrappers."""

from g1_to_t2.adapter import JointAdapter
from g1_to_t2.joints import G1_JOINT_NAMES, T2_JOINT_NAMES
from g1_to_t2.loader import inspect_checkpoint, load_rsl_actor
from g1_to_t2.policy import G1PolicyWrapper, PolicyConfig

__all__ = [
    "G1_JOINT_NAMES",
    "T2_JOINT_NAMES",
    "JointAdapter",
    "load_rsl_actor",
    "inspect_checkpoint",
    "G1PolicyWrapper",
    "PolicyConfig",
]

__version__ = "0.1.0"
