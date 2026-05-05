from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg


@configclass
class P73RslRlPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    algorithm: RslRlPpoAlgorithmCfg | P73RslRlPpoAlgorithmCfg = MISSING
@configclass
class P73RslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    bound_loss_cfg: RslRlBoundLossCfg | None = None


@configclass
class RslRlBoundLossCfg:
    """Configuration for the bound loss."""
    bound_loss_coef: float = 10.0
    """The coefficient for the bound loss."""

    bound_range: float = 1.1