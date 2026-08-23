"""Environment adapters with explicit multi-agent contracts."""

from .dmc import make_dmc
from .meltingpot import MeltingPotEnv
from .single_agent import SingletonAgentEnv
from .smac import SMACEnv

__all__ = ["MeltingPotEnv", "SMACEnv", "SingletonAgentEnv", "make_dmc"]
