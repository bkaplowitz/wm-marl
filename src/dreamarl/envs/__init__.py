"""Environment adapters with explicit multi-agent contracts."""

from .dmc import make_dmc
from .meltingpot import MeltingPotEnv
from .single_agent import SingletonAgentEnv

__all__ = ["MeltingPotEnv", "SingletonAgentEnv", "make_dmc"]
