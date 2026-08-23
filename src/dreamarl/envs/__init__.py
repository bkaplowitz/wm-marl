"""Environment adapters with explicit multi-agent contracts."""

from .dmc import make_dmc
from .single_agent import SingletonAgentEnv
from .smac import SMACEnv

__all__ = ["SMACEnv", "SingletonAgentEnv", "make_dmc"]
