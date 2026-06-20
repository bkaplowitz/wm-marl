"""Environment adapters."""

from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter

__all__ = [
    "GymnaxVectorAdapter",
    "JaxMARLCoinGameVectorAdapter",
    "MeltingPotVectorAdapter",
]
