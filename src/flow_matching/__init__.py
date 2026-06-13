"""Compact JAX flow-matching exercise package."""

from flow_matching.distributions import GaussianMixture2D, make_symmetric_gmm_2d
from flow_matching.paths import alpha, beta, beta_dt

__all__ = [
    "GaussianMixture2D",
    "alpha",
    "beta",
    "beta_dt",
    "make_symmetric_gmm_2d",
]
