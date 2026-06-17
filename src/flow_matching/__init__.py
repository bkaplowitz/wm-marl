"""Compact JAX flow-matching package."""

from flow_matching.distributions import GaussianMixture2D, make_symmetric_gmm_2d
from flow_matching.paths import (
    alpha,
    flow_schedule,
    gaussian_beta,
    gaussian_beta_dt,
    linear_beta,
    linear_beta_dt,
)

__all__ = [
    "GaussianMixture2D",
    "alpha",
    "flow_schedule",
    "gaussian_beta",
    "gaussian_beta_dt",
    "linear_beta",
    "linear_beta_dt",
    "make_symmetric_gmm_2d",
]
