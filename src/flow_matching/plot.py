"""Small plotting helpers for the exercise script."""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

from flow_matching.distributions import (
    GaussianMixture2D,
    sample_gmm,
    sample_standard_normal,
)
from flow_matching.paths import conditional_vector_field, sample_conditional_path
from flow_matching.simulate import euler_integrate


def _finish_figure(fig: Figure, output: Path | None) -> None:
    """Save or show a figure and always close it."""
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _hist2d(
    ax: Axes,
    samples: jax.Array,
    *,
    scale: float,
    bins: int,
    title: str,
    cmap: str,
    alpha: float = 1.0,
) -> None:
    """Draw a 2D histogram with consistent axes."""
    samples_np = np.asarray(samples)
    ax.hist2d(
        samples_np[:, 0],
        samples_np[:, 1],
        bins=bins,
        range=[[-scale, scale], [-scale, scale]],
        cmap=cmap,
        alpha=alpha,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _standard_normal_log_density(x: jax.Array) -> jax.Array:
    dim = x.shape[1]
    return -0.5 * jnp.sum(x**2, axis=1) - 0.5 * dim * jnp.log(2.0 * jnp.pi)


def _gmm_log_density(x: jax.Array, gmm: GaussianMixture2D) -> jax.Array:
    dim = x.shape[1]
    diff = x[:, None, :] - gmm.means[None, :, :]
    component_log_density = (
        -0.5 * jnp.sum(diff**2, axis=2) / (gmm.std**2)
        - dim * jnp.log(gmm.std)
        - 0.5 * dim * jnp.log(2.0 * jnp.pi)
        + jnp.log(gmm.weights)
    )
    return jax.nn.logsumexp(component_log_density, axis=1)


def _grid_points(scale: float, bins: int) -> jax.Array:
    xs = jnp.linspace(-scale, scale, bins)
    ys = jnp.linspace(-scale, scale, bins)
    x_grid, y_grid = jnp.meshgrid(xs, ys, indexing="ij")
    return jnp.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)


def _imshow_log_density(
    ax: Axes,
    log_density: jax.Array,
    *,
    scale: float,
    bins: int,
    cmap: str,
    alpha: float = 0.25,
    vmin: float = -10.0,
) -> None:
    density_np = np.asarray(log_density).reshape(bins, bins).T
    ax.imshow(
        density_np,
        extent=(-scale, scale, -scale, scale),
        origin="lower",
        vmin=vmin,
        alpha=alpha,
        cmap=plt.get_cmap(cmap),
    )


def _plot_source_target_density_backgrounds(
    ax: Axes,
    gmm: GaussianMixture2D,
    *,
    scale: float,
    bins: int = 200,
) -> None:
    xy = _grid_points(scale, bins)
    _imshow_log_density(
        ax,
        _standard_normal_log_density(xy),
        scale=scale,
        bins=bins,
        cmap="Reds",
    )
    _imshow_log_density(
        ax,
        _gmm_log_density(xy, gmm),
        scale=scale,
        bins=bins,
        cmap="Blues",
    )


def _configure_density_axis(ax: Axes, title: str, scale: float) -> None:
    ax.set_title(title)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _scatter_conditioning_sample(
    ax: Axes,
    x1: jax.Array,
    *,
    s: float,
    color: str = "red",
) -> None:
    x1_np = np.asarray(x1)
    ax.scatter(
        x1_np[:, 0],
        x1_np[:, 1],
        marker="*",
        s=s,
        color=color,
        label="x1",
        zorder=20,
    )


def _scatter_marginals(
    ax: Axes,
    samples_by_time: jax.Array,
    ts: jax.Array,
    *,
    title: str,
    scale: float,
) -> None:
    """Scatter sampled marginals at selected times."""
    ax.set_title(title)
    for idx, t in enumerate(np.asarray(ts)):
        samples = np.asarray(samples_by_time[idx])
        ax.scatter(
            samples[:, 0], samples[:, 1], s=8, alpha=0.35, label=f"t={float(t):.2f}"
        )
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8)


def _record_indices(num_timesteps: int, num_marginals: int) -> jax.Array:
    """Return approximately even trajectory indices including endpoints."""
    return jnp.linspace(0, num_timesteps - 1, num_marginals).round().astype(jnp.int32)


def _fixed_x1_drift(x1: jax.Array) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Construct the analytic conditional ODE drift for a fixed conditioning sample."""

    def drift(xt: jax.Array, t: jax.Array) -> jax.Array:
        tt = jnp.full((xt.shape[0], 1), t)
        x1_batch = jnp.repeat(x1, xt.shape[0], axis=0)
        return conditional_vector_field(xt, x1_batch, tt)

    return drift


def _sample_marginal_path(
    key: jax.Array,
    gmm: GaussianMixture2D,
    *,
    t: jax.Array,
    num_samples: int,
) -> jax.Array:
    """Helper that draws a single marginal path in the gmm case."""
    key_x1, key_xt = jax.random.split(key)
    x1 = sample_gmm(key_x1, gmm, n=num_samples)
    tt = jnp.full((num_samples, 1), t)
    return sample_conditional_path(key_xt, x1, tt)


def create_source_and_target_figure(
    gmm: GaussianMixture2D,
    bins: int = 200,
    scale: float = 15.0,
) -> Figure:
    """Create the original three-panel source/target density heatmap figure."""
    xy = _grid_points(scale, bins)
    source_log_density = _standard_normal_log_density(xy)
    target_log_density = _gmm_log_density(xy, gmm)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    _configure_density_axis(axes[0], "Heatmap of p0", scale)
    _imshow_log_density(
        axes[0],
        source_log_density,
        scale=scale,
        bins=bins,
        cmap="Reds",
    )

    _configure_density_axis(axes[1], "Heatmap of p1", scale)
    _imshow_log_density(
        axes[1],
        target_log_density,
        scale=scale,
        bins=bins,
        cmap="Blues",
    )

    _configure_density_axis(axes[2], "Heatmap of p0 and p1", scale)
    _imshow_log_density(
        axes[2],
        source_log_density,
        scale=scale,
        bins=bins,
        cmap="Reds",
    )
    _imshow_log_density(
        axes[2],
        target_log_density,
        scale=scale,
        bins=bins,
        cmap="Blues",
    )
    return fig


def create_conditional_probability_path_figure(
    key: jax.Array,
    gmm: GaussianMixture2D,
    num_samples: int = 1_000,
    num_marginals: int = 7,
    scale: float = 15.0,
    background_bins: int = 200,
) -> Figure:
    """Create the original-style conditional path figure with density backgrounds."""
    key_x1, *sample_keys = jax.random.split(key, num_marginals + 1)
    x1 = sample_gmm(key_x1, gmm, n=1)
    ts = jnp.linspace(0.0, 1.0 - 1e-4, num_marginals)

    fig, ax = plt.subplots(figsize=(10, 10))
    _configure_density_axis(ax, "Gaussian Conditional Probability Path", scale)
    _plot_source_target_density_backgrounds(ax, gmm, scale=scale, bins=background_bins)
    _scatter_conditioning_sample(ax, x1, s=75)

    for sample_key, t in zip(sample_keys, ts, strict=True):
        x1_batch = jnp.repeat(x1, num_samples, axis=0)
        tt = jnp.full((num_samples, 1), t)
        samples = np.asarray(sample_conditional_path(sample_key, x1_batch, tt))
        ax.scatter(
            samples[:, 0], samples[:, 1], alpha=0.25, s=8, label=f"t={float(t):.1f}"
        )

    ax.legend(prop={"size": 18}, markerscale=3)
    return fig


def create_analytic_flow_path_figure(
    key: jax.Array,
    gmm: GaussianMixture2D,
    num_samples: int = 1_000,
    num_timesteps: int = 100,
    num_marginals: int = 3,
    scale: float = 15.0,
    background_bins: int = 200,
) -> Figure:
    """Create a 3-panel analytic conditional ODE flow figure."""
    key_x1, key_x0, *path_keys = jax.random.split(key, num_marginals + 2)
    x1 = sample_gmm(key_x1, gmm, n=1)
    x0 = sample_standard_normal(key_x0, n=num_samples, dim=2)
    ts = jnp.linspace(0.0, 1.0 - 1e-4, num_timesteps)
    trajectory = euler_integrate(_fixed_x1_drift(x1), x0, ts)
    record_idx = _record_indices(num_timesteps, num_marginals)
    record_ts = ts[record_idx]
    ode_samples = trajectory[record_idx]

    ground_truth = []
    for sample_key, t in zip(path_keys, record_ts, strict=True):
        x1_batch = jnp.repeat(x1, num_samples, axis=0)
        tt = jnp.full((num_samples, 1), t)
        ground_truth.append(sample_conditional_path(sample_key, x1_batch, tt))
    ground_truth = jnp.stack(ground_truth)

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    titles = (
        "Ground-Truth Conditional Probability Path",
        "Samples from Conditional ODE",
        "Trajectories of Conditional ODE",
    )
    for ax, title in zip(axes, titles, strict=True):
        _configure_density_axis(ax, title, scale)
        _plot_source_target_density_backgrounds(
            ax, gmm, scale=scale, bins=background_bins
        )
        _scatter_conditioning_sample(ax, x1, s=200)

    for idx, t in enumerate(np.asarray(record_ts)):
        gt_samples = np.asarray(ground_truth[idx])
        ode_samples_at_t = np.asarray(ode_samples[idx])
        label = f"t={float(t):.2f}"
        axes[0].scatter(
            gt_samples[:, 0], gt_samples[:, 1], marker="o", alpha=0.5, label=label
        )
        axes[1].scatter(
            ode_samples_at_t[:, 0],
            ode_samples_at_t[:, 1],
            marker="o",
            alpha=0.5,
            label=label,
        )

    traj_np = np.asarray(trajectory)
    for traj_idx in range(min(15, num_samples)):
        axes[2].plot(
            traj_np[:, traj_idx, 0], traj_np[:, traj_idx, 1], alpha=0.5, color="black"
        )

    for ax in axes:
        ax.legend(prop={"size": 18}, loc="upper right", markerscale=1.8)
    return fig


def create_learned_flow_path_figure(
    key: jax.Array,
    gmm: GaussianMixture2D,
    apply_fn: Callable,
    params,
    num_samples: int = 1_000,
    num_timesteps: int = 100,
    num_marginals: int = 3,
    scale: float = 15.0,
    background_bins: int = 200,
) -> Figure:
    """Create a 3-panel learned marginal ODE flow figure."""

    def drift(xt: jax.Array, t: jax.Array) -> jax.Array:
        tt = jnp.full((xt.shape[0], 1), t)
        return apply_fn({"params": params}, xt, tt)

    key_x0, *path_keys = jax.random.split(key, num_marginals + 1)
    x0 = sample_standard_normal(key_x0, n=num_samples, dim=2)
    ts = jnp.linspace(0.0, 1.0 - 1e-4, num_timesteps)
    trajectory = euler_integrate(drift, x0, ts)
    record_idx = _record_indices(num_timesteps, num_marginals)
    record_ts = ts[record_idx]
    learned_samples = trajectory[record_idx]

    ground_truth = []
    for sample_key, t in zip(path_keys, record_ts, strict=True):
        ground_truth.append(
            _sample_marginal_path(sample_key, gmm, t=t, num_samples=num_samples)
        )
    ground_truth = jnp.stack(ground_truth)

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    titles = (
        "Ground-Truth Marginal Probability Path",
        "Samples from Learned Marginal ODE",
        "Trajectories of Learned Marginal ODE",
    )
    for ax, title in zip(axes, titles, strict=True):
        _configure_density_axis(ax, title, scale)
        _plot_source_target_density_backgrounds(
            ax, gmm, scale=scale, bins=background_bins
        )

    for idx, t in enumerate(np.asarray(record_ts)):
        gt_samples = np.asarray(ground_truth[idx])
        learned_samples_at_t = np.asarray(learned_samples[idx])
        label = f"t={float(t):.2f}"
        axes[0].scatter(
            gt_samples[:, 0], gt_samples[:, 1], marker="o", alpha=0.5, label=label
        )
        axes[1].scatter(
            learned_samples_at_t[:, 0],
            learned_samples_at_t[:, 1],
            marker="o",
            alpha=0.5,
            label=label,
        )

    traj_np = np.asarray(trajectory)
    for traj_idx in range(min(max(num_samples // 10, 1), num_samples)):
        axes[2].plot(
            traj_np[:, traj_idx, 0], traj_np[:, traj_idx, 1], alpha=0.5, color="black"
        )

    axes[0].legend(prop={"size": 18}, loc="upper right", markerscale=1.8)
    axes[1].legend(prop={"size": 18}, loc="upper right", markerscale=1.8)
    return fig


def plot_source_and_target(
    key: jax.Array,
    gmm: GaussianMixture2D,
    output: Path | None = None,
    n: int = 10_000,
    bins: int = 100,
    scale: float = 15.0,
) -> None:
    """Plot source samples, target samples, and an overlay."""
    del key, n
    fig = create_source_and_target_figure(gmm, bins=bins, scale=scale)
    _finish_figure(fig, output)


def plot_conditional_probability_path(
    key: jax.Array,
    gmm: GaussianMixture2D,
    output: Path | None = None,
    num_samples: int = 1_000,
    num_marginals: int = 7,
    scale: float = 15.0,
) -> None:
    """Plot Gaussian conditional path samples for one target conditioning point."""
    fig = create_conditional_probability_path_figure(
        key,
        gmm,
        num_samples=num_samples,
        num_marginals=num_marginals,
        scale=scale,
    )
    _finish_figure(fig, output)


def plot_analytic_flow_path(
    key: jax.Array,
    gmm: GaussianMixture2D,
    output: Path | None = None,
    num_samples: int = 1_000,
    num_timesteps: int = 100,
    num_marginals: int = 4,
    scale: float = 15.0,
) -> None:
    """Compare ground-truth conditional marginals with analytic ODE trajectories."""
    fig = create_analytic_flow_path_figure(
        key,
        gmm,
        num_samples=num_samples,
        num_timesteps=num_timesteps,
        num_marginals=num_marginals,
        scale=scale,
    )
    _finish_figure(fig, output)


def plot_learned_flow_path(
    key: jax.Array,
    gmm: GaussianMixture2D,
    apply_fn: Callable,
    params,
    output: Path | None = None,
    num_samples: int = 1_000,
    num_timesteps: int = 100,
    num_marginals: int = 4,
    scale: float = 15.0,
) -> None:
    """Plot marginal samples and trajectories from a learned vector field."""
    fig = create_learned_flow_path_figure(
        key,
        gmm,
        apply_fn,
        params,
        num_samples=num_samples,
        num_timesteps=num_timesteps,
        num_marginals=num_marginals,
        scale=scale,
    )
    _finish_figure(fig, output)


def plot_training_loss(losses: jax.Array, output: Path | None = None) -> None:
    """Plot the flow-matching training loss on a log scale."""
    fig, ax = plt.subplots(figsize=(7, 4))
    losses_np = np.asarray(losses)
    ax.semilogy(np.arange(len(losses_np)), losses_np)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title("Flow Matching Training Loss")
    _finish_figure(fig, output)


def summarize_samples(samples: jax.Array) -> dict[str, jax.Array]:
    """Return a small numeric summary useful in no-plot smoke tests."""
    return {
        "mean": jnp.mean(samples, axis=0),
        "std": jnp.std(samples, axis=0),
    }


# TODO: plot rollout from env against world model generated env fit.
