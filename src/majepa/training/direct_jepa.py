"""Direct multi-step JEPA supervision and action contrast."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..models.multistep_jepa import (
    isolated_creation_call,
)
from .multistep_jepa import (
    aligned_action_windows,
    all_legal_same_focal_action_interventions,
    direct_multistep_objective,
)


def direct_jepa_losses(
    self,
    grouped_hidden,
    grouped_source_state,
    grouped_target,
    grouped_present,
    grouped_alive,
    grouped_first,
    grouped_mask,
    grouped_action,
    *,
    training,
):
    """Predict factual future embeddings and contrast legal action alternatives."""

    if not self.ctde_multistep_jepa_enabled or not self.two_branch_replay:
        raise RuntimeError(
            "direct multi-step JEPA requires the recent world replay branch"
        )
    max_horizon = self.ctde_multistep_jepa_max_horizon
    length = grouped_target.shape[1]
    roots = length - max_horizon
    expected = (grouped_target.shape[0], length - 1, self.team.size)
    if grouped_hidden.shape[:3] != expected:
        raise ValueError(
            "multi-step shared hidden is not aligned with factual replay: "
            f"{grouped_hidden.shape[:3]} versus {expected}"
        )
    if grouped_source_state.shape[:3] != expected:
        raise ValueError(
            "multi-step local roots are not aligned with factual replay: "
            f"{grouped_source_state.shape[:3]} versus {expected}"
        )

    action_windows, all_valid = aligned_action_windows(
        grouped_action[:, 1:],
        grouped_mask[:, :-1],
        grouped_present,
        grouped_alive,
        grouped_first,
        action_low=self.ctde_action_low,
        max_horizon=max_horizon,
    )
    horizons = self.ctde_multistep_jepa_horizons
    valid = {horizon: all_valid[horizon] for horizon in horizons}
    root_hidden = grouped_hidden[:, :roots]

    predictions = isolated_creation_call(
        self.ctde_multistep_jepa,
        0x4D534A50,
        root_hidden,
        action_windows,
    )
    if self.ctde_multistep_jepa_action_scale > 0.0:
        counterfactual_windows, distinct_tail = (
            all_legal_same_focal_action_interventions(
                action_windows,
                grouped_mask[:, :-1],
                action_low=self.ctde_action_low,
                horizons=horizons,
            )
        )
        counterfactual_predictions = {}
        counterfactual_valid = {}
        classes = grouped_mask.shape[-1]
        batch = root_hidden.shape[0]
        expanded_root = jnp.broadcast_to(
            root_hidden[:, None],
            (batch, classes, *root_hidden.shape[1:]),
        ).reshape((batch * classes, *root_hidden.shape[1:]))
        for horizon in horizons:
            candidate_valid = valid[horizon][..., None] & distinct_tail[horizon]
            counterfactual_valid[horizon] = candidate_valid
            if horizon == 1:
                counterfactual_predictions[horizon] = jnp.broadcast_to(
                    jax.lax.stop_gradient(predictions[horizon])[..., None, :],
                    (
                        *predictions[horizon].shape[:-1],
                        classes,
                        predictions[horizon].shape[-1],
                    ),
                )
                continue
            window = counterfactual_windows[horizon]
            flat_window = jnp.transpose(window, (0, 3, 1, 2, 4)).reshape(
                (batch * classes, *window.shape[1:3], window.shape[-1])
            )
            flat_prediction = self.ctde_multistep_jepa(
                expanded_root,
                flat_window,
                selected_horizon=horizon,
            )[horizon]
            counterfactual_predictions[horizon] = jnp.transpose(
                flat_prediction.reshape((batch, classes, *flat_prediction.shape[1:])),
                (0, 2, 3, 1, 4),
            )
        counterfactual_enabled = jnp.asarray(1.0, jnp.float32)
    else:
        counterfactual_predictions = {
            horizon: jax.lax.stop_gradient(predictions[horizon]) for horizon in horizons
        }
        counterfactual_valid = {
            horizon: jnp.zeros_like(valid[horizon]) for horizon in horizons
        }
        distinct_tail = {
            horizon: jnp.zeros_like(valid[horizon]) for horizon in horizons
        }
        counterfactual_enabled = jnp.asarray(0.0, jnp.float32)

    targets = {
        horizon: jax.lax.stop_gradient(grouped_target[:, horizon : horizon + roots])
        for horizon in horizons
    }
    root_losses, raw_metrics = direct_multistep_objective(
        predictions,
        targets,
        valid,
        counterfactual_predictions,
        counterfactual_valid,
        distinct_tail,
        horizons=horizons,
        decay=self.ctde_multistep_jepa_decay,
        action_margin=self.ctde_multistep_jepa_action_margin,
    )
    losses = {}
    for name, root_loss in root_losses.items():
        padded = jnp.pad(root_loss, ((0, 0), (0, max_horizon), (0, 0)))
        padded *= length / roots
        losses[f"ctde_multistep_jepa_{name}"] = self.team.fold_sequence(padded)

    metrics = {
        f"ctde/multistep_jepa_{key}": value for key, value in raw_metrics.items()
    }
    metrics.update(
        {
            "ctde/multistep_jepa_root_count": jnp.asarray(
                grouped_target.shape[0] * roots * self.team.size,
                jnp.float32,
            ),
            "ctde/multistep_jepa_max_horizon": jnp.asarray(max_horizon, jnp.float32),
            "ctde/multistep_jepa_action_counterfactual_enabled": (
                counterfactual_enabled
            ),
            "ctde/multistep_jepa_recent_training_view": jnp.asarray(
                float(bool(training)), jnp.float32
            ),
        }
    )
    metrics["ctde/multistep_jepa_action_counterfactual_all_legal_enabled"] = (
        jnp.asarray(float(self.ctde_multistep_jepa_action_scale > 0.0), jnp.float32)
    )
    return losses, metrics
