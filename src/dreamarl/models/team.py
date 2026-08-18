"""Training-only agent-axis JEPA and team-belief modules.

B1 extends the local JEPA along the agent axis. B2 also uses these modules to
construct a causal team belief for centralized value learning. The online branch sees a
randomly masked set of complete agents. An EMA teacher sees the complete
active team and produces aligned fixed-width team slots. Nothing here is part
of the execution-time actor.
"""

import math

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj


f32 = jnp.float32
sg = jax.lax.stop_gradient


def _masked_softmax(logits, mask):
    mask = mask.astype(bool)
    values = jax.nn.softmax(jnp.where(mask, logits, -1e30), axis=-1)
    values = values * mask.astype(values.dtype)
    return values / jnp.maximum(values.sum(axis=-1, keepdims=True), 1e-8)


def _competitive_slot_weights(logits, mask):
    """Slot-Attention normalization: members compete across slots first."""

    mask = mask.astype(bool)
    values = jax.nn.softmax(logits, axis=-2)
    values = values * mask.astype(values.dtype)
    return values / jnp.maximum(values.sum(axis=-1, keepdims=True), 1e-8)


class TeamSlotEncoder(nj.Module):
    """Perceiver-style set encoder returning K permutation-invariant slots."""

    slots: int = 8
    width: int = 256
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, **kwargs):
        if self.slots < 1 or self.layers < 1:
            raise ValueError("team encoder needs at least one slot and layer")
        if self.width % self.heads:
            raise ValueError(
                f"team width {self.width} must be divisible by {self.heads} heads"
            )
        del kwargs

    def __call__(self, members, visible, active):
        """Encode [B,T,A,E] members into [B,T,K,D] team slots.

        ``visible`` controls available member contents. ``active`` is retained
        separately, so roster size is represented without leaking masked
        member content.
        """

        if members.ndim != 4 or visible.shape != members.shape[:3]:
            raise ValueError(
                "team encoder expects members [B,T,A,E] and visible [B,T,A], "
                f"got {members.shape} and {visible.shape}"
            )
        if active.shape != visible.shape:
            raise ValueError(
                f"active mask {active.shape} does not match visible {visible.shape}"
            )
        batch, length, agents = members.shape[:3]
        visible = visible.astype(bool) & active.astype(bool)
        active = active.astype(bool)
        members = self.sub("input_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(members)
        )
        members = self.sub("member_norm", nn.Norm, self.norm)(members)

        queries = self.value(
            "queries", nn.init("trunc_normal"), (self.slots, self.width), f32
        )
        slots = nn.cast(
            jnp.broadcast_to(queries, (batch, length, self.slots, self.width))
        )
        active_count = active.sum(axis=-1, keepdims=True).astype(f32)
        visible_count = visible.sum(axis=-1, keepdims=True).astype(f32)
        count_features = jnp.concatenate(
            [
                jnp.log1p(active_count),
                active_count / max(agents, 1),
                jnp.log1p(visible_count),
                visible_count / max(agents, 1),
            ],
            axis=-1,
        )
        count_embedding = self.sub(
            "count_projection", nn.Linear, self.width, winit=self.winit
        )(nn.cast(count_features))
        slots = slots + count_embedding[:, :, None]

        for index in range(self.layers):
            slots = self._cross_attention(
                slots,
                members,
                visible,
                index,
                residual=index > 0,
            )
            residual = slots
            value = self.sub(f"ffn{index}_norm", nn.Norm, self.norm)(slots)
            value = self.sub(
                f"ffn{index}_in", nn.Linear, self.width * self.ffup, winit=self.winit
            )(value)
            value = nn.act(self.act)(value)
            value = self.sub(
                f"ffn{index}_out", nn.Linear, self.width, winit=self.winit
            )(value)
            slots = residual + value
        return self.sub("output_norm", nn.Norm, self.norm)(slots)

    def _cross_attention(self, slots, members, visible, index, *, residual):
        batch, length, _, _ = slots.shape
        agents = members.shape[2]
        head_width = self.width // self.heads
        previous = slots
        slots = self.sub(f"cross{index}_norm", nn.Norm, self.norm)(slots)
        queries = self.sub(
            f"cross{index}_query", nn.Linear, self.width, winit=self.winit
        )(slots)
        keys = self.sub(f"cross{index}_key", nn.Linear, self.width, winit=self.winit)(
            members
        )
        values = self.sub(
            f"cross{index}_value", nn.Linear, self.width, winit=self.winit
        )(members)
        queries = queries.reshape(batch, length, self.slots, self.heads, head_width)
        keys = keys.reshape(batch, length, agents, self.heads, head_width)
        values = values.reshape(batch, length, agents, self.heads, head_width)
        logits = jnp.einsum("btkhd,btahd->bthka", queries, keys)
        weights = _competitive_slot_weights(
            f32(logits) / math.sqrt(head_width),
            visible[:, :, None, None, :],
        ).astype(values.dtype)
        update = jnp.einsum("bthka,btahd->btkhd", weights, values)
        update = update.reshape(batch, length, self.slots, self.width)
        update = self.sub(f"cross{index}_out", nn.Linear, self.width, winit=self.winit)(
            update
        )
        # Learned queries determine attention but never enter the returned
        # representation additively. At the first layer, learned slot codes
        # gate member content multiplicatively, breaking slot symmetry while
        # remaining exactly zero without member content. Later layers refine
        # the resulting content slots residually.
        if not residual:
            codes = self.value(
                "content_codes",
                nn.Initializer("normal", "none", 1.0),
                (self.slots, self.width),
                f32,
            )
            # Each feature is competitively routed across slots while the mean
            # gate remains one. The learned code has no additive path to the
            # representation; it can only partition attended member content.
            gate = jax.nn.softmax(codes, axis=0) * self.slots
            update = update * nn.cast(gate)[None, None]
        return previous + update if residual else update


class AgentContextEncoder(nj.Module):
    """Encode visible local world-model histories into fixed-width slots."""

    slots: int = 8
    width: int = 256
    heads: int = 4
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, **kwargs):
        if self.width % self.heads:
            raise ValueError(
                f"team width {self.width} must be divisible by {self.heads} heads"
            )
        del kwargs

    def __call__(self, histories, visible):
        if histories.ndim != 4 or visible.shape != histories.shape[:3]:
            raise ValueError(
                "agent context encoder expects histories [B,T,A,H] and visible "
                f"[B,T,A], got {histories.shape} and {visible.shape}"
            )
        batch, length, agents = histories.shape[:3]
        members = self.sub("input_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(histories)
        )
        members = self.sub("member_norm", nn.Norm, self.norm)(members)
        queries = self.value(
            "queries", nn.init("trunc_normal"), (self.slots, self.width), f32
        )
        queries = nn.cast(
            jnp.broadcast_to(queries, (batch, length, self.slots, self.width))
        )
        queries = self.sub("query_norm", nn.Norm, self.norm)(queries)
        queries = self.sub("query", nn.Linear, self.width, winit=self.winit)(queries)
        keys = self.sub("key", nn.Linear, self.width, winit=self.winit)(members)
        values = self.sub("value", nn.Linear, self.width, winit=self.winit)(members)
        head_width = self.width // self.heads
        queries = queries.reshape(batch, length, self.slots, self.heads, head_width)
        keys = keys.reshape(batch, length, agents, self.heads, head_width)
        values = values.reshape(batch, length, agents, self.heads, head_width)
        logits = jnp.einsum("btkhd,btahd->bthka", queries, keys)
        weights = _masked_softmax(
            f32(logits) / math.sqrt(head_width), visible[:, :, None, None, :]
        ).astype(values.dtype)
        slots = jnp.einsum("bthka,btahd->btkhd", weights, values)
        slots = slots.reshape(batch, length, self.slots, self.width)
        slots = self.sub("attention_out", nn.Linear, self.width, winit=self.winit)(
            slots
        )
        residual = slots
        slots = self.sub("ffn_norm", nn.Norm, self.norm)(slots)
        slots = self.sub("ffn_in", nn.Linear, self.width * self.ffup, winit=self.winit)(
            slots
        )
        slots = nn.act(self.act)(slots)
        slots = self.sub("ffn_out", nn.Linear, self.width, winit=self.winit)(slots)
        return self.sub("output_norm", nn.Norm, self.norm)(residual + slots)


class TeamActionConditioner(nj.Module):
    """Fuse every agent's current content with its aligned joint action."""

    hidden: int = 512
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, output_dim, **kwargs):
        self.output_dim = int(output_dim)
        if self.output_dim < 1:
            raise ValueError("team action conditioner needs a positive output width")
        del kwargs

    def __call__(self, members, actions, visible, active):
        if members.ndim != 4 or actions.ndim != 4:
            raise ValueError("conditioner expects [B,T,A,E] members and actions")
        if members.shape[:3] != actions.shape[:3]:
            raise ValueError(
                f"member/action axes do not match: {members.shape} and {actions.shape}"
            )
        if visible.shape != members.shape[:3] or active.shape != visible.shape:
            raise ValueError("conditioner masks do not match the member agent axis")
        visible = visible.astype(bool) & active.astype(bool)
        active = active.astype(bool)
        content = jnp.where(visible[..., None], members, 0)
        content = self.sub("content_norm", nn.Norm, self.norm)(nn.cast(content))
        content = self.sub("content_hidden", nn.Linear, self.hidden, winit=self.winit)(
            content
        )
        flags = jnp.stack([visible, active], axis=-1).astype(actions.dtype)
        condition = jnp.concatenate([actions, flags], axis=-1)
        condition = self.sub("action_hidden", nn.Linear, self.hidden, winit=self.winit)(
            nn.cast(condition)
        )
        value = content + condition
        value = nn.act(self.act)(value)
        value = self.sub("output", nn.Linear, self.output_dim, winit=self.winit)(value)
        return value * active[..., None].astype(value.dtype)


class TeamSlotPredictor(nj.Module):
    """Predict complete EMA team slots from masked content and history slots."""

    width: int = 256
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, **kwargs):
        if self.width % self.heads:
            raise ValueError(
                f"predictor width {self.width} must be divisible by {self.heads}"
            )
        del kwargs

    def __call__(self, content_slots, history_slots):
        if content_slots.shape != history_slots.shape:
            raise ValueError(
                f"slot inputs must match, got {content_slots.shape} and "
                f"{history_slots.shape}"
            )
        value = jnp.concatenate([content_slots, history_slots], axis=-1)
        value = self.sub("input", nn.Linear, self.width, winit=self.winit)(value)
        for index in range(self.layers):
            value = self._attention(value, index)
            residual = value
            update = self.sub(f"ffn{index}_norm", nn.Norm, self.norm)(value)
            update = self.sub(
                f"ffn{index}_in", nn.Linear, self.width * self.ffup, winit=self.winit
            )(update)
            update = nn.act(self.act)(update)
            update = self.sub(
                f"ffn{index}_out", nn.Linear, self.width, winit=self.winit
            )(update)
            value = residual + update
        value = self.sub("output_norm", nn.Norm, self.norm)(value)
        return self.sub("output", nn.Linear, self.width, winit=self.winit)(value)

    def _attention(self, slots, index):
        batch, length, count, _ = slots.shape
        head_width = self.width // self.heads
        residual = slots
        value = self.sub(f"attn{index}_norm", nn.Norm, self.norm)(slots)
        query = self.sub(f"attn{index}_query", nn.Linear, self.width, winit=self.winit)(
            value
        )
        key = self.sub(f"attn{index}_key", nn.Linear, self.width, winit=self.winit)(
            value
        )
        value = self.sub(f"attn{index}_value", nn.Linear, self.width, winit=self.winit)(
            value
        )
        query = query.reshape(batch, length, count, self.heads, head_width)
        key = key.reshape(batch, length, count, self.heads, head_width)
        value = value.reshape(batch, length, count, self.heads, head_width)
        logits = jnp.einsum("btkhd,btlhd->bthkl", query, key)
        weights = jax.nn.softmax(f32(logits) / math.sqrt(head_width), axis=-1).astype(
            value.dtype
        )
        update = jnp.einsum("bthkl,btlhd->btkhd", weights, value)
        update = update.reshape(batch, length, count, self.width)
        update = self.sub(f"attn{index}_out", nn.Linear, self.width, winit=self.winit)(
            update
        )
        return residual + update


class TeamContentPredictor(nj.Module):
    """Decode every team slot into the shared EMA local-content space."""

    hidden: int = 512
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, target_dim, **kwargs):
        self.target_dim = int(target_dim)
        del kwargs

    def __call__(self, team_slots):
        value = self.sub("norm", nn.Norm, self.norm)(team_slots)
        value = self.sub("hidden", nn.Linear, self.hidden, winit=self.winit)(value)
        value = nn.act(self.act)(value)
        return self.sub("output", nn.Linear, self.target_dim, winit=self.winit)(value)


def mask_active_agents(active, key, *, minimum=0.25, maximum=0.5):
    """Mask 25--50% of complete active agents, leaving both sides nonempty."""

    if active.ndim != 3:
        raise ValueError(f"active mask must be [B,T,A], got {active.shape}")
    if not 0.0 < minimum <= maximum < 1.0:
        raise ValueError("agent mask range must satisfy 0 < minimum <= maximum < 1")
    active = active.astype(bool)
    eligible = active.sum(axis=-1) >= 2
    ratio_key, order_key = jax.random.split(key)
    ratio = jax.random.uniform(
        ratio_key, active.shape[:2], minval=minimum, maxval=maximum
    )
    count = active.sum(axis=-1)
    hidden_count = jnp.rint(count * ratio).astype(jnp.int32)
    hidden_count = jnp.clip(hidden_count, 1, jnp.maximum(count - 1, 1))
    scores = jax.random.uniform(order_key, active.shape, dtype=f32)
    scores = jnp.where(active, scores, 2.0)
    ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
    hidden = active & (ranks < hidden_count[..., None]) & eligible[..., None]
    return active & ~hidden, hidden, eligible


def scale_gradient(value, scale):
    """Keep the forward value while scaling gradients into the local model."""

    scale = f32(scale)
    return sg(value) + scale * (value - sg(value))


def _cosine_loss(prediction, target):
    prediction = f32(prediction)
    target = sg(f32(target))
    prediction = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8
    )
    target = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    cosine = (prediction * target).sum(axis=-1)
    return 1.0 - cosine, cosine


def team_slot_jepa_loss(prediction, target, eligible, *, name="team"):
    """Slot-aligned cosine prediction loss against the full EMA team teacher."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"team slots must be matching [B,T,K,D], got {prediction.shape} and "
            f"{target.shape}"
        )
    loss, cosine = _cosine_loss(prediction, target)
    weight = eligible.astype(f32)
    normalizer = jnp.maximum(weight.mean(), 1e-8)
    per_transition = loss.mean(axis=-1) * weight / normalizer
    metrics = {
        f"agent_jepa/{name}_cosine": (cosine.mean(axis=-1) * weight).sum()
        / jnp.maximum(weight.sum(), 1),
        f"agent_jepa/{name}_target_std": _weighted_std(target, weight),
        f"agent_jepa/{name}_prediction_std": _weighted_std(prediction, weight),
        f"agent_jepa/{name}_effective_rank": team_slot_effective_rank(target, weight),
        f"agent_jepa/{name}_prediction_effective_rank": team_slot_effective_rank(
            prediction, weight
        ),
    }
    return per_transition, metrics


def team_set_matching_loss(
    prediction,
    target,
    active,
    valid,
    *,
    temperature=0.1,
    iterations=5,
    name,
):
    """Balanced optimal-transport loss between slots and active EMA members.

    The transport plan is stop-gradient. Equal member marginals force every
    active agent to receive coverage, while equal slot marginals prevent dead
    slots. The loss remains permutation invariant in both sets.
    """

    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("set matching expects [B,T,K,D] and [B,T,A,D]")
    if (
        prediction.shape[:2] != target.shape[:2]
        or prediction.shape[-1] != target.shape[-1]
    ):
        raise ValueError(
            f"incompatible slot/member shapes {prediction.shape} and {target.shape}"
        )
    if active.shape != target.shape[:3] or valid.shape != target.shape[:2]:
        raise ValueError("active or valid mask does not match team targets")
    if temperature <= 0 or iterations < 1:
        raise ValueError("matching needs positive temperature and iterations")

    prediction = f32(prediction)
    target = sg(f32(target))
    prediction, target = _center_team_content(prediction, target, active)
    normalized_prediction = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8
    )
    normalized_target = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8
    )
    similarity = jnp.einsum("btkd,btad->btka", normalized_prediction, normalized_target)
    cost = 1.0 - similarity

    # Supply one inert member to numerically define transport at empty padded
    # timesteps; ``valid`` removes those transitions from every optimized term.
    active = active.astype(bool)
    any_active = active.any(axis=-1)
    safe_active = active.at[..., 0].set(active[..., 0] | ~any_active)
    member_count = safe_active.sum(axis=-1).astype(f32)
    slots = prediction.shape[-2]
    log_row = jnp.full(prediction.shape[:3], -jnp.log(f32(slots)))
    log_col = jnp.where(
        safe_active,
        -jnp.log(member_count)[..., None],
        -1e30,
    )
    log_kernel = -sg(cost) / f32(temperature)
    log_kernel = jnp.where(safe_active[:, :, None, :], log_kernel, -1e30)
    log_u = jnp.zeros_like(log_row)
    log_v = jnp.where(safe_active, 0.0, -1e30)
    for _ in range(int(iterations)):
        log_u = log_row - jax.nn.logsumexp(log_kernel + log_v[:, :, None, :], axis=-1)
        log_v = log_col - jax.nn.logsumexp(log_kernel + log_u[:, :, :, None], axis=-2)
    plan = jnp.exp(log_u[:, :, :, None] + log_kernel + log_v[:, :, None, :])
    plan = sg(jnp.where(safe_active[:, :, None, :], plan, 0.0))

    weight = valid.astype(f32)
    normalizer = jnp.maximum(weight.mean(), 1e-8)
    transport_cost = (plan * cost).sum(axis=(-2, -1))
    per_transition = transport_cost * weight / normalizer
    plan_entropy = -(plan * jnp.log(jnp.maximum(plan, 1e-12))).sum(axis=(-2, -1))
    member_weight = active.astype(f32) * weight[..., None]
    conditional_plan = plan * member_count[:, :, None, None]
    assignment_peak = conditional_plan.max(axis=-2)
    similarity_spread = similarity.std(axis=-2)
    metrics = {
        f"agent_jepa/{name}_cosine": (
            (plan * similarity).sum(axis=(-2, -1)) * weight
        ).sum()
        / jnp.maximum(weight.sum(), 1),
        f"agent_jepa/{name}_target_std": _weighted_member_std(target, active, weight),
        f"agent_jepa/{name}_prediction_std": _weighted_std(prediction, weight),
        f"agent_jepa/{name}_assignment_entropy": (plan_entropy * weight).sum()
        / jnp.maximum(weight.sum(), 1),
        f"agent_jepa/{name}_assignment_peak": (assignment_peak * member_weight).sum()
        / jnp.maximum(member_weight.sum(), 1),
        f"agent_jepa/{name}_slot_similarity_std": (
            similarity_spread * member_weight
        ).sum()
        / jnp.maximum(member_weight.sum(), 1),
    }
    return per_transition, metrics


def masked_agent_coverage_loss(
    prediction,
    target,
    active,
    hidden,
    eligible,
    *,
    temperature=0.1,
):
    """Require every completely hidden EMA member to match at least one slot."""

    prediction = f32(prediction)
    target = sg(f32(target))
    if active.shape != target.shape[:3] or hidden.shape != active.shape:
        raise ValueError("coverage masks do not match team targets")
    prediction, target = _center_team_content(prediction, target, active)
    prediction = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8
    )
    target = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    similarity = jnp.einsum("btkd,btad->btka", prediction, target)
    cost = 1.0 - similarity
    assignment = jax.nn.softmax(-sg(cost) / f32(temperature), axis=-2)
    member_cost = (assignment * cost).sum(axis=-2)
    hidden_weight = hidden.astype(f32)
    transition_cost = (member_cost * hidden_weight).sum(axis=-1) / jnp.maximum(
        hidden_weight.sum(axis=-1), 1
    )
    weight = eligible.astype(f32)
    per_transition = transition_cost * weight / jnp.maximum(weight.mean(), 1e-8)
    return per_transition, {
        "agent_jepa/hidden_coverage_cosine": 1.0
        - (transition_cost * weight).sum() / jnp.maximum(weight.sum(), 1),
        "agent_jepa/eligible_fraction": weight.mean(),
    }


def team_utility_probe_metrics(
    sources,
    actions,
    rewards,
    active,
    hidden,
    eligible,
    resets,
    *,
    action_count,
    horizon=8,
    ridge=1e-2,
):
    """Report frozen linear accessibility of behavior and short returns.

    A ridge readout is fitted in closed form on the first half of the replay
    batch and evaluated on the second half. Inputs and targets are stop-gradient
    and no probe parameters are created, so these diagnostics cannot train or
    otherwise alter the representation being measured.
    """

    if not sources:
        return {}
    reference = next(iter(sources.values()))
    if reference.ndim != 4:
        raise ValueError("team utility sources must be [B,T,K,D]")
    batch, length = reference.shape[:2]
    if batch < 2 or length <= horizon:
        return {}
    expected = (batch, length, active.shape[-1])
    if actions.shape != expected or rewards.shape != expected:
        raise ValueError("utility action/reward tensors do not match agent axis")
    if active.shape != expected or hidden.shape != expected:
        raise ValueError("utility masks do not match agent axis")
    if eligible.shape != (batch, length) or resets.shape != (batch, length):
        raise ValueError("utility transition masks do not match replay sequence")

    horizon = int(horizon)
    usable = length - horizon
    target_active = active[:, 1 : usable + 1]
    target_hidden = hidden[:, :usable] & target_active
    hidden_count = target_hidden.sum(axis=-1)
    action_onehot = jax.nn.one_hot(
        actions[:, 1 : usable + 1].astype(jnp.int32), action_count, dtype=f32
    )
    action_target = (action_onehot * target_hidden[..., None].astype(f32)).sum(
        axis=-2
    ) / jnp.maximum(hidden_count[..., None], 1)

    active_weight = active.astype(f32)
    team_reward = (rewards * active_weight).sum(axis=-1) / jnp.maximum(
        active_weight.sum(axis=-1), 1
    )
    return_target = jnp.zeros((batch, usable), f32)
    future_valid = jnp.ones((batch, usable), bool)
    discount = 1.0
    for offset in range(1, horizon + 1):
        return_target += discount * team_reward[:, offset : offset + usable]
        future_valid &= ~resets[:, offset : offset + usable]
        discount *= 0.99

    valid = (eligible[:, :usable] & (hidden_count > 0) & future_valid).astype(f32)
    split = batch // 2
    train_weight = valid.at[split:].set(0).reshape(-1)
    test_weight = valid.at[:split].set(0).reshape(-1)
    action_target = sg(action_target.reshape((-1, action_count)))
    return_target = sg(return_target.reshape((-1, 1)))

    action_prior = _weighted_probe_mean(action_target, train_weight)
    prior_class = jnp.argmax(action_prior)
    target_class = jnp.argmax(action_target, axis=-1)
    metrics = {
        "agent_jepa/probe/valid_fraction": valid.mean(),
        "agent_jepa/probe/hidden_action_prior_accuracy": _weighted_probe_mean(
            (target_class == prior_class).astype(f32)[:, None], test_weight
        )[0],
    }
    for name, values in sources.items():
        if values.shape[:2] != (batch, length):
            raise ValueError(f"utility source {name!r} has incompatible shape")
        features = sg(f32(values[:, :usable]).reshape((batch * usable, -1)))
        prediction = _ridge_probe(
            features,
            jnp.concatenate([action_target, return_target], axis=-1),
            train_weight,
            ridge=float(ridge),
        )
        action_prediction = prediction[:, :action_count]
        return_prediction = prediction[:, action_count:]
        metrics[f"agent_jepa/probe/{name}_hidden_action_r2"] = _probe_r2(
            action_prediction, action_target, train_weight, test_weight
        )
        metrics[f"agent_jepa/probe/{name}_hidden_action_accuracy"] = (
            _weighted_probe_mean(
                (jnp.argmax(action_prediction, axis=-1) == target_class).astype(f32)[
                    :, None
                ],
                test_weight,
            )[0]
        )
        metrics[f"agent_jepa/probe/{name}_return8_r2"] = _probe_r2(
            return_prediction, return_target, train_weight, test_weight
        )
    return metrics


def team_slot_regularization(slots, valid, *, target_std=0.1):
    """VICReg-style variance plus slot decorrelation for the online encoder."""

    if slots.ndim != 4 or valid.shape != slots.shape[:2]:
        raise ValueError(
            f"expected slots [B,T,K,D] and valid [B,T], got {slots.shape} and "
            f"{valid.shape}"
        )
    slots = f32(slots)
    weight = valid.astype(f32)
    count = jnp.maximum(weight.sum(), 1)
    mean = (slots * weight[..., None, None]).sum(axis=(0, 1)) / count
    centered = slots - mean
    variance = (jnp.square(centered) * weight[..., None, None]).sum(axis=(0, 1)) / count
    std = jnp.sqrt(variance + 1e-4)
    variance_loss = jax.nn.relu(f32(target_std) - std).mean()
    normalized = centered / jnp.maximum(
        jnp.linalg.norm(centered, axis=-1, keepdims=True), 1e-6
    )
    gram = jnp.einsum("btkd,btld->btkl", normalized, normalized)
    count_slots = slots.shape[-2]
    off_diagonal = 1.0 - jnp.eye(count_slots, dtype=gram.dtype)
    covariance_loss = (
        jnp.square(gram) * off_diagonal[None, None] * weight[..., None, None]
    ).sum() / jnp.maximum(weight.sum() * count_slots * max(count_slots - 1, 1), 1)
    metrics = {
        "agent_jepa/online_slot_std": std.mean(),
        "agent_jepa/online_effective_rank": team_slot_effective_rank(slots, weight),
        "agent_jepa/variance_penalty": variance_loss,
        "agent_jepa/covariance_penalty": covariance_loss,
    }
    return variance_loss, covariance_loss, metrics


def team_slot_effective_rank(slots, weight):
    """Effective rank across the small slot axis (at most an 8x8 eigensolve)."""

    slots = sg(f32(slots))
    weight = f32(weight)
    count = jnp.maximum(weight.sum(), 1)
    mean = (slots * weight[..., None, None]).sum(axis=(0, 1)) / count
    centered = (slots - mean) * jnp.sqrt(weight[..., None, None])
    matrix = centered.transpose(2, 0, 1, 3).reshape((slots.shape[2], -1))
    gram = matrix @ matrix.T / jnp.maximum(matrix.shape[-1], 1)
    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(gram), 0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / jnp.maximum(total, 1e-12)
    entropy = -(probabilities * jnp.log(jnp.maximum(probabilities, 1e-12))).sum()
    return jnp.where(total > 1e-12, jnp.exp(entropy), 0.0)


def _weighted_std(values, weight):
    values = sg(f32(values))
    weight = f32(weight)
    count = jnp.maximum(weight.sum(), 1)
    expanded = weight.reshape((*weight.shape, *((1,) * (values.ndim - 2))))
    mean = (values * expanded).sum(axis=(0, 1)) / count
    variance = (jnp.square(values - mean) * expanded).sum(axis=(0, 1)) / count
    return jnp.sqrt(jnp.maximum(variance, 0)).mean()


def _weighted_member_std(values, active, transition_weight):
    values = sg(f32(values))
    weight = active.astype(f32) * transition_weight[..., None]
    count = jnp.maximum(weight.sum(), 1)
    mean = (values * weight[..., None]).sum(axis=(0, 1, 2)) / count
    variance = (jnp.square(values - mean) * weight[..., None]).sum(
        axis=(0, 1, 2)
    ) / count
    return jnp.sqrt(jnp.maximum(variance, 0)).mean()


def _center_team_content(prediction, target, active):
    """Remove common team content so set matching identifies distinct agents."""

    active_weight = active.astype(f32)
    target_mean = (target * active_weight[..., None]).sum(axis=-2) / jnp.maximum(
        active_weight.sum(axis=-1, keepdims=True), 1
    )
    target = target - target_mean[..., None, :]
    prediction = prediction - prediction.mean(axis=-2, keepdims=True)
    return prediction, target


def _weighted_probe_mean(values, weight):
    values = f32(values)
    weight = f32(weight)
    return (values * weight[:, None]).sum(axis=0) / jnp.maximum(weight.sum(), 1)


def _ridge_probe(features, targets, train_weight, *, ridge):
    """Weighted dual-form ridge fit with predictions for every row."""

    features = f32(features)
    targets = f32(targets)
    train_weight = f32(train_weight)
    feature_mean = _weighted_probe_mean(features, train_weight)
    centered = features - feature_mean
    feature_rms = jnp.sqrt(
        _weighted_probe_mean(jnp.square(centered), train_weight).mean() + 1e-6
    )
    centered = centered / feature_rms
    target_mean = _weighted_probe_mean(targets, train_weight)
    target_centered = targets - target_mean
    root_weight = jnp.sqrt(train_weight)
    train_features = centered * root_weight[:, None]
    train_targets = target_centered * root_weight[:, None]
    kernel = train_features @ train_features.T
    scale = jnp.maximum(jnp.trace(kernel) / kernel.shape[0], 1.0)
    kernel += f32(ridge) * scale * jnp.eye(kernel.shape[0], dtype=f32)
    coefficients = jnp.linalg.solve(kernel, train_targets)
    return centered @ train_features.T @ coefficients + target_mean


def _probe_r2(prediction, target, train_weight, test_weight):
    baseline = _weighted_probe_mean(target, train_weight)
    error = _weighted_probe_mean(jnp.square(prediction - target), test_weight).mean()
    variance = _weighted_probe_mean(jnp.square(target - baseline), test_weight).mean()
    return 1.0 - error / jnp.maximum(variance, 1e-8)


__all__ = [
    "AgentContextEncoder",
    "TeamActionConditioner",
    "TeamContentPredictor",
    "TeamSlotEncoder",
    "TeamSlotPredictor",
    "mask_active_agents",
    "masked_agent_coverage_loss",
    "scale_gradient",
    "team_set_matching_loss",
    "team_slot_effective_rank",
    "team_slot_jepa_loss",
    "team_slot_regularization",
    "team_utility_probe_metrics",
]
