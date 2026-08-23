# DreaMARL B3 Implementation Plan

Written 2026-08-22. Companion to `EXPERIMENT_LEDGER.md`. Every external claim and
codebase claim below was re-verified against primary sources (papers, `oxwhirl/smac`
source, this repo) on that date; corrections vs. earlier drafts are marked.

---

## 1. Verification record

### 1.1 External claims

| Claim | Verdict | Source |
| --- | --- | --- |
| MATWM exists; decentralized focal world model on STORM; DreamerV3-style training | Verified | arXiv:2506.18537 |
| MATWM teammate predictor: transformer head on **stop-gradient latents** predicts all teammates' actions; logits enter **actor and critic state** (`s_t = [z_t, h_t, p̂_{t,i}]`) | Verified | §3.1, Eq. 8 |
| MATWM PER: exponential recency weighting for **world-model batches only**; agent-training starts sampled uniformly | Verified | §3 step 2/3 |
| MATWM Melting Pot Table 4 = Chicken 21.5(2.4), Coop Mining 19.0(4.1), Externality Dense 146.8(18.5), Gift 75.0(25.1), Pure 6.8(0.7), Rationalizable 12.2(0.5), Stag Hunt 7.2(2.5); ablations −TP 130.0 / −PER 133.5 / −AS 135.7 | Verified exactly | Table 4, Table 5 |
| MATWM aggregation convention ambiguous: "mean win rate or mean reward of all agents"; baselines near-random so no scale anchor | Unresolved → parity task T0.6 mandatory | §4 |
| MARIE: local transformers + Perceiver tokens inserted into each local sequence; VQ-VAE; MAPPO-like PPO (epochs=5, ε=0.2); critic consumes other agents' reconstructed obs; actors consume only own reconstructed obs; execution without world model | Verified | arXiv:2406.15836 v2 Tables 14–15, Fig. 2 |
| MARIE numbers quoted as "MAMBA 87.7 / MARIE 99.5 on 3m" and "MAMBA 29.3 / MARIE 73.0 on 3s_vs_4z" | **Corrected**: v1 Table 1 reads 3m MARIE 99.5(0.4), MAMBA 86.4(7.1); 3s_vs_4z MARIE 63.6(24.9), MAMBA 27.7(12.3) | arXiv:2406.15836v1 Table 1 |
| MAMBA: DreamerV2-RSSM multi-agent extension; attention across agents sharing discrete stochastic states (~160 bits) during execution; PPO actor on imagined rollouts; gradients **not** backpropagated through dynamics | Verified | Egorov & Shpilman, AAMAS 2022, arXiv:2205.15023 |
| MABL: bi-level latents, global teacher informs local student during training; execution uses lower level only | Verified | Venugopal et al., AAMAS 2024, arXiv:2304.06011 |
| COMA: centralized all-action Q(s, a_{-i}, ·), counterfactual baseline marginalizing focal action with teammates fixed | Verified | Foerster et al., AAAI 2018 |
| COCOA: reward-centric counterfactual contribution beats crediting future states (spurious attribution) | Verified | Meulemans et al., NeurIPS 2023 |
| SMAC shield-regeneration reward artifact: default `reward_only_positive=True`; `reward = abs(delta_enemy + delta_deaths)  # shield regeneration`; max-reward docstring excludes shield regen; applies to Protoss enemies (3s_vs_4z), not Terran 3m | **Verified verbatim** from source | oxwhirl/smac `starcraft2.py` |
| MAPPO protocol: win rate over 32 eval games per iteration, median of final ten evaluations; fewer epochs/no minibatching best on hard maps | Verified verbatim | Yu et al., NeurIPS 2022 D&B |

### 1.2 Codebase claims (audited against this repo)

| Claim | Verdict | Evidence |
| --- | --- | --- |
| `advance()` / `prior()` / `complete(..., logit=...)` injection hook exists | **Verified, unused** — clean injection point for B3 residuals | `world_model/transformer.py:497,510,519` |
| TeamAxis fold/unfold/group machinery | Verified | `marl/axes.py:21` (`fold_batch`:34, `group_starts`:72) |
| `repval_grad: True` default (replay value grads reshape representations) | Verified | `configs.yaml:138`, `training/learner.py:192` |
| `ac_grads: False` (imagination features stop-gradiented) | Verified | `training/learner.py:121-122` |
| Recency replay `0.9998^age`, inverse-CDF sampler | Verified | `replay.py:9,72`, `configs.yaml:39` |
| Score-function actor loss, no PPO ratio/clip anywhere | Verified | `training/objectives.py:46-50`; repo-wide grep |
| Single joint optimizer @ 4e-5 (GroupedOptimizer alternative exists) | Verified | `agent.py:103-147`, `configs.yaml:107` |
| Grouped synchronized imagination exists today (independent per agent) | Verified | `marl/core.py:730-736` |
| Peer-history intervention test pattern exists | Verified | `tests/test_dreamarl_marl.py:194` |
| Eval logs `per_agent_return_mean` + `team_return_mean` | Verified | `evaluation.py:112,116` |
| **Active-agent team-reward training path exists** | **FALSE — corrected.** Active-agent mean survives only inside a diagnostic probe (`models/team.py:613`). `learner.py:564` trains folded raw rewards; F-JEPA is docs-only. T0.1 is an implementation task, not a rerun. | audit 2026-08-22 |

### 1.3 Consequences of the audit

1. Stage 0's "B0-recent-team" arm **cannot be launched today**; build T0.1 first behind a flag.
2. The `complete(logit=)` hook means Phase 2 needs **no changes to the local Transformer**.
3. `baselines/marie/` already in-tree — usable for protocol-parity probes.
4. All published SMAC comparisons must note MARIE used 10-game evals / 4 seeds vs our fixed-checkpoint evals.

---

## 2. Frozen substrate decisions

Unchanged from B0 + exact recent replay, plus two pending upgrades gated by Phase 0/1a:

- Local JEPA encoder/stochastic state/causal Transformer: untouched.
- Strict decentralized execution: untouched (peer tensors never reach `pi(a|s_t^i)`).
- Exact recent replay (`recency_decay=0.9998`): untouched until Phase 1a's anchor experiment.
- Joint optimizer @4e-5, train ratio 256, horizon 15, λ=0.95: untouched.
- New config flags (all default-off ⇒ bit-exact current behavior):
  - `team_reward_mode ∈ {individual, active_mean, active_sum}` (T0.1)
  - `actor_update ∈ {score, ppo}`, `ppo.clip`, `ppo.epochs`, `ppo.kl_stop` (Phase 1a)
  - `replay_anchor_frac` (default 0)
  - `b3.*` namespace for interaction residual, CF critic, outcome JEPA
- `marl_stage` enum extended `"b3"` in `config.py` alongside b0/b1/b2; new code lives in new
  modules, never inline edits of B0 paths.

---

## 3. Phased plan

### Phase 0 — Protocol & instrumentation (no behavior change; ~1 week)

| ID | Task | Files | Done when |
| --- | --- | --- | --- |
| T0.1 | Team-reward objective: aggregate **pre-fold** on the `[B,T,A]` axis (sum or mean over active agents), feed reward-head targets, critic targets, λ-returns. Keep raw individual rewards logged. Default `individual` = historical parity. Mean-vs-sum is an ablation arm (mean halves a 2-agent gold event per agent — decide empirically). | `training/learner.py` (pre-fold site near :564), `config.py`, `configs.yaml` | unit tests + flags-off parity |
| T0.2 | SMAC dual-reward logging: legacy scaled reward stays authoritative; add parallel components `r_damage=Σ max(ΔHP+Δshield,0)`, enemy deaths, ally deaths, win, timeout, survivors. Do not change returned reward yet. | `envs/smac.py`, `diagnostics.py` | both series logged on one episode |
| T0.3 | Eval upgrade: `curve_eval_eps` 20→32 for SMAC runs; report median-of-last-N fixed evals and peak-to-final drop (MAPPO protocol precedent). | `configs.yaml:53`, `evaluation.py`, `train.py:154-172` | new metrics in eval JSON |
| T0.4 | Policy diagnostics: KL vs update-start snapshot, KL vs best checkpoint, entropy per agent, prob(selected action), grad norms, replay age histogram. | `training/reporting.py`, `learner.py` | logged every update |
| T0.5 | Fixed reference batch: freeze K batches at first healthy 3m checkpoint (~50k); log JEPA cosines/critic EV on it each eval (current-distribution metrics hide forgetting). | `replay.py`, `evaluation.py` | anchor batch persisted |
| T0.6 | MATWM parity attempt: clone `azaddeihim/matwm`, extract Melting Pot adapter + aggregation convention into `docs/dreamarl/matwm_parity.md`. If unreproducible, quoted numbers are declared non-comparable targets. | docs | written verdict |
| T0.7 | Success-reservoir replay hook: small reservoir of top episodes (wins for 3m; corrected-outcome-ranked otherwise), sampled with weight `replay_anchor_frac` (default 0). | `replay.py` (`RecentReplay` sibling class) | sampling test |

**Gate G0:** all flags off ⇒ metrics-equal to current B0 on a short 3m smoke + DMC A=1 smoke; pytest green.

**Stage 0 controls (after G0):** arms A = B0-recent-individual (historical parity, Externality target ≈46.83), B = same with `team_reward_mode` ∈ {active_mean, active_sum}. Isolates the reward correction from all architecture work.

### Phase 1a — SMAC stabilization track (cheap; parallel with Phase 1b)

Matched continuations from a healthy ~50k 3m checkpoint, 25k steps, 2 seeds each:

| Arm | Change |
| --- | --- |
| A | control |
| B | actor frozen (does drift initiate model collapse?) |
| C | world model frozen (do actor/critic alone lose coordination?) |
| D | `repval_grad=False` (value→latent interference?) |
| E | imagined-PPO (clip {0.1,0.2} × epochs {1,3}, early-stop KL≈0.01–0.03) |
| F | E + `repval_grad=False` |

Readouts: peak-to-final win drop, reference-batch JEPA/EV, entropy trajectory.
Decision rules: E/F fix both maps → adopt proximal actor as substrate; only F fixes seed-123 mode → value-interference confirmed; nothing fixes → continual-learning problem first (anchor replay), everything else waits.
Then E1: anchor replay 0.9/0.1 on the stabilized config. Promotion: late median ≥80%, drop ≤30pp.

### Phase 1b — Frozen counterfactual audit (zero env steps)

Data: retained B0+recent Externality/Coop-Mining checkpoints + replay snapshots; recompute grouped posterior states offline. Train an all-action critic `Q_i(S,a_{-i},·)` (set attention over per-agent tokens `[sg(s_j), a_j]`, focal mask token) on realized team returns (T0.1 aggregation recomputed offline).

Metrics:
- M1 policy-weighted counterfactual baseline ≈ 0
- M2 causative actions rank above alternatives on held-out cooperative events (AUC); peer-action shuffling destroys the margin
- M3 blue-mushroom causer gets positive A_cf despite zero own reward; uninvolved agents' gaps materially smaller
- M4 gold-producing actions/precursors credited above iron collection
- M5 privileged controller (argmax-Q focal actions, training-only oracle) beats B0 return

**Gate:** M5 pass AND ≥2 of {M2,M3,M4} strong → CF signal exists → Phase 3. M5 fail → local observability problem → roles path (Phase 4c) before any new world-model machinery.

### Phase 2 — Minimal joint interaction residual (gated, not default)

Pre-gate probe (cheap, frozen): does grouped true peer state/action improve multi-step prediction of outcome-relevant quantities (enemy health/shield deltas, kill/survival/win, Coop-Mining team events) vs shuffled peers? MARIE's own ablation shows aggregation matters more as agent count grows and is negligible at 3 agents — a null result here is informative, not failure.

If pass: implement `DirectedInteractionResidual` in `models/interaction.py`:
```
cache, deter = dynamics.advance(carry, local_actions)
delta_logits = interaction_model(grouped_states, joint_actions, active_mask)
carry, feat = dynamics.complete(cache, deter, logit=local_logits + g * ungroup(delta_logits))
```
with zero-init gate g (exact B0 parity at init). Heads: team reward (`team_reward_mode`), raw individual rewards, continuation, mask. One-step joint imagination at a time via existing `group_starts`/fold helpers. Actor never sees interaction latents.

### Phase 3 — Amortized counterfactual actor learning

`CounterfactualQ` in `training/counterfactual.py`: branch targets K=1 first, K=3 only if held-out ranking improves; common random numbers across branches; branch one focal agent per imagination start (coverage via parameter sharing). Complexity O(A·|A|).
Ramp CF actor coefficient 0→1 after audit gates hold on live rollouts. Keep score-function normalization S.

Factorial kill gates (Externality + Coop Mining, 2 seeds × 30k):
| Arm | Team reward | Joint transition | CF actor |
| --- | --- | --- | --- |
| C | yes | no | yes |
| D | yes | yes | no |
| E | yes | yes | yes |

C↑/D flat ⇒ credit was primary; D↑/C flat ⇒ dynamics were primary; only E ⇒ joint necessity proven; E fixes Externality but not Coop Mining ⇒ discovery/roles next (Phase 4).

### Phase 4 — Conditional extensions

- 4a Outcome JEPA `u_{t,k}`: EMA encoder over future reward windows (reward-only first; COCOA warns unrestricted state credit is spurious), action-conditioned predictor; used as compact critic feature / contribution model. Only after E passes.
- 4b Event-stratified replay: `0.8·p_recent + 0.2·p_recent∩rare-event` precursor windows. Only if attribution works but gold stays rare.
- 4c Roles ρ_t^i (MABL pattern): global teacher → locally distilled student; actor may consume student only. Only on realizability failure (audit M5 fail or distillation gap).

### Phase 5 — Paper run

7 tasks × ≥5 seeds × 100k (Coop Mining 200k), fixed 32-episode eval every 10k, dual reward conventions reported. Ablations: −CF, −recent replay, −team reward, sum-vs-mean, ±joint residual, ±outcome JEPA. Optional SMAX generality check. MATWM comparison only post-T0.6.

---

## 4. Executable test checklist (extend `tests/test_dreamarl_marl.py` patterns)

1. flags-off bit-parity (short-run metric equality)
2. A=1 numerical equivalence for every new module
3. zero-gate B0 parity for the residual (g=0)
4. permutation equivariance of interaction + Q modules (reuse :185 pattern)
5. active/inactive carry preservation through joint step
6. legal-action masking inside every CF branch
7. CRN reproducibility of branch targets
8. stop-gradient checks (actor→critic/WM; TP latents; outcome targets)
9. no-peer-history effect on online policy (extend :194 to B3 paths)
10. team-reward aggregation incl. inactive agents; sum vs mean unit tests
11. checkpoint restore with/without B3 modules
12. corrected SMAC reward decomposition matches hand-computed episode

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| CF baseline variance | CRN, K small, batch averaging, apply-at-events option |
| Actor exploits reward-head error on counterfactual latents | short branch depth, ensemble disagreement penalty, PPO trust region |
| Recent-replay forgetting (seed-223 mode) | anchor reservoir after Phase 1a diagnosis |
| Value grads corrupting latents (seed-123 mode) | `repval_grad=False` arm decides default |
| Compute creep | amortized Q keeps O(A·|A|); branches budgeted inside existing train ratio |
| Protocol drift | fixed-checkpoint eval only; dual reward conventions published |

## 6. Immediate next actions (this week)

1. T0.1 implementation + tests (blocks Stage 0 controls)
2. T0.2 SMAC dual reward logging
3. Launch Stage 0 arms A/B once G0 passes
4. Kick off Phase 1a arms A–F from the healthy 3m checkpoint
5. Start Phase 1b frozen CF audit on retained Externality checkpoints (no GPU contention)

