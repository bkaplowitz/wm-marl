Let us begin by characterizing our state space $S$. Let $\mathcal{V}=\left\{v_1, \cdots, v_V\right\}$ be our vocabulary. The state space is given by $S=\mathcal{V}^d$ where $d \in \mathbb{N}$ is sequence length and $V \in \mathbb{N}$ is the vocabulary size. For language, $\left\{v_1, \cdots, v_V\right\}$ could enumerate our alphabet or a set of discrete tokens and $S$ would represent the set of sequences (or sentences) of length $d$. For DNA, $\left\{v_1, \cdots, v_V\right\}$ could be all 4 DNA bases and $S$ all DNA sequences of length $d$.

Next, let $X_t$ be a stochastic process on $S$, i.e. a random trajectory $X:[0,1] \rightarrow S, t \mapsto X_t$ in $S$. We require $X_t$ to be a Markov process, i.e. a process that has no memory.
## Algorithm 7 — Sampling from a Factorized CTMC Model (Euler / $\tau$-leaping)

**Require:** Rate network $Q_t^\theta$ (factorized), initial distribution $p_{\text{init}}$, number of steps $n$.

**Setup:**

1. Set $t \leftarrow 0$.
2. Set step size $h \leftarrow \frac{1}{n}$.
3. Draw a sample $X_0 \sim p_{\text{init}}$, where $X_0 = (X_0^{(1)}, \dots, X_0^{(d)}) \in \mathcal{V}^d$.

**Main loop — repeat for $i = 1, \dots, n$:**

1. Compute factorized jump rates $\{q_j(v)\}_{j=1..d,\, v \in \mathcal{V}} \leftarrow Q_t^\theta(\cdot \mid X_t)$.
2. Update every position $j = 1, \dots, d$ in parallel. For each $j$, with current token $x = X_t^{(j)}$, define the per-position Euler transition probabilities $\tilde{p}_{j,t}(\cdot \mid X_t^{(j)} = x)$ by

   $$\tilde{p}_{j,t}(v \mid x) = \begin{cases} h\, q_j(v), & v \neq x, \\ 1 - h \sum\limits_{v' \in \mathcal{V} \setminus \{x\}} q_j(v'), & v = x, \end{cases}$$

   and sample $X_{t+h}^{(j)} \sim \mathrm{Categorical}\big(\{\tilde{p}_{j,t}(v \mid x)\}_{v \in \mathcal{V}}\big)$.
3. Set $t \leftarrow t + h$.

**Return** $X_1$.

---

## Algorithm 8 — Training a Factorized CTMC Model (Discrete Diffusion)

**Require:** Dataset of sequences $z \sim p_{\text{data}}$ with $z = (z_1, \dots, z_d) \in \mathcal{V}^d$; initial (noise) token marginals $p_{\text{init}}^{(j)}$ on $\mathcal{V}$; schedule $\kappa_t \in [0, 1]$; posterior network $f_\theta$ returning per-position logits over $\mathcal{V}$; optimizer $\mathrm{Opt}$.

**Repeat for each training iteration:**

1. Sample a data point $z \sim p_{\text{data}}$.
2. Sample time $t \sim \mathrm{Unif}[0, 1]$ and compute $\kappa \leftarrow \kappa_t$.
3. Sample a noisy state $x \sim p_t(\cdot \mid z)$ (factorized mixture path). For each position $j = 1, \dots, d$ in parallel:
    1. Sample mask $m_j \sim \mathrm{Bernoulli}(\kappa)$.
    2. Sample noise token $\xi_j \sim p_{\text{init}}^{(j)}$.
    3. Set $x_j \leftarrow m_j z_j + (1 - m_j) \xi_j$.

   Then set $x \leftarrow (x_1, \dots, x_d)$.
4. Predict terminal-token posteriors via the network logits:

   $$\ell_j(\cdot) \leftarrow f_\theta(x, t)_j \quad \Rightarrow \quad p_{1|t}^\theta(v \mid x)_j = \mathrm{Softmax}(\ell_j)(v).$$
5. Compute the Discrete Flow Matching loss (token-wise NLL of $z$):

   $$\mathcal{L}_{\text{DFM}}(\theta) \leftarrow \sum_{j=1}^d \left[ -\log p_{1|t}^\theta(z_j \mid x)_j \right].$$
6. Update parameters: $\theta \leftarrow \mathrm{Opt.Step}\big(\nabla_\theta \mathcal{L}_{\text{DFM}}(\theta)\big)$.
