\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}

\hypersetup{pdfauthor={}, pdftitle={Algorithm 7 Sampling from a Factorized CTMC Model}}

\begin{document}

\setlength{\parindent}{0pt}

\textbf{Algorithm 7} Sampling from a Factorized CTMC Model (Euler / $\tau$-leaping)
\hrule
\vspace{0.5em}
\textbf{Require:} Rate network $Q_t^\theta$ (factorized), initial distribution $p_{\text{init}}$, number of steps $n$
\begin{algorithmic}[1]
\State Set $t \leftarrow 0$
\State Set step size $h \leftarrow \frac{1}{n}$
\State Draw a sample $X_0 \sim p_{\text{init}}$, where $X_0 = (X_0^{(1)}, \dots, X_0^{(d)}) \in \mathcal{V}^d$
\For{$i = 1, \dots, n$}
    \State Compute factorized jump rates $\{q_j(v)\}_{j=1..d, \, v \in \mathcal{V}} \leftarrow Q_t^\theta(\cdot \mid X_t)$
    \For{$j = 1, \dots, d$ (in parallel)}
        \State $x \leftarrow X_t^{(j)}$ \{current token at position $j$\}
        \State Define the per-position Euler transition probabilities $\tilde{p}_{j,t}(\cdot \mid X_t^{(j)} = x)$ by
        \begin{equation*}
        \tilde{p}_{j,t}(v \mid x) = \begin{cases} h q_j(v), & v \neq x, \\ 1 - h \sum\limits_{v' \in \mathcal{V} \setminus \{x\}} q_j(v'), & v = x. \end{cases}
        \end{equation*}
        \State Sample $X_{t+h}^{(j)} \sim \textsc{Categorical}(\{\tilde{p}_{j,t}(v \mid x)\}_{v \in \mathcal{V}})$
    \EndFor
    \State Set $t \leftarrow t + h$
\EndFor
\State \textbf{return} $X_1$
\end{algorithmic}
\hrule

\noindent \textbf{Algorithm 8} Training factorized CTMC Model (Discrete Diffusion) \\
\hrule
\vspace{0.5em}
\noindent \textbf{Require:} Dataset of sequences $z \sim p_{\text{data}}$ with $z = (z_1, \dots, z_d) \in \mathcal{V}^d$; \\
initial (noise) token marginals $p_{\text{init}}^{(j)}$ on $\mathcal{V}$; schedule $\kappa_t \in [0, 1]$; \\
posterior network $f_\theta$ returning per-position logits over $\mathcal{V}$; optimizer $\textsc{Opt}$

\begin{enumerate}
    \item[\textbf{1:}] \textbf{for} each training iteration \textbf{do}
    \item[\textbf{2:}] \quad Sample a data point $z \sim p_{\text{data}}$
    \item[\textbf{3:}] \quad Sample time $t \sim \text{Unif}[0, 1]$ and compute $\kappa \leftarrow \kappa_t$
    \item[\textbf{4:}] \quad Sample a noisy state $x \sim p_t(\cdot \mid z)$ (factorized mixture path):
    \item[\textbf{5:}] \quad \textbf{for} $j = 1, \dots, d$ (\textbf{in parallel}) \textbf{do}
    \item[\textbf{6:}] \quad \quad Sample mask $m_j \sim \text{Bernoulli}(\kappa)$
    \item[\textbf{7:}] \quad \quad Sample noise token $\xi_j \sim p_{\text{init}}^{(j)}$
    \item[\textbf{8:}] \quad \quad Set $x_j \leftarrow m_j z_j + (1 - m_j) \xi_j$
    \item[\textbf{9:}] \quad \textbf{end for}
    \item[\textbf{10:}] \quad $x \leftarrow (x_1, \dots, x_d)$
    \item[\textbf{11:}] \quad Predict terminal-token posteriors via logits from the network:
    \begin{equation*}
        \ell_j(\cdot) \leftarrow f_\theta(x, t)_j \quad \Rightarrow \quad p_{1|t}^\theta(v \mid x)_j = \text{Softmax}(\ell_j)(v)
    \end{equation*}
    \item[\textbf{12:}] \quad Discrete Flow Matching loss (token-wise NLL of $z$):
    \begin{equation*}
        \mathcal{L}_{\text{DFM}}(\theta) \leftarrow \sum_{j=1}^d \left[ -\log p_{1|t}^\theta(z_j \mid x)_j \right]
    \end{equation*}
    \item[\textbf{13:}] \quad Update parameters: $\theta \leftarrow \textsc{Opt.Step}(\nabla_\theta \mathcal{L}_{\text{DFM}}(\theta))$
    \item[\textbf{14:}] \textbf{end for}
\end{enumerate}
\hrule


\end{document}
