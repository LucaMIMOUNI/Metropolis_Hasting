# Metropolis–Hastings sampling

An animated, from-scratch implementation of the Metropolis–Hastings (MH) algorithm,
sampling a two-dimensional density that is only known up to a constant.

![Metropolis–Hastings walking up a rough surface](media/MH_algo_demo.gif)

The chain starts in a corner of the domain, proposes a Gaussian step at every
iteration, and either accepts it or stays where it is. The side panel shows the
acceptance test being evaluated at each step.

---

## The problem

We want samples from a probability density $\pi$ on $\mathbb{R}^d$. In almost every
interesting case $\pi$ is only known up to its normalising constant:

$$\pi(x) = \frac{\tilde{\pi}(x)}{Z}, \qquad Z = \int_{\mathbb{R}^d} \tilde{\pi}(x)\,\mathrm{d}x$$

and $Z$ is an intractable integral. Direct sampling (inversion, rejection) is out of
reach, so we build a **Markov chain** $(X_n)_{n \ge 0}$ whose transition kernel
$P(x, \cdot)$ has $\pi$ as its **invariant** (stationary) distribution:

$$\int \pi(x)\,P(x, A)\,\mathrm{d}x = \pi(A) \qquad \text{for every measurable } A,$$

which in the finite-state case is exactly the eigenvector equation $\pi P = \pi$ for the
transition matrix $P$. If, in addition, the chain is irreducible and aperiodic, the
Markov chain convergence theorem gives

$$\lVert P^n(x, \cdot) - \pi \rVert_{\mathrm{TV}} \xrightarrow[n \to \infty]{} 0
\quad \text{for } \pi\text{-almost every } x,$$

so after a burn-in the states $X_n$ are (correlated) draws from $\pi$, and the ergodic
theorem lets us estimate expectations by time averages:

$$\frac{1}{N}\sum_{n=1}^{N} f(X_n) \xrightarrow[N \to \infty]{\text{a.s.}} \mathbb{E}_{\pi}[f].$$

The whole difficulty is **constructing** a $P$ that admits the $\pi$ we want. That is
what Metropolis–Hastings does.

## Detailed balance

Invariance is hard to impose directly; a stronger and much easier condition is
**detailed balance** (reversibility):

$$\pi(x)\,P(x, y) = \pi(y)\,P(y, x) \qquad \forall x, y.$$

It implies invariance — integrate both sides over $x$:

$$\int \pi(x) P(x,y)\,\mathrm{d}x = \int \pi(y) P(y,x)\,\mathrm{d}x = \pi(y) \underbrace{\int P(y,x)\,\mathrm{d}x}_{=\,1} = \pi(y).$$

So any kernel satisfying detailed balance with respect to $\pi$ leaves $\pi$ invariant.
MH is a recipe for building one.

## The algorithm

Pick a **proposal** density $q(y \mid x)$ that is easy to sample. From the current state
$x$, draw $y \sim q(\cdot \mid x)$ and accept it with probability

$$\alpha(x, y) = \min\left(1,\; \frac{\pi(y)\,q(x \mid y)}{\pi(x)\,q(y \mid x)}\right).$$

If accepted, $X_{n+1} = y$; otherwise $X_{n+1} = x$ — the chain **stays**, and the
current state is recorded a second time. The resulting kernel is

$$P(x, \mathrm{d}y) = \underbrace{q(y \mid x)\,\alpha(x, y)\,\mathrm{d}y}_{\text{move}}
\;+\; \underbrace{r(x)\,\delta_x(\mathrm{d}y)}_{\text{stay}},
\qquad r(x) = 1 - \int q(y \mid x)\,\alpha(x, y)\,\mathrm{d}y.$$

Two things make this work.

**1. The normalising constant cancels.** $\alpha$ only ever sees the *ratio*
$\pi(y)/\pi(x) = \tilde{\pi}(y)/\tilde{\pi}(x)$, so $Z$ is never needed. This is the
reason MH is usable at all.

**2. Detailed balance holds by construction.** For $x \neq y$,

$$\pi(x) P(x,y) = \pi(x)\, q(y \mid x) \min\left(1, \frac{\pi(y) q(x \mid y)}{\pi(x) q(y \mid x)}\right)
= \min\Big(\pi(x) q(y \mid x),\; \pi(y) q(x \mid y)\Big),$$

using $a \min(1, b/a) = \min(a, b)$ for $a > 0$. The right-hand side is **symmetric** in
$x$ and $y$, so swapping them gives $\pi(y) P(y,x)$ — the same quantity. The $x = y$ case
is trivial. Hence $\pi$ is invariant. $\blacksquare$

## Symmetric proposals: the Metropolis case

This implementation uses an isotropic Gaussian random walk,

$$y = x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I),$$

which is symmetric: $q(y \mid x) = q(x \mid y)$. The Hastings ratio
$q(x \mid y) / q(y \mid x)$ collapses to $1$ and the test reduces to the original
Metropolis form used in the code:

$$\alpha(x, y) = \min\left(1, \frac{\pi(y)}{\pi(x)}\right).$$

Read plainly: **uphill moves are always accepted; downhill moves are accepted with
probability equal to the density ratio.** That second clause is what separates sampling
from hill climbing — the chain is allowed to go down, which is how it escapes a local
mode and explores the whole distribution.

## The target used here

The demo samples

$$\pi(x, y) \;\propto\; \exp\big(f(x, y)\big), \qquad
f(x, y) = -\big(x^2 + y^2 + 10 \sin x \cos y\big),$$

a Boltzmann density at unit temperature built from a classic multi-modal optimisation
test surface. It is a fair test case: multi-modal, unnormalised, and with a $Z$ nobody
wants to compute. Note that with $\alpha = \min(1, e^{f(y) - f(x)})$ this is also exactly
the acceptance rule of **simulated annealing** at temperature $T = 1$; annealing simply
lets $T \to 0$ so the chain concentrates on the global maximiser instead of sampling.

A numerical remark: the code evaluates $e^{f}$ directly, which is safe on this surface
($f \in [-60, 10]$ roughly). In general the ratio should be taken in log space —
accept when $\log u < \log \tilde{\pi}(y) - \log \tilde{\pi}(x)$ with $u \sim \mathcal{U}(0,1)$ —
to avoid overflow and underflow.

## Tuning

The proposal scale $\sigma$ (`proposal_std`) is the one real knob, and it is a trade-off:

- **$\sigma$ too small** — almost everything is accepted, but the chain crawls. Successive
  samples are strongly autocorrelated and the effective sample size collapses.
- **$\sigma$ too large** — proposals land in the tails, $\pi(y)/\pi(x)$ is tiny, almost
  everything is rejected and the chain sits still.

The usual target is an acceptance rate around $0.44$ in one dimension and $\approx 0.234$
as $d \to \infty$ for random-walk MH (Roberts, Gelman & Gilks, 1997). The defaults here
are $\sigma = 0.5$ over a domain of size $10$, with $400$ iterations.

Two standard caveats apply to the output: the first samples are **not** from $\pi$ (discard
a burn-in), and consecutive samples are **correlated** — $N$ MH samples carry less
information than $N$ independent ones.

## Running it

```bash
pip install numpy matplotlib
python main.py
```

`main.py` contains:

| function | role |
| --- | --- |
| `surface(x, y)` | the log-density $f$, up to an additive constant |
| `metropolis_hastings(initial_state, iterations, proposal_std)` | the chain; returns the samples and a per-step record of the acceptance test |
| `animate_MH(...)` | 3D animation of the walk with the live acceptance panel |

Parameters live at the top of `main()`: `initial_state`, `iterations`, `proposal_std`.

## References

- N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller,
  *Equation of State Calculations by Fast Computing Machines*, J. Chem. Phys. 21 (1953).
- W. K. Hastings, *Monte Carlo sampling methods using Markov chains and their
  applications*, Biometrika 57 (1970).
- G. O. Roberts, A. Gelman, W. R. Gilks, *Weak convergence and optimal scaling of random
  walk Metropolis algorithms*, Ann. Appl. Probab. 7 (1997).
- C. P. Robert, G. Casella, *Monte Carlo Statistical Methods*, Springer (2004).
