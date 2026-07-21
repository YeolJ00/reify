"""Cross-entropy method: zeroth-order fallback for when gradients are unusable."""

import numpy as np


def cem_nd(
    loss_fn,
    mu0: np.ndarray,
    sigma0: np.ndarray,
    pop_size: int = 32,
    elite_frac: float = 0.25,
    iters: int = 25,
    sigma_floor: float = 1.0e-4,
    noise0: np.ndarray | None = None,
    seed: int = 42,
    verbose: bool = True,
):
    """Diagonal-Gaussian CEM over an n-dim vector. Returns (mu, history)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu0, dtype=np.float64).copy()
    sigma = np.asarray(sigma0, dtype=np.float64).copy()
    if noise0 is None:
        noise0 = np.asarray(sigma0) / 4.0
    n_elite = max(2, int(round(pop_size * elite_frac)))
    history = []
    best_ever = (np.inf, mu.copy())

    for it in range(iters):
        samples = rng.normal(mu, sigma, size=(pop_size, mu.size))
        losses = np.array([loss_fn(s) for s in samples])
        order = np.argsort(losses)  # NaN (exploded sims) sorts last
        elite = samples[order[:n_elite]]
        noise = noise0 * max(0.0, 1.0 - it / max(1, iters - 2))
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0) + noise, sigma_floor)
        best = float(losses[order[0]])
        if np.isfinite(best) and best < best_ever[0]:
            best_ever = (best, samples[order[0]].copy())
        history.append({"iter": it, "mu": mu.copy(), "sigma": sigma.copy(), "best_loss": best})
        if verbose:
            print(f"  cem iter {it:3d}  best_loss={best:.6e}  max_sigma={sigma.max():.4f}")

    # return the best sample ever seen, not the final mean — with sloppy loss
    # directions the mean can drift along the flat valley
    return best_ever[1], history


def cem_1d(
    loss_fn,
    mu0: float,
    sigma0: float,
    pop_size: int = 16,
    elite_frac: float = 0.25,
    iters: int = 20,
    sigma_floor: float = 1.0e-3,
    noise0: float | None = None,
    seed: int = 42,
    lower: float | None = None,
    upper: float | None = None,
    verbose: bool = True,
):
    """Minimize a scalar->scalar black-box loss with CEM. Returns (mu, history).

    Decaying additive noise (noise0, linearly annealed to 0) prevents premature
    variance collapse; defaults to sigma0 / 4.
    """
    rng = np.random.default_rng(seed)
    mu, sigma = float(mu0), float(sigma0)
    if noise0 is None:
        noise0 = sigma0 / 4.0
    n_elite = max(2, int(round(pop_size * elite_frac)))
    history = []

    for it in range(iters):
        samples = rng.normal(mu, sigma, size=pop_size)
        if lower is not None or upper is not None:
            samples = np.clip(samples, lower, upper)
        losses = np.array([loss_fn(float(s)) for s in samples])
        elite = samples[np.argsort(losses)[:n_elite]]
        noise = noise0 * max(0.0, 1.0 - it / max(1, iters - 2))
        mu = float(elite.mean())
        sigma = max(float(elite.std()) + noise, sigma_floor)
        best = float(losses.min())
        history.append({"iter": it, "mu": mu, "sigma": sigma, "best_loss": best})
        if verbose:
            print(f"  cem iter {it:3d}  mu={mu:9.4f}  sigma={sigma:8.4f}  best_loss={best:.6e}")

    return mu, history
