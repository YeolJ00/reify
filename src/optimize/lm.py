"""Levenberg-Marquardt on the trajectory residual, with an FD Jacobian.

Built for the ill-conditioned theta landscape (cond ~3e5): per-direction damping
lets stiff directions (mass, damping) take tiny steps while sloppy directions
(wind forcing) take large ones. Uses only forward rollouts (16 per Jacobian for
8 params), so it works with any solver, including non-differentiable VBD.
"""

import numpy as np


def lm_recover(
    residual_fn,
    theta0: np.ndarray,
    fd_h: np.ndarray,
    iters: int = 25,
    lam0: float = 1.0e-2,
    lam_max: float = 1.0e8,
    tol: float = 1.0e-10,
    verbose: bool = True,
):
    """Minimize ||r(theta)||^2. residual_fn returns the flattened residual vector
    (may contain non-finite entries if the sim explodes — treated as rejection).
    Returns (theta, history)."""
    theta = np.asarray(theta0, dtype=np.float64).copy()
    n = theta.size
    lam = lam0
    history = []

    def loss_of(r):
        return float(r @ r) if np.isfinite(r).all() else np.inf

    r = residual_fn(theta)
    L = loss_of(r)

    for it in range(iters):
        # FD Jacobian, central differences
        J = np.zeros((r.size, n))
        ok = True
        for i in range(n):
            e = np.zeros(n)
            e[i] = fd_h[i]
            rp = residual_fn(theta + e)
            rm = residual_fn(theta - e)
            if not (np.isfinite(rp).all() and np.isfinite(rm).all()):
                ok = False
                break
            J[:, i] = (rp - rm) / (2 * fd_h[i])
        if not ok:
            if verbose:
                print(f"  lm iter {it:3d}: Jacobian hit unstable region, shrinking fd_h")
            fd_h = fd_h * 0.5
            continue

        JtJ = J.T @ J
        g = J.T @ r
        d = np.diag(JtJ).copy()
        d[d <= 0.0] = 1.0e-12

        accepted = False
        while lam <= lam_max:
            try:
                delta = np.linalg.solve(JtJ + lam * np.diag(d), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r_new = residual_fn(theta + delta)
            L_new = loss_of(r_new)
            if L_new < L:
                theta = theta + delta
                r, L = r_new, L_new
                lam = max(lam / 3.0, 1.0e-12)
                accepted = True
                break
            lam *= 10
        history.append({"iter": it, "loss": L, "lam": lam, "step": float(np.linalg.norm(delta))})
        if verbose:
            print(f"  lm iter {it:3d}  loss={L:.6e}  lam={lam:.1e}  |step|={history[-1]['step']:.3e}")
        if not accepted:
            if verbose:
                print("  lm: no acceptable step (lam_max reached), stopping")
            break
        if L < tol:
            if verbose:
                print("  lm: converged (loss < tol)")
            break

    return theta, history
