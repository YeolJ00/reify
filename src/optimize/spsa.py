"""SPSA: gradient-scaling optimisation without a usable gradient.

Why this and not CEM, and not finite differences:

    method              evaluations per step     scales with parameter count?
    CEM                 population ~ O(d)        no  -- population must grow with d
    finite differences  2d                       no  -- linear in d, per step
    SPSA                2                        YES -- independent of d
    true gradient       1 fwd + 1 bwd            yes -- but needs differentiability

SPSA perturbs EVERY parameter at once with a random +-1 vector and forms
    g_hat_i = (f(x + c*delta) - f(x - c*delta)) / (2*c*delta_i)
which is a noisy but unbiased-in-direction estimate of the gradient, at a cost of two
objective evaluations no matter how many parameters there are. That is the property this
project needs: the objective costs a simulation, a Cycles render and a judge call, so the
count of evaluations is the entire budget.

It is the right backup here specifically because the measured gradient through the judge
is not trustworthy -- ds/dpixels is real and structured, but the scalar it differentiates
is ~85% motion magnitude, and the judge has no arrow of time (at chance on real filmed
video). SPSA needs only the scalar objective, so it is agnostic to whether that objective
is reached by backprop, by a black-box renderer, or by anything else. Swap in a better
judge and this code is unchanged.

Noise. The objective here is noisy (prompt spread, judge stochasticity). SPSA tolerates
that by design -- the gain sequences average it out over iterations -- but `n_avg` allows
several perturbation pairs per step when the budget allows, which reduces the variance of
each step rather than relying on the schedule alone.

Gain sequences follow Spall's standard recommendation: a_k = a/(A+k+1)^alpha with
alpha=0.602, c_k = c/(k+1)^gamma with gamma=0.101. c should be roughly the noise standard
deviation of the objective, and A about 10% of the planned iteration count.
"""
import numpy as np


class SPSA:
    def __init__(self, objective, x0, bounds, a=0.15, c=0.10, alpha=0.602, gamma=0.101,
                 A=None, n_iter=40, n_avg=1, seed=0, maximize=True, log=print):
        """objective(x) -> float. Called 2 * n_avg times per iteration.

        bounds is (d, 2). x is in whatever space the caller chooses -- use log-parameters
        so a multiplicative perturbation is a fixed additive step and c has one meaning
        across parameters with different units.
        """
        self.f = objective
        self.x = np.clip(np.asarray(x0, float), bounds[:, 0], bounds[:, 1])
        self.bounds = np.asarray(bounds, float)
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma
        self.A = (0.1 * n_iter) if A is None else A
        self.n_iter, self.n_avg = int(n_iter), int(n_avg)
        self.rng = np.random.default_rng(seed)
        self.sign = 1.0 if maximize else -1.0
        self.log = log
        self.history = []

    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def step(self, k):
        a_k = self.a / (self.A + k + 1) ** self.alpha
        c_k = self.c / (k + 1) ** self.gamma
        d = len(self.x)
        ghat = np.zeros(d)
        evals = []
        for _ in range(self.n_avg):
            # Bernoulli +-1. Uniform or gaussian perturbations break SPSA's variance
            # argument because 1/delta_i must have bounded expectation.
            delta = self.rng.choice([-1.0, 1.0], size=d)
            xp, xm = self._clip(self.x + c_k * delta), self._clip(self.x - c_k * delta)
            yp, ym = self.f(xp), self.f(xm)
            evals += [(xp, yp), (xm, ym)]
            ghat += (yp - ym) / (2.0 * c_k * delta)
        ghat /= self.n_avg
        x_new = self._clip(self.x + self.sign * a_k * ghat)
        rec = {"iter": k, "a_k": a_k, "c_k": c_k,
               "x": self.x.tolist(), "x_new": x_new.tolist(),
               "ghat": ghat.tolist(),
               "y": [float(y) for _x, y in evals],
               "step_norm": float(np.linalg.norm(x_new - self.x))}
        self.x = x_new
        self.history.append(rec)
        return rec

    def run(self):
        for k in range(self.n_iter):
            r = self.step(k)
            if self.log:
                self.log(f"  iter {k:>3}  a_k={r['a_k']:.4f} c_k={r['c_k']:.4f}  "
                         f"y={np.mean(r['y']):+.4f}  |step|={r['step_norm']:.4f}  "
                         f"x=[{', '.join(f'{v:+.3f}' for v in r['x_new'])}]")
        return self.x, self.history


def selftest():
    """Verify on a function with a known optimum before trusting it on the real one."""
    # noisy quadratic, optimum at [1, -2, 0.5, 3, -1]; noise comparable to the real
    # objective's prompt spread
    target = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    rng = np.random.default_rng(1)

    def f(x):
        return -float(np.sum((np.asarray(x) - target) ** 2)) + rng.normal(0, 0.05)

    b = np.tile(np.array([-5.0, 5.0]), (5, 1))
    opt = SPSA(f, np.zeros(5), b, a=0.6, c=0.15, n_iter=250, seed=0, log=None)
    x, _h = opt.run()
    err = float(np.linalg.norm(x - target))
    print(f"SPSA selftest  recovered {np.round(x, 3)}")
    print(f"               target    {target}")
    print(f"               |error| = {err:.4f}   evals = {2*250}")
    print(f"               (finite differences would need {2*5*250} evals for the "
          f"same number of steps)")
    return err < 0.25


if __name__ == "__main__":
    ok = selftest()
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
