"""M5.1: does a tape gradient flow through XPBD rigid contact?

CLAUDE.md mandates a finite-difference check on every gradient, especially
through contact. We test d(final body z)/d(initial velocity) through a
drop-and-bounce. If the tape gradient is zero/garbage vs FD, rigid recovery
uses the solver-agnostic FD-Jacobian LM (as VBD cloth did).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

import newton  # noqa: E402


@wp.kernel
def final_z(bq: wp.array(dtype=wp.transform), loss: wp.array(dtype=float)):
    loss[0] = wp.transform_get_translation(bq[0])[2]


def build():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1,
                    cfg=newton.ModelBuilder.ShapeConfig(density=500, restitution=0.5))
    return b.finalize(requires_grad=True)


def rollout_z(vx, n=60, grad=False):
    m = build()
    sol = newton.solvers.SolverXPBD(m, iterations=10, enable_restitution=True)
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    states = [m.state(requires_grad=True) for _ in range(n + 1)]
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    states[0].body_qd.assign(np.array([[vx, 0, 0, 0, 0, 0]], dtype=np.float32))
    tape = wp.Tape() if grad else None
    ctx = tape if grad else _Null()
    with ctx:
        for t in range(n):
            states[t].clear_forces()
            pipe.collide(states[t], contacts)
            sol.step(states[t], states[t + 1], None, contacts, 1.0 / 240)
        wp.launch(final_z, dim=1, inputs=[states[-1].body_q], outputs=[loss])
    if grad:
        tape.backward(loss)
        g = float(states[0].body_qd.grad.numpy()[0, 0])
        return float(loss.numpy()[0]), g
    return float(loss.numpy()[0])


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        L, g = rollout_z(0.5, grad=True)
        print(f"final z = {L:.5f}   tape grad d(z)/d(vx0) = {g:+.6e}")
        h = 0.05
        fd = (rollout_z(0.5 + h) - rollout_z(0.5 - h)) / (2 * h)
        print(f"FD central (h={h}) = {fd:+.6e}")
        if abs(g) < 1e-9:
            print("VERDICT: XPBD tape gradient is ZERO through contact -> use FD-Jacobian LM")
        else:
            rel = abs(fd - g) / max(abs(fd), abs(g), 1e-12)
            print(f"rel err = {rel:.3e} -> {'tape usable' if rel < 0.1 else 'tape UNRELIABLE, use LM'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
