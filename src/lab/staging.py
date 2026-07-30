"""Where a staged body sits, and the contact constants the rollouts use.

Extracted from the retired grid fitter, which mixed three unrelated jobs in one
file: producing the data (tracker seeds, point tracks), placing bodies, and
estimating parameters. Only the estimator was replaced; placement and the
tracking pass are still needed, so they live here and in scripts/prepare_lab.py.
"""
from pathlib import Path

import numpy as np

# Contact stiffness is deliberately stiff: at k=4e3 a parked object visibly sank
# into the ground in proportion to its density, which leaked a fake density signal
# into every measurement (M?? soft-contact bug). PITCH is the sphere-cover spacing.
SUBSTEPS, K_CONTACT, PITCH = 60, 2500.0, 0.020


def obj_geom(so, pos, load_asset, decimate, sphere_cover, ground_z):
    """Body centre matching the simulator's own sphere cover.

    The simulator centres a body on its vertex mean and represents it with
    finite-radius spheres, so placing a body by its mesh bounding box buried it
    and made frame 1 pop up by ~6 cm. Deriving the centre the same way the
    sphere cover does takes the residual drift to ~0.
    """
    cat = so["asset"].split("/")[0]
    asset = Path(so["asset"]).name
    scale = so["scale"]
    tm = decimate(load_asset(cat, asset), 400).copy()
    tm.apply_scale(scale)
    centers, r = sphere_cover(tm, PITCH * scale)
    vmean = np.asarray(tm.vertices).mean(0)
    c = np.array([pos[0] + vmean[0], pos[1] + vmean[1],
                  ground_z + r - float(centers[:, 2].min())])
    return c, f"{cat}/{asset}", scale
