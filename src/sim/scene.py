"""Build the M0 scene: a cloth flag pinned on one edge, as a Newton Model.

Verified against newton 1.4.0 / warp-lang 1.15.0 installed APIs.
"""

import warp as wp

import newton


def build_flag_model(cfg: dict, requires_grad: bool = False) -> newton.Model:
    """Build a cloth-flag Model from the `cloth` section of a config dict.

    The grid is authored in its local XY plane and rotated +90 deg about X so the
    cloth hangs in the world X-Z plane (Z-up). `fix_left` pins the hoist edge.
    """
    c = cfg["cloth"]

    builder = newton.ModelBuilder()
    builder.default_particle_radius = 0.01

    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.5),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(c["dim_x"]),
        dim_y=int(c["dim_y"]),
        cell_x=float(c["cell"]),
        cell_y=float(c["cell"]),
        mass=float(c["mass"]),
        fix_left=bool(c["fix_left"]),
        tri_ke=float(c["tri_ke"]),
        tri_ka=float(c["tri_ka"]),
        tri_kd=float(c["tri_kd"]),
        edge_ke=float(c["edge_ke"]),
        edge_kd=float(c["edge_kd"]),
    )

    builder.color()  # particle graph coloring, required by SolverVBD
    model = builder.finalize(requires_grad=requires_grad)
    return model
