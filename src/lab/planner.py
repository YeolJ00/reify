"""The lab: turn a REQUESTED PHYSICAL ATTRIBUTE into the experiment that reveals it.

A user does not ask for "tri_ke = 250". They say *"I want this duck to be squishy"*, or
*"how heavy is this thing?"*. Two different jobs come out of that:

  MEASURE  — "how squishy is it?"  -> we must run an experiment in which squishiness is
             actually visible, then recover it.
  AUTHOR   — "make it squishy"     -> we must generate motion of a squishy object, then
             recover the parameters that reproduce it, so the asset really is squishy in
             simulation (and physically consistent, because it came through the solver).

Both need the same core service: **pick an excitation that makes the target parameter
observable**. That is exactly what the probe->parameter matrix measured, so the numbers
in PROBES below are measured effect sizes from our own runs, not guesses:

    parameter        visible in            invisible in
    restitution      drop  19.0 px         push  0.1 px
    friction         push  67.3 px         drop  3.3 px
    mass ratio       collide 66.9 px       drop/push 0.4 px
    soft stiffness   6 cm drop (0.0% err)  2.5 cm drop (56% err)
    cloth stiffness  tilted drop           flat drop (no signal at all)

A generated video also has to be *usable*: Cosmos reversed directions, curved a rolling
ball 17% off a straight line, and faked 4 of 6 collisions. So every take is screened
before it is allowed to inform a parameter (see `screen_take`).
"""
from dataclasses import dataclass, field


@dataclass
class Probe:
    """One excitation: how to set the scene up and what to ask the video model for."""
    name: str
    excites: str                      # the parameter it makes observable
    why: str                          # measured justification
    prompt: str                       # template for the i2v model
    setup: str                        # how the initial frame must be arranged
    escalation: list = field(default_factory=list)   # do these if the signal is too weak


PROBES = {
    "drop": Probe(
        name="drop", excites="restitution (bounciness)",
        why="moves the image 19.0 px in a drop vs 0.1 px in a push — a push never impacts, "
            "so bounciness is literally not in the picture",
        prompt="The {obj} falls straight down onto the {surface}, bounces, and settles.",
        setup="{obj} held clear above the surface, nothing in its path",
        escalation=["raise the drop height", "drop onto a harder surface"]),
    "push": Probe(
        name="push", excites="friction",
        why="67.3 px in a push vs 3.3 px in a drop — friction only shows where there is sliding",
        prompt="The {obj} slides across the {surface}, slowing down, and comes to rest.",
        setup="{obj} resting on the surface with a long clear runway",
        escalation=["push harder", "start it sliding rather than rolling"]),
    "collide": Probe(
        name="collide", excites="mass ratio (relative weight)",
        why="66.9 px in a collision vs 0.4 px otherwise — weight is invisible until "
            "momentum is transferred to something else",
        prompt="The {obj} rolls into the {other} and knocks it away.",
        setup="{obj} aimed squarely at a second object a short distance away",
        escalation=["hit it faster", "aim more squarely (less glancing)"]),
    "hard_drop": Probe(
        name="hard_drop", excites="soft-body stiffness (squishiness)",
        why="a 6 cm drop recovers stiffness to 0.0% while a 2.5 cm drop is 56% off — the "
            "impact has to be hard enough to actually deform the object",
        prompt="The {obj} falls onto the {surface} and squashes on impact, then springs back.",
        setup="{obj} held well above the surface so the impact is hard",
        escalation=["increase the drop height further", "drop onto a rigid surface"]),
    "tilted_drop": Probe(
        name="tilted_drop", excites="cloth stiffness (drape)",
        why="a flat sheet landing flat gives NO stiffness signal; dropped at a tilt the "
            "floppy and stiff versions separate by 8.8 cm",
        prompt="The {obj} falls at an angle onto the {surface} and collapses into a heap.",
        setup="{obj} tilted, one corner leading",
        escalation=["increase the tilt", "drop from higher"]),
}

# what a user might say -> which parameter they mean
ATTRIBUTE_ALIASES = {
    "bouncy": "restitution", "bounciness": "restitution", "springy": "restitution",
    "restitution": "restitution", "dead": "restitution",
    "slippery": "friction", "grippy": "friction", "friction": "friction",
    "heavy": "mass", "light": "mass", "weight": "mass", "mass": "mass", "dense": "mass",
    "squishy": "soft_stiffness", "soft": "soft_stiffness", "squashy": "soft_stiffness",
    "stiff": "soft_stiffness", "firm": "soft_stiffness",
    "floppy": "cloth_stiffness", "drapey": "cloth_stiffness", "flowy": "cloth_stiffness",
}
PARAM_TO_PROBE = {
    "restitution": "drop", "friction": "push", "mass": "collide",
    "soft_stiffness": "hard_drop", "cloth_stiffness": "tilted_drop",
}

# screening thresholds, all taken from measurements on generated video
MIN_TRACK_COVERAGE = 0.60      # fraction of frames the object must be found in
MAX_PATH_CURVATURE = 0.10      # a free-rolling ball measured 17% -> non-physical; good takes ~7%
MAX_CONTACT_GAP_PX = 25.0      # target may only start moving once it is really touched
MAX_FIT_RESIDUAL_PX = 20.0     # above this the sim cannot explain the motion at all
MIN_EFFECT_PX = 2.5            # tracking-noise floor: below this a parameter is invisible


def resolve(attribute: str):
    """'I want the duck squishy' -> ('soft_stiffness', Probe(hard_drop))."""
    key = attribute.strip().lower()
    param = ATTRIBUTE_ALIASES.get(key)
    if param is None:
        for word, p in ATTRIBUTE_ALIASES.items():
            if word in key:
                param = p
                break
    if param is None:
        raise KeyError(f"don't know which experiment reveals '{attribute}'; "
                       f"known: {sorted(set(ATTRIBUTE_ALIASES))}")
    return param, PROBES[PARAM_TO_PROBE[param]]


def plan(attribute: str, obj="object", other="second object", surface="table"):
    """Full plan for measuring/authoring one requested attribute."""
    param, probe = resolve(attribute)
    return {
        "requested": attribute, "parameter": param, "probe": probe.name,
        "why_this_probe": probe.why,
        "initial_frame": probe.setup.format(obj=obj, surface=surface),
        "prompt": probe.prompt.format(obj=obj, other=other, surface=surface),
        "escalation_if_weak": probe.escalation,
    }


def screen_take(coverage, curvature=None, contact_gap=None, fit_residual=None):
    """Is this generated take physically usable? Returns (ok, reasons_rejected)."""
    bad = []
    if coverage < MIN_TRACK_COVERAGE:
        bad.append(f"object only tracked in {coverage*100:.0f}% of frames")
    if curvature is not None and curvature > MAX_PATH_CURVATURE:
        bad.append(f"path curves {curvature*100:.0f}% off straight — nothing applies that force")
    if contact_gap is not None and contact_gap > MAX_CONTACT_GAP_PX:
        bad.append(f"target starts moving {contact_gap:.0f} px before contact — not causal")
    if fit_residual is not None and fit_residual > MAX_FIT_RESIDUAL_PX:
        bad.append(f"physics cannot explain the motion ({fit_residual:.0f} px residual)")
    return (len(bad) == 0), bad
