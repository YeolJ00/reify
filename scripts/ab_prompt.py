"""Build the four prompt arms for the A/B, including the JSON form the model card asks for.

The Cosmos3-Nano card states that "text prompts should be upsampled into a specific JSON
structure", and ships assets/example_i2v_prompt.json (18 keys, ~8k chars) as the reference
plus assets/negative_prompt.json. This project had been sending a one-key {"scene": ...}
stub and no negative prompt at all.

Two parts of the schema map directly onto our failure modes, which is why this is worth
testing rather than assuming:
  * `actions` / `segments` carry explicit timestamps, and our commonest failure is an
    object that never moves or moves on its own schedule;
  * the shipped negative prompt names "floating or improperly grounded" and "distorted
    features", which is close to a description of our degrading brass pot.
"""
import json
from pathlib import Path

CACHE = Path("/home/nas5/jooyeolyun/hf_cache/hub/models--nvidia--Cosmos3-Nano/"
             "snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa/assets")

NOUN = {"ceramic_vase": ("white ceramic vase",
                         "Smooth glossy white glazed ceramic, narrow neck widening to a "
                         "rounded body, about 25 cm tall"),
        "brass_pot": ("copper cooking pot",
                      "Polished copper body with a domed lid and two side handles, warm "
                      "metallic reflections, about 19 cm across"),
        "rubber_duck": ("yellow rubber duck",
                        "Glossy yellow rubber with an orange beak, about 18 cm long")}


def negative_prompt():
    p = CACHE / "negative_prompt.json"
    return json.dumps(json.load(open(p))) if p.exists() else None


def plain(subject):
    noun = NOUN.get(subject, (subject.replace("_", " "), ""))[0]
    return (f"The {noun} falls onto the wooden tabletop, hits it and visibly bounces "
            f"back up, then comes to rest. Nothing else moves.")


def structured(subject, height_m=0.18):
    """The documented JSON form, with the drop laid out on a timeline."""
    noun, look = NOUN.get(subject, (subject.replace("_", " "), ""))
    return json.dumps({
        "subjects": [{
            "description": f"A {noun} released in mid-air above a wooden tabletop, "
                           f"falling under gravity.",
            "appearance_details": look,
            "relationship": "The only moving object in the scene; everything else rests "
                            "still on the table.",
            "location": f"Initially airborne about {height_m*100:.0f} cm above the "
                        f"tabletop, left of centre.",
            "relative_size": "Small relative to the table, occupying roughly a fifth of "
                             "the frame height."}],
        "background_setting": "A wooden dining table in front of a plain grey-blue wall, "
                              "with a rubber duck, an apple, a wooden bowl, a book and a "
                              "baseball resting motionless further back on the table.",
        "lighting": {"conditions": "Bright natural daylight from a window",
                     "direction": "Top-left, slightly in front",
                     "shadows": "Soft contact shadows directly beneath each object",
                     "illumination_effect": "Even, clear, high visibility"},
        "aesthetics": {"composition": "Static eye-level three-quarter view of the table",
                       "color_scheme": "Warm wood browns, neutral grey-blue wall",
                       "mood_atmosphere": "Calm, observational, documentary"},
        "cinematography": {"camera_motion": "None. The camera is locked on a tripod for "
                                            "the entire shot.",
                           "framing": "Medium shot of the tabletop",
                           "camera_angle": "Eye level",
                           "depth_of_field": "Deep", "focus": "The falling object"},
        "style_medium": "Live-action video",
        "artistic_style": "Realistic locked-off product footage",
        "context": "A physics demonstration: a single object is dropped onto a table and "
                   "allowed to bounce and settle.",
        # the timeline is the point of this arm
        "actions": [
            {"time": "0:00-0:00.2",
             "description": f"The {noun} is airborne and begins to accelerate downward."},
            {"time": "0:00.2-0:00.3",
             "description": f"The {noun} strikes the tabletop and rebounds a short "
                            f"distance upward."},
            {"time": "0:00.3-0:02",
             "description": f"The {noun} settles onto the tabletop and comes to rest, "
                            f"remaining upright and intact."}],
        "text_and_signage_elements": [],
        "segments": [
            {"segment_index": 0, "time_range": "0:00-0:00.3",
             "description": f"The {noun} falls and hits the table.",
             "key_changes": "Rapid downward motion, then contact and rebound",
             "camera": "Locked off, no motion"},
            {"segment_index": 1, "time_range": "0:00.3-0:02",
             "description": f"The {noun} settles and remains still.",
             "key_changes": "Motion decays to rest; the object keeps its exact shape",
             "camera": "Locked off, no motion"}],
        "transitions": [],
        "temporal_caption":
            f"A {noun} hangs in mid-air above a wooden table, then falls quickly and "
            f"strikes the surface within the first fraction of a second. It bounces "
            f"slightly, rocks, and settles into stillness, keeping its shape exactly "
            f"throughout. Nothing else in the scene moves and the camera never moves.",
        "audio_description": "A short hard knock as the object contacts the wooden "
                             "surface, followed by quiet.",
        "resolution": {"W": 544, "H": 448},
        "aspect_ratio": "17,14", "duration": "2s", "fps": 24})


ARMS = {
    "control":   dict(structured=False, negative=False),
    "neg":       dict(structured=False, negative=True),
    "json":      dict(structured=True,  negative=False),
    "json_neg":  dict(structured=True,  negative=True),
}
