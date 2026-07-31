# Project summary — one line per concept

**Goal:** recover the physical parameters of a roughly-placed simulation asset so a
differentiable Newton/Warp simulation reproduces its observed motion (physics from video).

- **Differentiable rollout (M0–M1):** to get a physics gradient, we built a cloth-flag sim with a custom wind force and checked its gradient against finite differences — Warp's semi-implicit solver was differentiable, VBD/XPBD were not.
- **Parameter recovery (M2–M3):** to prove we can invert the sim, we recovered wind/gravity/stiffness from a synthetic target by gradient descent, and found mass/density is a *gauge freedom* — unobservable when it doesn't change the motion.
- **From video (M4):** to move toward real observation, we rendered the sim, tracked it with Lucas–Kanade, and recovered parameters through a differentiable camera — hitting a ~4 px tracking-noise accuracy floor.
- **i2v model choice (bake-off):** to pick a video generator, we scored Wan / HunyuanVideo / Cosmos3 by which one's generated motion best fit our physics, and chose Wan 2.2.
- **Real scene render (city):** to leave the white background, we rendered the flag in a real city street with Blender + CC0 props (bench, lamp, hydrant, HDRI).
- **Real-asset rigid recovery (M5–M6):** to test real geometry, we dropped scanned objects in Newton and recovered their physics — confirming density is unobservable in free fall and only weakly coupled by gentle collisions.
- **The wall (M7):** to make density observable we forced a violent collision, which worked in theory but was unrecoverable — the loss landscape went chaotic and both gradient and global search failed.
- **Root cause (M7b):** we traced every contact failure to Newton's XPBD contact being non-differentiable, non-scale-invariant, and leaking 57 % of momentum.
- **The fix (M8–M11):** we wrote our own contact as a Warp kernel — a penalty/friction contact that is differentiable and conserves momentum by construction — and recovered density, friction, and restitution from a collision of real meshes with full 6-DOF rotation.
- **Full pipeline (M12):** to close the loop, we ran it all end-to-end — real objects → simulate → render video → track → recover physics through the differentiable sim+camera — getting density within ~5 % from the video alone.

**Honest gaps:** the collision demos are bare physics test rigs, not staged scenes; the "in-the-wild" version (replace the rendered video with a Wan generation, on objects placed in a real environment) is plumbed but not yet run.
