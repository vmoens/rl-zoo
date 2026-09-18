# Initial physical feasibility measurements

Native MuJoCo 3.10.0, Torch 2.11.0, TensorDict 0.14.2, the pinned nine-skill
artifact and one PyTorch CPU thread on macOS 26.6.2 arm64. These are short
mechanics probes with fixed skills, not training runs or independent learning
seeds. Run the scripts from this directory after installing the matching zoo.
The JSON files record each condition and preserve unsuccessful trials.

## Box geometry

The original 14x20x10 cm box tipped under two ducks using the forward skill at
25, 50 and 100 g. Lowering its height to 3 or 5 cm made the feet collide with
it and did not fix stability. A 28x36x10 cm, 50 g box stayed upright (minimum
upright cosine 0.977) and moved 1.025 m under both forward skills in ten seconds.
A push-then-stand rule moved it 0.785 m in x, with minimum upright cosine 0.978,
but did not satisfy the settled-delivery condition. No numerical physics
errors occurred. Ducks still fell frequently; the recorded `falls` is the
number of fallen per-duck decision flags, not distinct fall events.

The initial pushing recipe uses the wider 50 g box and correspondingly wider
spawn separation. Friction is unchanged. This establishes movable, upright
box mechanics; successful cooperative delivery remains an evaluation target.
The pilot is still capped at two hours and must be compared with the standing,
forward-only and single-duck baselines.

## Actual barrier crossings

The relay probe starts a duck before the physical barrier with 1 cm spawn
perturbations, using the forward skill (index 1) or forward-hop skill (index 5).
A crossing only succeeds upright if the duck clears the far face within the
barrier's width and has never fallen; in-place respawns cannot count as success.
Both evaluated resets crossed flat ground and a 5 mm barrier upright with
both skills. At 10 mm, forward walked across upright but forward-hop fell.
At 15 mm, neither skill had an upright crossing. No numerical physics errors
occurred. This small deterministic evaluation does not establish robust
clearance under arbitrary approach phase or noise.

The first relay pilot therefore remains flat. The explicit 5 mm barrier
variant is feasible in these probes; obstacle-course training follows a
validated flat relay and its separately selected compute budget. Foot-site
height alone is not treated as obstacle clearance.
