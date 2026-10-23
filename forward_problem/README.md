# Forward Problem (Legacy)

**This is the original forward model implementation.** It uses a
non-alternating refractive index scheme that does NOT match the
physical air-glass-air model described in the paper.

The correct forward model used in the paper is in
`reverse_problem_v2/core.py`, which implements:
- Two interfaces per prism (entry + exit face)
- Alternating air-glass-air refractive indices
- Lockstep rotation of both faces

This code is retained for historical reference and for the
`generate_examples.py` gallery output.
