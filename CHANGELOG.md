# Changelog

Record every change to prompts, question sets or grading rules made after the `prompts-frozen` tag,
with the reason and the rerun it triggered.

## Unreleased

- 2026-11-03: scaffold created.
- 2026-11-03: M1. Physics core (bounds, Mie, quasistatic/Padé) and reproduction of PRB 94,
  140301(R) (arXiv:1606.05523v4): Eq. (7), Fig. 1 and Fig. 2. The paper does not say how its
  "exact Mie" references were extracted; we use Re(eps) at the complex pole of b_1 for Fig. 1 (the
  |b_1| = 1 real-eps resonance is kept as a secondary column) and critical coupling (c = 1/2) for
  Fig. 2. Physics core implemented by Claude at Dimi's request; CLAUDE.md updated accordingly.
- 2026-11-16: M3 gradient agent: constant-rate Adam (2000 steps, lr 0.01) bounced around the naive
  pole and stopped the corrected run at 92% of 6/x^2 (first run on Dimi's PC). Switched to a
  cosine-decaying rate, 5000 steps, lr 0.02 (configs/hack.yaml); tests now check the end state
  far above the limit (naive) and within 2% below it (corrected).
