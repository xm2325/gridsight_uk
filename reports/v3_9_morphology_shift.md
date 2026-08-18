# v3.9 Post-final morphology-shift audit

**Analysis only. These findings must not be used to retune the v3.8 final-holdout result.**

## Training geometry

- 5 training towers / 30 insulator references.
- Training Mahalanobis d² q95: **6.567**; maximum: **7.008**.
- Median width/tower-width: **0.127**.
- Median height/tower-height: **0.130**.
- Median relative shape ratio `(h/tower_h)/(w/tower_w)`: **0.982**.

## Frozen final holdout

### POS_3437435 — England

- 6/6 insulators exceed training q95; **6/6 exceed even the maximum training d²**.
- Median d²: **39.629**.
- Median width/tower-width: **0.251**.
- Median height/tower-height: **0.051**.
- Relative shape ratio: **0.216**.

This is a substantial morphology shift toward wide, horizontally oriented/strain-like insulators. It explains why the frozen geometry prior removed correct YOLOE candidates.

### POS_7561805 — Scotland

- 1/6 exceed training q95; 0/6 exceed training maximum.
- Median d²: **2.482**.
- Median width/tower-width: **0.150**.
- Median height/tower-height: **0.113**.
- Relative shape ratio: **0.747**.

This tower is much closer to the training geometry distribution.

## Engineering conclusion

The independent holdout falsified the assumption that a narrow geometry prior learned from five similar towers would generalise across UK tower designs. The open-vocabulary YOLOE text model preserved recall much better; the post-processing prior created the larger generalisation failure. The correct next phase is a new morphology-diverse acquisition/training cycle with a **new preregistered holdout**, not tuning against these two final images.
