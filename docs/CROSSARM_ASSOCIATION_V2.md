# Crossarm association v2

This is a deterministic development guardrail over the pinned Grounding DINO v1 outputs. It performs no model inference or training.

The v1 UK-development audit found three displayed candidates: one compact proposal overlapping a plausible pole-top support region, one box extending down a pole shaft, and one box spanning a fallen pole. The v2 rule therefore requires an upright pole input (height/width at least `1.5`) and limits candidate height to `0.45` of the associated pole height. The full frozen rule is in `configs/crossarm_association_v2.json`.

The rule was fixed after looking at the UK development failures. Its UK output is therefore a development comparison and cannot be called evaluation. UK v3 was not accessed.

On the publisher-labelled 80-image EPRI circuit-4 cohort, the fixed rule produces TP/FP/FN `18/0/27`, precision `1.000`, recall `0.400` and F1 `0.571` at IoU `0.5` and raw threshold `0.30`. The input arm produced `22/3/23`, precision `0.880`, recall `0.489` and F1 `0.629`. The guardrail trades recall for morphology and source-domain precision.

On the 27 UK development images, displayed proposals fall from three on two images to one on one image. UK precision, recall and accuracy remain unknown because no reference crossarm boxes exist. The guardrail is not promoted to UK v3 or the main overlay.

Result SHA-256: `1f3094c5cd85d79bbf9804176d0714e4eef8ff9bfe9066e3684a4a343d98949d`.
