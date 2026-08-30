# UK material prospective transfer v1

30 August 2026. This is a bounded source-evidenced diagnostic, not UK population accuracy or a deployment claim.

## Frozen data boundary

The source pool contains six adaptation images from three asset groups and five prospective test images from five disjoint asset groups. The test set has 18 analyst-selected insulator regions: 13 glass and 5 porcelain/ceramic. Publisher text on each Geograph photo page supplies the material evidence. The rectangles are source-assisted analyst oracle regions, not expert inspection annotations. No UK polymer/composite target was found, so this run makes no UK polymer claim.

The manifest preserves author, licence, photo page, original image URL, evidence excerpt, image/page hashes and asset group. It was frozen before inference at SHA-256 `54fd6a24adc1e49a4c1af7cf21338d1bfa22b9382f7e5ee4a315fae22eb3a45c`. Adaptation and prospective test asset groups do not overlap.

## Frozen protocol and runtime

Roihu Slurm job `944218` ran on one NVIDIA GH200 in `gputest` and completed with exit code `0:0` in 19 seconds. The SigLIP2 encoder was frozen (`0` encoder gradient steps). A single final-layer adaptation ran for 120 steps using the academic base and nine UK adaptation regions. Rejection thresholds came from academic development data. Prospective test images were not used for training, threshold selection or retry. There was no automatic retry.

Pinned protocol SHA-256: `f40f2c7412ddaee2d109453f6632b800327f1572cb0fa538413cb38e2e8beb68`. Result SHA-256: `f959a8a7ea8e1b6f476567b8a01833e88249ab50e8358a3becc605d56c10b6f5`.

## Results

| Evaluation layer | Frozen v2 head | Adapted final head |
|---|---:|---:|
| Accepted oracle regions | 8/18 (44.4% coverage) | 14/18 (77.8% coverage) |
| Correct among accepted oracle regions | 6/8 (75.0%) | 12/14 (85.7%) |
| Accepted assets | 5/5 | 4/5 |
| Correct among accepted assets | 3/5 (60.0%) | 3/4 (75.0%) |
| End-to-end localised regions | 1/18 | 1/18 |

The existing EPRI component detector accepted eight full-image predictions at score 0.25. Only one matched a source-assisted region at IoU 0.30; seven were unmatched. End-to-end region coverage is therefore 5.6%. The single matched crop was accepted and correctly classified by both material heads, but one observation cannot support an accuracy estimate.

This separates the current bottleneck: the adapted material head is promising on source-assisted crops, while UK full-image insulator localisation fails on this prospective set. The next bounded experiment should adapt the material-agnostic localiser with full-image UK boxes, small-object tiling and pole/crossarm context, then keep the same material head and untouched asset-group evaluation. A separate source-evidenced UK polymer set is required before three-material acceptance or calibration.

## Presentation

The all-English gallery at `runs/uk_capabilities/v3_20260827/report/material_prospective/index.html` shows every test asset in four panels: source-assisted regions, frozen v2 decisions, adapted decisions and actual detector output. Unknowns, wrong accepted materials, unmatched detections and localisation misses remain visible. Model margins and detector scores are not presented as probabilities.
