# Steel-structure and pole-shaft endpoint evidence lab

30 August 2026. This addendum defines and tests two previously ambiguous outputs. It does not establish UK distribution-pole steelwork accuracy or a physical pole-top detector.

## Pole-shaft endpoint geometry

`configs/pole_top_keypoint_v1.json` freezes a development-only target on 80 EPRI images. An image is eligible only when publisher polygons contain exactly one pole and at least one crossarm and the pole geometry yields an elongated, unambiguous, non-truncated endpoint. The target is the visible pole-shaft axis endpoint nearest the publisher crossarm polygons. It is deterministically derived from human publisher masks; it is neither a model pseudo-label nor an independently annotated physical tip.

Roihu job **940348** completed in 5 seconds after job 940344 failed before computation because one fixed metrics dependency was absent. No model was rerun. The audit compares preserved output from jobs 910514 and 921987:

| Development geometry | Eligible | Accepted | Coverage | Median normalised error | PCK@0.25 over eligible |
|---|---:|---:|---:|---:|---:|
| Segmentation mask | 18 | 10 | 55.6% | 0.048 | 55.6% |
| Detection box | 18 | 8 | 44.4% | 0.057 | 44.4% |

Every accepted output was within 0.25 of the square-root pole-mask-area normaliser, but abstention remains substantial. The result supports a better navigation keypoint, not a Keen-style scored `pole-top` component. The current UK upper-pole window therefore remains an unscored inspection region.

## Lattice transmission-tower steel structure

The TTPLA paper states that lattice transmission towers are composed of steel angle sections and labels the tower assembly `tower-lattice`. This provides material evidence for a transmission-tower structural assembly mask. It does not label individual steel members, corrosion, or the material of distribution-pole crossarms.

The official Google Drive archive was temporarily rate-limited by its publisher. The failed attempt left no ZIP. A deterministic 60-image subset was acquired through the Hugging Face `grantmwilkinson/epri-transmission-ttpla` mirror while retaining row indices, signed download URLs, source categories and polygons, file and label hashes, the official repository commit, paper and licence links. The subset contains 24/8/8 positive images and 12/4/4 wooden or tubular/concrete/hybrid hard negatives across train/validation/test. Filename-prefix source groups do not cross splits. This is a small mirror demo, not the full TTPLA benchmark. One raw coordinate at y=-1.1865 px was preserved in the manifest and explicitly clamped to the image edge under the frozen sub-2-pixel boundary policy.

Roihu gputest job **940391** completed once in 59 seconds. It trained YOLOE-26m-seg for 20 fixed epochs on 36 images and used the final epoch without test or UK selection. The 12-image test has ten lattice instances and four hard-negative images:

| Fixed score 0.25 / mask IoU 0.50 | TP | FP | FN | Recall | Negative specificity | Mean IoU of matches |
|---|---:|---:|---:|---:|---:|---:|
| Open vocabulary | 2 | 0 | 8 | 20.0% | 100% | 0.630 |
| Supervised | 1 | 0 | 9 | 10.0% | 100% | 0.803 |

The supervised model localised its one accepted example more precisely but reduced recall. This small intervention does not justify deployment or another threshold search. It establishes a source-compatible steel-structure visualisation and exposes the gap to distribution-pole steelwork.

## English workbench

The 8772 review service now links to `report/upgrade/index.html`. The evidence gallery includes all 12 fixed steel-structure test images and all 18 eligible pole geometry images with Previous/Next controls. It shows publisher extents, accepted outputs, misses and abstentions separately. Browser QA verified English `en-GB`, 12 and 18 options, loaded 2134×600 and 1280×960 assets, both Next controls and zero console warnings/errors.

The remaining route to the supplied Keen-style image is evidence, not a different colour scheme:

1. asset-grouped UK pole, crossarm and insulator labels plus a new untouched acceptance set;
2. independent UK whole-assembly glass, porcelain and polymer material evidence for encoder adaptation and abstention calibration;
3. distribution-pole steel member or connected-assembly labels with explicit material basis;
4. a business choice between physical shaft-tip keypoints and upper-assembly extents, followed by compatible UK labels;
5. target-domain calibration before displaying percentages as probabilities.
