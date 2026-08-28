# Supervised public-material experiment v1

28 August 2026. This is a completed, bounded development experiment, not UK or independent-asset validation.

Subsequent audit and one frozen-feature classifier are now complete: see [material head v1](MATERIAL_HEAD_V1.md). The orientation exclusion breakdown is resolved there; this original run and the historical observations below remain unchanged.

## Execution and verification

- Roihu gputest job **919934**, NVIDIA GH200 120GB, **COMPLETED / 0:0 / 3m38s**.
- One fixed 20-epoch YOLOE-26m detection adaptation from the pinned segmentation checkpoint; fixed final `last.pt`, not selected against target performance. Training took 201.11 seconds; training plus final inference took 202.65 seconds inside the Python runner.
- 600 training images, 25 development images; 271 training images contain glass and 525 contain porcelain (these sets overlap). All labels come from original publisher polygons, never generated candidates.
- Final checkpoint SHA256: `ff77b7f66faf3011477028eaf6dac6adb90375d024cad857e5ae27da7f07dcd1`.
- Dataset manifest SHA256: `9b8c5495ac6967682130e67df752b3050b77a2bbae40a206b56ed6b1a7bc6290`.
- Protocol SHA256: `f5e5eea672f63923f3fff0afd07491f97b5f138e06b1ccad76cbdd9f29e359f2`.
- All 625 derived label files independently reconstructed from the retained original JSON; 33 prediction files and display images hash checked; duplicate graph and disjoint capture groups recomputed; 20 finite-loss epochs and final checkpoint checked.

The new English report is at local port **8774**, root `runs/substation_material/v1_20260828`. All 25 development images and all eight previously inspected UVInsDet transfer examples are shown. Ports 8771–8773 retain their earlier results. No new model ran on Apple.

## Actual result

Class-aware matching at detector score ≥0.25, IoU ≥0.50:

| Development class | Source objects | TP | FP | FN | Precision | Recall | AP50 from saved outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Glass | 116 | 50 | 14 | 66 | 78.1% | 43.1% | 51.7% |
| Porcelain | 407 | 133 | 74 | 274 | 64.3% | 32.7% | 39.2% |

AP uses all saved post-NMS boxes down to 0.05, with independent 101-point interpolation; it is not the library's default COCO evaluation. The measured material names follow publisher categories, not an independent physical material assay. High detector scores are not calibrated material probabilities.

The model now produces source-matching porcelain localisations on this dataset. However, it misses most development objects and has substantial false positives. The preselected bright portrait contains 8 TP, 2 FP and 4 missed source objects; its attractive overlay must not hide these errors.

**Cross-source transfer failed:** none of the eight UVInsDet demo images has a detection above 0.25. Across the seven annotated inputs, all 8 glass and 12 porcelain references are missed; the eighth image is unannotated and excluded from accuracy calculations. The previous Grounding DINO arm still locates the six main glass strings there. This new supervised detector is not promoted as a general replacement.

## Data selection limitations discovered

The fixed groups were `undated_FLIR` and `date_20210721` for development, with other capture groups for training. Date/camera grouping is a conservative proxy, not asset identity; every image is from one substation. Exact pixel hashing plus an aspect-aware dHash distance ≤4 removes duplicate representatives. No selected duplicate edges or capture-group overlap remained. Ninety-one same-split duplicates were dropped; no cross-split duplicate component was retained.

The first preparer excluded **791** files when dimensions did not match or EXIF orientation was not exactly 1/absent. This rule was overly conservative: at least the audited `FLIR0335_rgb.jpg` and `FLIR6829_rgb_AdgweI4.jpg` have orientation **0 (undefined)**, matching raw image/annotation dimensions, and exactly reproducible masks. They were excluded despite their demonstrated coordinate compatibility. This explains why the planned maximum of 96 development images became 25. The experiment is preserved as run; no retrospective split change or automatic rerun was made.

The exact decomposition of all 791 exclusions into undefined orientation, actual rotation and dimension mismatch has not yet been audited. Do not describe all excluded files as corrupt, rotated, or incorrectly labelled. Only one of the three earlier format-audit images entered this development set. Before another experiment, inspect these cases and explicitly define raw-pixel versus EXIF-transposed coordinates, verifying against publisher masks. Preserve the v1 record.

## Viewer semantics

The score field filters only the display; aggregate metrics remain fixed at the stated operating point. The class filters include a separate unknown state for proposals below 16 native pixels on either side or 512 square pixels. That is a pixel-based abstention, not a trained unknown class or a guarantee against out-of-domain mistakes. Raw proposals remain available.

Publisher reference boxes are dashed, separately toggled, and never drawn as model predictions. The inspector crops original source pixels; display scaling does not create additional material evidence. Invalid score input preserves the last valid threshold. Loading images cannot be cropped as though they were the next image.

## Next technical direction, not another launch instruction

The results support separating localisation from material inference: retain the stronger generic Grounding DINO proposals, then test a small supervised material classifier on frozen visual features, using original material crops and equipment/background negatives. This would expose whether the current transfer failure is mainly localisation, object scale, or material representation. Report classification on source crops separately from the end-to-end automatic-crop result; missed detections must remain in the denominator.

First fix and audit the orientation handling. Do not simply extend the 20-epoch run or tune thresholds against these eight transfer images. A future protocol should include rejection evidence and explicit label-unit compatibility. Steelwork and pole-top still need their own definitions and evidence; this material experiment adds neither class.
