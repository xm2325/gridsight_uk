# Component masks and explicit pole-end geometry

28 August 2026. This adds a verified region-localisation capability and a bounded comparison. It does **not** establish UK production accuracy, steel material identification, or supervised pole-top detection.

## Execution and recovery

The experiment used cached public EPRI polygons: 320 training images and 80 development images from disjoint publisher circuits. The original 100-image evaluation split was not used. Twenty fixed epochs trained a YOLOE-26m instance segmenter from the pinned original segmentation weights, with overlapping instances retained. No pseudo labels, source-label corrections, material labels, new images or expert annotations were introduced.

Training job **921053** ran on Roihu gputest for **4m51s** and ended FAILED after training. Its CSV contains exactly epochs 1–20; the checkpoint is complete. Ultralytics invoked the final epoch callback again during final validation, yielding callback events 1–20,20. The initial script's strict event-count assertion then stopped execution before inference. The failed record and original source snapshot remain unchanged.

Inference-only continuation **921987** completed **0:0 in 56 seconds**, processing all 80 development and 27 previously used UK qualitative images. It loaded the existing final `last.pt`, performed **zero training steps**, and retained separate records under `inference/`. Model inference took 46.88 seconds inside Python; the training CSV's cumulative time is 271.806 seconds. No model ran on Apple.

The source runner now records duplicate final-validation callbacks separately. Source-frame coordinates also now multiply in Python float64: the executed inference's float32 serialization produced 86 boxes just outside the native canvas, by at most **0.000244 pixels**. The verifier preserves those serialized values and independently remaps the original working-frame boxes in float64 for metrics/display, requiring agreement within 0.001 pixels. Neither saved prediction files nor model outputs were rewritten.

## Measured result

At score ≥0.25 / class-aware IoU ≥0.50 on all 80 development images:

| Output | Class | TP | FP | FN | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Saved 40-epoch detector boxes | Pole | 70 | 3 | 11 | 95.9% | 86.4% |
| New segmenter boxes, fixed NMS | Pole | 68 | 3 | 13 | 95.8% | 84.0% |
| Saved detector boxes | Crossarm | 9 | 2 | 36 | 81.8% | 20.0% |
| New segmenter boxes, fixed NMS | Crossarm | 9 | 5 | 36 | 64.3% | 20.0% |
| Saved detector boxes | Insulator | 105 | 18 | 43 | 85.4% | 70.9% |
| New segmenter boxes, fixed NMS | Insulator | 106 | 14 | 42 | 88.3% | 71.6% |

This is not a controlled architecture ablation: the old detector used 40 epochs and development checkpoint selection; the new segmenter uses fixed final epoch 20. It is not an overall replacement for the existing detector. Crossarm misses remain substantial.

The explicit class-aware NMS diagnostic uses the already specified IoU **0.5**, with no threshold search. It was added while inspecting development outputs and is labelled as derived postprocessing, not untouched-test validation. Before this explicit pass, the new model returns duplicate candidates: pole FP17 becomes FP3; insulator TP108/FP21 becomes TP106/FP14. Raw returned boxes and masks remain selectable, including discarded candidates. All 486 saved instance masks are retained.

The useful improvement is **region extent**. On the new model's correct-class box matches, compare its mask with its own filled rectangle against the same supplied polygon:

| Class | Matched objects | Mean mask IoU | Mean rectangle IoU |
| --- | ---: | ---: | ---: |
| Pole | 68 | 0.900 | 0.699 |
| Crossarm | 9 | 0.698 | 0.568 |
| Insulator | 108 | 0.785 | 0.656 |

These are conditional extent measurements, not full-input recall. With fixed NMS, full-input mask TP/FP/FN are pole **68/3/13**, crossarm **8/6/37**, insulator **106/14/42**. Source polygons are coarse outlines rather than independently verified pixel-perfect physical masks. The class-review flag for `epri_c4_176` remains visible and uncorrected.

**UK transfer remains weak.** Across 27 existing UK images, the as-returned new model has only two pole and two insulator candidates at 0.25, with no crossarm candidates; the saved supervised detector has six pole and one insulator candidates. These are proposal counts, not accuracy or a reason to rank models without labels. No UK component/material truth exists for this comparison.

## Steelwork and pole-top boundaries

A crossarm mask identifies a structural role; it does not establish steel. The new report separately offers the **35 earlier Grounding DINO steelwork hypotheses** at their original 0.30 threshold, explicitly marked unreviewed. No steelwork training, new DINO inference, steel material probability or corrosion estimate is claimed.

An unscored pole-end candidate is derived from an elongated predicted pole mask and proximity to predicted crossarms/insulators. Ambiguous association, missing component context, or a candidate at an image edge causes abstention. The as-returned outputs produce 30 candidate ends and 55 abstentions on development masks; both UK pole masks abstain. Duplicates are not distinct physical poles; the viewer filters endpoint overlays according to the selected raw/NMS mode.

This geometry is a navigation aid, not a labelled shaft-tip detector. It may choose the wrong end or follow a wrong mask. There are no independent tip annotations, no physical-tip accuracy, no asset IDs and no confidence score. The API retains `derived=true`, `score=null`, `supervised_pole_top=false`.

Keen's published steelwork system separates structural pixels before assessing corrosion and localising tower parts. That motivates better region localisation; it does not make EPRI crossarm masks equivalent to steel masks. [Keen's case study](https://keen-ai.com/case-study/corrosion-detection-on-overhead-line-towers/)

## Artifacts and verification

- English viewer: local **8776**, `runs/component_masks/v1_20260828/inference/report/index.html`.
- Protocol SHA256: `5be95b74726a59b71a79cd958bd7a4dbed3dfe24ed8d6c83d85cf9d9746d6432`.
- Segmentation manifest SHA256: `224245afb6b00e9e0f816e250e1e1b76f74bb122f5b4856ba02685235a58acd2`.
- Final checkpoint SHA256: `c9c683d9c2ad9b3413d7fa5ce219167ff2913be6be7b8a94d1577548aba40ba1`.
- Failed training-record SHA256: `0beaaa88557b13a8718932aba8acdb5ae04cf07b00dad4483f4a6418c78162c4`.
- Original training CSV SHA256: `d112c584d348c1406abc1bd59dbc84307e79855dbec10b3a8633a25d3c6b1f2e`.

Verification reconstructs all 400 segmentation label files from retained source polygons, checks split boundaries and finite epoch losses, verifies the final checkpoint and both phase records, reconstructs all 107 working images exactly from original pixels on Roihu, and compares all 486 rendered mask PNGs to their packed binary outputs. The 1,280-pixel working raster is explicitly distinguished from original native resolution. Local computation is numerical verification/rendering only.

The report exposes all inputs, all returned candidates, fixed-NMS output, the saved box baseline, separate source polygons, class filters, source credits and failures. Browser checks cover 107 images × three modes, masks and class controls, score validation, source-review flags, unreviewed steelwork labels, and unscored geometry. Mobile layout was not exercised. Ports 8771–8775 and previous frozen results remain unchanged.

Source: [EPRI Drone-based Distribution Inspection Imagery](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery), CC BY-SA 4.0. Implementation reference: [official YOLOE segmentation documentation](https://docs.ultralytics.com/models/yoloe).

Do not repeat either job or extend training automatically. Remaining priorities are compatible steel-member evidence, a precise labelled pole-top target, and component localisation that transfers to distribution-pole images. More colours or higher displayed scores cannot supply those missing capabilities.
