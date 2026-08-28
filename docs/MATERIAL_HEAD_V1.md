# Frozen visual features and supervised material head

28 August 2026. A completed diagnostic, **not an end-to-end improvement** or independent UK validation.

## What ran

Roihu `gputest` job **920543** completed with exit 0:0 in **46 seconds**. The Python model pipeline took **11.50 seconds**, excluding source-pixel verification and process startup. No model ran on Apple.

The encoder is the existing pinned SigLIP2 base NaFlex model. Its parameters were frozen; one three-class linear head was fitted for **400 full-batch AdamW steps**, learning rate 0.01, weight decay 0.001, seed 17, inverse class frequency weighting. Training uses tight crops only. Context crops are consistency checks. No search, selected checkpoint, extra epochs or detector rerun occurred.

| Split | Distinct source photos | Glass crops | Porcelain crops | Other crops |
| --- | ---: | ---: | ---: | ---: |
| Train | 458 | 500 | 500 | 300 |
| Development | 88 | 102 | 150 | 40 |

The 1,592 crops yield 3,184 tight/context views. These are not 1,592 independent assets. Source photos come from one Brazilian substation, with development capture groups separated from training. Exact pixel hashes and an aspect-aware dHash graph removed 183 duplicate representatives; no cross-split duplicate component was retained. Camera/date groups are not asset identities.

Materials follow original [Substation15 publisher polygons](https://zenodo.org/records/7884270), CC BY 4.0. Other crops are 64-pixel squares fully inside original Background, Power transformer, Muffle or Breaker polygons and disjoint from all supplied material boxes. They are source-derived non-target regions, not verified material assays or polymer examples. Their restricted size and source context may not represent real detector false positives.

The orientation audit checked all 1,660 original images: 627 have undefined EXIF orientation 0 and compatible raw-image/annotation/mask dimensions. They were restored to eligibility, alongside 824 orientation-1 and 45 orientation-absent files. The 164 actual rotation/dimension conflicts remain excluded. No coordinates were silently transposed, no original images changed, and no completed detector experiment rerun.

## Actual findings

### Given a correct source crop

| Diagnostic | Raw tight-crop argmax agreement | Accepted after rejection | Correct among accepted |
| --- | ---: | ---: | ---: |
| 292 same-substation development crops | 266/292 (91.1%) | 195/292 | 190/195 |
| 20 source-box crops from seven previously inspected UVInsDet photos | 19/20 (95.0%) | 14/20 | 14/14 |

The second row is an **oracle location diagnostic**: boxes come from publisher labels, not the detector. The 14 accepted crops comprise six glass and eight porcelain. The other six are unknown. All twelve porcelain examples come from **one** external photo and are individual source-labelled sheds; 14/14 is not evidence of reliable UK performance.

Rejection requires at least 16 native pixels per side and 512 pixels of area, matching tight/context argmax, and a raw logit margin of at least 0.5 in both views. Other/background predictions are rejected. The margin is a fixed heuristic, not a probability or calibrated confidence. All outputs remain provisional.

### Using automatic detector boxes

The head reuses the previous Grounding DINO proposals and their saved SigLIP2 embeddings on all eight [UVInsDet demo photos](https://zenodo.org/records/18197601), CC BY 4.0. Target labels and source-box crops never enter training or this automatic branch. The eighth photo has no annotations and is excluded from metrics, not declared negative.

At detector score ≥0.25 and class-aware IoU ≥0.50, agreement with the seven annotated source images is:

| Method | Class | TP | Unmatched predictions (FP) | Missed references (FN) | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original image prototypes | Glass | 6 | 6 | 2 | 50.0% | 75.0% |
| Original image prototypes | Porcelain | 0 | 26 | 12 | 0.0% | 0.0% |
| Supervised head | Glass | 6 | 30 | 2 | 16.7% | 75.0% |
| Supervised head | Porcelain | 0 | 3 | 12 | 0.0% | 0.0% |

The classifier changes many porcelain hypotheses into glass, but does not solve localisation or duplicate subpart proposals. At this operating point its aggregate material detection result is not better. It is retained for diagnosis, not promoted over the previous method.

Ignoring material entirely, existing proposals cover **7/8 glass and 0/12 porcelain source boxes** at IoU ≥0.50. The best overlap for any of the twelve porcelain sheds is 0.429. A classifier cannot recover source instances absent from its proposal set. This is proposal coverage, not a one-to-one detection metric.

The glass references often cover whole strings, while porcelain references in the mixed photo cover individual sheds. A model can propose an entire unannotated column or a visually plausible subpart and still have no matching source box. “FP” here means unmatched under the supplied reference protocol; exhaustive physical ground truth is not established. Do not infer a material for unlabelled structures from appearance alone, merge sheds into invented whole-string ground truth, or silently change the metric to obtain a better score.

## Viewer and reproducibility

English report: local **8775**, `runs/material_head/v1_20260828/report/index.html`. It shows both automatic methods on all eight original photos, source references as a separate dashed overlay, native candidate crops, raw logits, rejected outputs, development source crops and a separately labelled oracle gallery. Display thresholds do not change the fixed aggregate metrics. Ports 8771–8774 remain unchanged.

- Protocol: `configs/material_head_v1.json`, SHA256 `f99205856665c33ed4639cc940c1f0b936dabf6a2ac7aaa0346d1fa2bcab821b`.
- Manifest: `data/external/material_head_v1/manifest.json`, SHA256 `3bd69eede670d7e837314ee5f77bed5810a788fefa9df41513090bc7f837467f`.
- Classifier: `head.npz`, SHA256 `5e9037f738433ab684bcb2ead4150f6e1ff93feef66a1cdb4ad3b965a3fc9707`.
- Original-pixel audit: all 3,184 crops compared exactly against the original ZIP images on Roihu before inference.
- Independent report verification: original annotation classes/geometry, negative containment, crop hashes/dimensions/EXIF, saved embeddings and classifier arithmetic, all 400 finite losses, old detector/material/embedding hashes, oracle separation and automatic decisions.

Raw features, weights, logits, every rejected candidate, the original label files and run code snapshot are retained locally/remotely but excluded from the source-only GitHub branch. The preparer's later polygon-bounds grid optimisation preserves the same globally aligned valid negative candidates; the successful job snapshot retains the pre-optimisation implementation. Latest source code is not claimed to be byte-identical to the executed snapshot.

Do not resubmit job 920543. `submit_material_head.py` refuses an existing receipt/output or matching queue entry, and records an exclusive submission intent before calling Slurm. If a submission result is uncertain, inspect accounting and the receipt instead of retrying blindly.

## What this means for the Keen-style goal

This experiment supports a separated architecture but also identifies its current bottleneck. The next work should define whole-component versus shed/member units and improve localisation at that unit, with full-image misses retained in evaluation. More classifier steps alone will not fix missing proposals. Any automatic crop/tiling second pass must derive regions without target annotations, map boxes back to original pixels, and be compared under a fixed protocol; it has **not** run here.

Steelwork and pole-top are not added by a material head. Crossarm is a structural role, not proof of steel. A steel support member/assembly needs its own source evidence and extent definition. Pole-top should distinguish a visible shaft-tip keypoint from an upper-assembly extent; the existing geometry crop stays an unscored inspection region. Public data may replace unavailable user review, but only where it supplies compatible original labels. No new UK accuracy, polymer, corrosion, steelwork or pole-top recognition claim follows from these results.
