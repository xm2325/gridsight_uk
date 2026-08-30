# GridSight: material, steelwork and pole-top capability design

27 August 2026. This is an engineering design and a bounded development diagnostic, not a claim about Keen AI's proprietary model or the performance of its screenshot.

**29 August update:** a fixed two-epoch MPID three-material detector and one prospective same-asset UK porcelain check are complete on Roihu. Internal MPID filename-family mAP50 reached 0.761, but the prospective porcelain string was classified as composite and only avoided an accepted wrong result because the matching box fell below the fixed score gate. A shorter wrong-material fragment received score 0.730 and passed the per-box gate. This proves that the direct detector is useful for proposals but not safe as the final UK material decision. See the [MPID detector record](docs/MPID_MATERIAL_DETECTOR_V1.md), [current capability lab](docs/CAPABILITY_LAB.md) and [source audit](docs/SUBSTATION15_AUDIT.md).

## Architecture to build towards

Use a hierarchy: image suitability → pole/component localisation → instance crops → material or structural attributes → reviewed parent/assembly relationships → asset aggregation when real asset IDs exist. Keep geometry-derived regions separate from physical components.

Keen's published distribution-pole case describes image filtering, parallel configuration classification and insulator detection, followed by asset aggregation. It reports using DETA for insulator detection. That is useful architectural evidence, but does not establish which model produced the supplied screenshot. [Keen AI case study](https://keen-ai.com/case-study/detecting-high-risk-distribution-poles-with-ai/)

## What this release actually adds

- A completely English review interface, independently versioned from the previous report.
- Frozen SigLIP2 on automatic DINO insulator proposals, with tight/context crops, native-pixel checks, raw embeddings and explicit abstention.
- Independent DINO metal-crossarm and metal-bracket queries for possible steelwork. These are additional hypotheses, not a relabelling of the prior crossarm detector.
- Editable pole-top inspection regions derived from existing predicted-pole search windows, without a detector score.
- Draft evidence for steelwork, parent-pole relationships, visibility flags, and schema validation that prevents a derived region from masquerading as a scored detector result.

The fixed experiment uses the same 27 UK development images. It does not train any weights, inspect an old frozen holdout, or rerun the completed InsPLAD/EPRI evaluations. There are no UK accuracy metrics without reviewed labels.

## Material recognition

### Observed diagnostic results

Slurm job 915794 completed successfully in 31 seconds (Python inference workflow: 12.97 seconds). Of 216 fixed detector proposals, 115 failed the native-pixel gate. The remaining 101 produced 202 exact source-image crops: 32 had tight/context disagreement, 22 consistently ranked a non-insulator alternative first, and 47 had consistent material hypotheses but no target-domain validation. These are diagnostic counts, not classification accuracy; every final material is unknown.

On close-up 7106830, the first clearly visible insulator's tight crop ranked polymer slightly above porcelain, while the contextual crop reversed that order. This illustrates instability without establishing its true material. The structural branch produced 35 proposals across 27 images at score 0.30. On 7106830 it roughly located the support assembly but included unrelated extent; on 5722811 its higher-scoring box covered almost the entire pole. These failures are retained in the report. Neither more candidates nor a higher score proves improved recognition.

**Recommended next trainable baseline:** retain a reviewed component detector, then compare a small supervised classification head on frozen SigLIP2 features against a direct joint material-aware detector. Use glass, porcelain, polymer and unknown/rejection semantics, with equipment-box, metal-fitting and other hard negatives. The two-stage design makes detector and material errors easier to diagnose; a direct joint detector is simpler but can conflate missing components and wrong material.

That direct comparison has now been executed with MPID. The result confirms the anticipated conflation: many UK predictions cover only a fragment, while the material class remains highly confident. The next material release must union material-agnostic and material-aware proposals, enforce a complete-assembly unit, and classify the dielectric crop separately. A rejection gate must include assembly completeness; raw score, native pixels and class margin alone are insufficient.

SigLIP2 supports image-text retrieval and transferable visual representations; this makes it a practical initial feature extractor, not a validated insulator-material classifier. The work is by Michael Tschannen, Alexey Gritsenko and colleagues. [Paper](https://arxiv.org/abs/2502.14786) · [Official Google model](https://huggingface.co/google/siglip2-base-patch16-naflex)

The current zero-shot experiment is a diagnostic baseline: symmetric material prompts, six alternatives, no colour shortcuts, no fitted thresholds, and no percentages labelled as confidence. Compare tight and contextual crops to expose reliance on background. All final labels remain unknown until target-domain validation exists.

A second feature baseline worth testing after labels exist is DINOv3 plus a linear head; it provides visual/dense representations but does not supply material names by itself. Meta FAIR's implementation is by Oriane Simeoni and colleagues and uses its own model licence; access and licence must be checked before downloading or deployment. It was not downloaded or run in this release. [Official DINOv3 repository](https://github.com/facebookresearch/dinov3)

Do not train from the current zero-shot argmax labels. The previous EPRI attribute audit found occlusion/truncation attributes, not usable insulator-material labels. The earlier UVInsDet audit found porcelain instances concentrated in one train image and one test image, with disc/shed units; it is not an adequate independent UK whole-insulator material benchmark.

## Steelwork

First agree whether the business output means individual metal members or connected steel support assemblies. Crossarm is a structural role, while steel is a material: a wooden crossarm must remain a crossarm without becoming steelwork. Metal fixtures and transformer enclosures are not automatically structural steelwork.

This release adds open-vocabulary structural hypotheses and a dedicated draft class with evidence. The next supervised experiment should include wooden beams, cables, equipment boxes and non-steel fixtures as hard negatives. Measure individual-member localisation and assembly-level grouping separately.
### Detection versus segmentation

Bounding boxes are adequate for an initial inventory demo. For condition assessment, segment the actual metal pixels and exclude background and attached equipment before learning corrosion or defect labels. A prompted segmenter can assist human mask annotation, but cannot prove material or condition. No segmentation or corrosion model was run in this release.
## Pole-top

The screenshot does not establish a precise pole-top annotation convention. Agree whether the target means the visible shaft tip, the upper assembly, or a crop used for inspection. They are different tasks. The current implementation deliberately uses an inspection region around a predicted pole, with no score. It is a useful navigation/crop mechanism but can be misplaced when the parent pole box is wrong.

After the definition is agreed, either annotate a shaft-tip keypoint with visibility flags and derive a consistently sized region, or annotate upper-assembly regions for a supervised detector. Include tilted/fallen poles, truncated tops and multiple-pole structures. Do not assume the top of the image is always the physical pole tip. Evaluate keypoint error or region coverage against the chosen definition.

## Data and acceptance requirements before training

The limiting resource is reviewed target-domain evidence, not more GPU time on unreviewed labels. First complete the development annotations, resolve disagreement and verify asset identities. Add new UK inspection views with clear provenance and licensing, independent material evidence where possible, and representative negatives. Choose the required sample size from per-class independent-asset counts and learning curves rather than promising that a fixed number of photographs is sufficient.

Freeze a new experiment only after that audit. Select model and rejection settings on development assets. Report per-class localisation metrics, material confusion on correct crops, automatic-crop end-to-end performance, and coverage versus error. Do not hide detector misses by evaluating only successful crops. No asset-level reliability or safety claim is justified by the current unlabelled development set.

## Reproducibility and local use

Inference protocol: `configs/uk_capabilities_v3.json`. Run: `runs/uk_capabilities/v3_20260827`. Model weights are pinned and verified against publisher hashes. Original token logits, boxes, embeddings, crops and source predictions are retained. `report/verification.json` records the checks actually completed; `report/ui_qa.json` records interface acceptance separately.

Serve with `scripts/serve_uk_review.py --port 8772 --run runs/uk_capabilities/v3_20260827`. This requires the dedicated local API server; a generic static server cannot save drafts. The old experiment is linked as immutable source evidence, not copied into a new training set. User drafts and isolated UI-test drafts are separate directories. No GitHub publication or further training is implied by this document.

## Next bounded experiment: proposal, not a launched job

First finish the target-domain annotation and evidence audit. The current review has no real saved draft objects, so there is no approved UK training set. More GPU time cannot resolve that missing evidence. Use existing development images to clarify annotation units; acquire better original close-ups when the source pixels cannot support a decision. Upscaling and synthetic enhancement must not create supposed material evidence.

Keep functional role and material as separate attributes in the next production schema. A crossarm can be wooden or metal; steelwork can contain several members with different structural roles. Define whether inventory counts refer to members or connected assemblies, and preserve parent-pole and assembly IDs. Do not count the same member twice simply because it has both a crossarm role and a steelwork attribute. The current five-class draft UI is a review aid, not a final mutually exclusive taxonomy.

After reviewed labels and independent asset splits exist, freeze a small comparison:

1. **Localisation:** retain the current supervised YOLO checkpoint as the reproducible baseline. Compare target-domain adaptation with an automatic pole-region second pass, using the same reviewed definitions. Choose any thresholds on development assets only. Include full-image misses and report results by native object size. Grounding DINO remains an annotation assistant and diagnostic comparator.
2. **Material:** compare frozen SigLIP2 embeddings plus a small supervised head against a joint material-aware detector. Fit rejection/calibration on development assets. Evaluate both correctly reviewed crops and automatically predicted crops, counting missed insulators in the end-to-end denominator. Include non-insulator hard negatives and unknown cases. Report per-material confusion, accepted coverage and error among accepted predictions.
3. **Steelwork and pole-top:** first evaluate corrected component/member boxes; use segmentation only when metal area or corrosion is the actual downstream target. For a shaft-tip definition, annotate a visible-tip keypoint and use a scale-normalised keypoint metric; for an assembly definition, annotate its extent and measure localisation/coverage instead. A geometry-derived navigation window remains a separate output without a detection score.

Do not select models by matching the confidence numbers in a marketing screenshot. Acceptance requires new independent UK assets, correct instance units, documented material evidence, retained failure examples and a reproducible runtime measurement. Numerical operating targets must be agreed for the intended inspection use and frozen before evaluation; none has yet been measured or promised here.

## 30 August evidence-lab addendum

Two bounded protocols now make the earlier ambiguity measurable without changing the UK drafts. The EPRI development audit derives a visible pole-shaft endpoint from publisher pole/crossarm polygons and compares preserved box and mask outputs. Mask geometry covers 10/18 eligible targets versus 8/18 for box geometry, with median normalised error 0.048 versus 0.057. Because the target is publisher-mask-derived rather than a physical-tip annotation, the UK `pole-top` output remains an unscored inspection region.

TTPLA supplies a paper-backed `tower-lattice` structural assembly class: the paper states lattice towers are composed of steel angle sections. A group-separated 60-image mirror subset was used for one fixed YOLOE segmentation comparison. On ten fixed test instances, open vocabulary matched 2 and the 20-epoch supervised final checkpoint matched 1 at score 0.25 / mask IoU 0.50; both stayed empty on four hard-negative images. This is a transmission-tower assembly demonstration, not distribution-pole crossarm material evidence. Full evidence and all-image visualisations are documented in `docs/STEELWORK_POLE_TOP_V1.md` and the English 8772 `report/upgrade/index.html` gallery.

The English EPRI explorer is a separate presentation at `runs/keen_components/epri_components_en_20260827/report/index.html`, served on port 8771. It preserves the original data and three class toggles while linking to the current UK review on port 8772. The historical EPRI and v3 archives are unchanged.

### Prospective UK material transfer

A source-evidenced prospective run now tests five new, asset-disjoint UK images after one frozen final-layer adaptation using six other UK images. On 18 analyst oracle regions, the adapted material head accepted 14 and classified 12 correctly (85.7% accepted accuracy, 77.8% coverage), versus 6/8 correct at 44.4% coverage for the frozen v2 head. These rectangles are source-assisted analyst regions rather than expert inspection labels, and the set has no UK polymer target.

The end-to-end result is much weaker: the existing EPRI component detector matched only 1/18 regions at the frozen score/IoU gates. This makes UK small-object localisation the immediate bottleneck. The verified report is `report/material_prospective/index.html`; the complete protocol and claim boundary are in `docs/UK_MATERIAL_PROSPECTIVE_V1.md`. No model score is shown as a probability and no deployment or UK population-accuracy claim is made.

### Prospective UK small-insulator localisation

An independent eight-image distribution-pole set with 40 analyst visible-object references was frozen before model inference. It includes seven positive images, one hard negative and seven asset groups; source pages, authors, licences, download URLs and byte hashes are retained. The references are not expert-reviewed inspection ground truth.

At the predeclared score 0.05 / IoU 0.30 point, the existing EPRI full-frame detector matched 1/40. Fixed 320-pixel tiling matched 2/40 with no false positives. The MPID full+tile proposal arm matched 2/40 with 22 false positives. Fixed-priority fusion matched 4/40 with the same 22 false positives, including four on the hard negative. At score 0.25 / IoU 0.50, fusion fell to 1/40 with three false positives.

This rules out simple tiling or uncalibrated material-proposal fusion as the next deployment upgrade. The next bounded model experiment should adapt a generic insulator localiser on UK development assets with asset-grouped labels and small-object sampling, while keeping these eight images untouched. A later acceptance test must use a new UK asset group because this set has now been consumed. The full failure gallery and raw Roihu outputs are at `report/localisation_prospective/index.html`; the frozen protocol is documented in `docs/UK_INSULATOR_LOCALISATION_V1.md`.

### Prospective UK target-domain adaptation

The proposed bounded adaptation has now been executed once on Roihu job 951851. A single-class YOLOE-26m specialist started from the preserved EPRI component checkpoint and trained for exactly ten epochs on a deterministic EPRI subset plus ten already-consumed UK development assets. The UK development portion contains 21 analyst visible-object boxes across seven positive assets and three negatives. Full frames and deterministic crops are repeated training views, not independent assets. The specialist does not replace the preserved pole/crossarm detector and does not classify material.

Before this training run, a second pixel-disjoint UK cohort was frozen from five Geograph assets: four positives, one hard negative and 14 analyst visible-object references. It was unavailable to training, checkpoint selection and operating-point selection. At the predeclared raw score 0.05 / IoU 0.30 point, the original EPRI full-plus-tile baseline matched 0/14 with one false positive. The adapted specialist matched 5/14 with three false positives. At IoU 0.50 it matched 4/14 with four false positives. Neither model produced a false positive on the hard negative, but the specialist completely missed the three references in the 33 kV asset.

This is a real prospective transfer gain, not a deployment result: primary recall is 35.7%, nine references remain missed, the cohort is small and its boxes are not expert-reviewed inspection truth. The next localisation release needs broader UK asset morphology, more native pixels or closer views, small-object-aware sampling and a larger asset-disjoint acceptance set. Do not tune thresholds or retry this frozen cohort. The complete English gallery is `report/localisation_adaptation/index.html`; the protocol and claim boundary are in `docs/UK_INSULATOR_ADAPTATION_V1.md`.
