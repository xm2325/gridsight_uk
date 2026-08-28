# GridSight: material, steelwork and pole-top capability design

27 August 2026. This is an engineering design and a bounded development diagnostic, not a claim about Keen AI's proprietary model or the performance of its screenshot.

**28 August update:** the small public-data material demo and one fixed supervised pilot are complete; porcelain recognition remains unresolved. A larger explicitly labelled public substation source has now passed archive and sample-mask format checks. See the [current capability lab](docs/CAPABILITY_LAB.md) and [source audit](docs/SUBSTATION15_AUDIT.md). The historical UK-specific requirements below still apply to UK validation; they do not prohibit a separately labelled public-data demonstration or imply that the user must supply new photos. No new training was launched by this audit.

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

The English EPRI explorer is a separate presentation at `runs/keen_components/epri_components_en_20260827/report/index.html`, served on port 8771. It preserves the original data and three class toggles while linking to the current UK review on port 8772. The historical EPRI and v3 archives are unchanged.
