# UK distribution component review: annotation guide

Version 3, 27 August 2026. All 27 photographs are development images. No draft, model proposal or derived region is approved for training by this tool.

## Object definitions

| ID | Name | Annotation unit and boundary |
|---|---|---|
| 0 | pole | The visible pole shaft, not the entire attached equipment assembly. Separate poles are separate objects. |
| 1 | crossarm | One main crossarm beam. Do not include unrelated cables, support braces or the whole pole. This class says nothing about material. |
| 2 | insulator | One complete independently mounted insulator or complete string, excluding connection fittings. Do not mix individual sheds/discs with complete units. |
| 3 | steelwork | A draft hypothesis for a coherent structural steel member or connected support assembly. Exclude wood, cables, insulators and equipment enclosures. Record which members belong together and the basis for identifying steel. Do not rename all crossarms as steelwork. |
| 4 | pole-top | An inspection region, not a physical component or a trained detector class. Define the intended upper-pole assembly window explicitly and correct its boundary. No detector confidence is permitted. |

Crossarm and steelwork can overlap when they describe different levels of the same assembly. An assembly-level steelwork box may contain background; future pixel-level corrosion work needs masks of the actual metal, not the entire bounding rectangle. No corrosion or safety label is inferred here.

Use source-image coordinates `[x1,y1,x2,y2]`. Record occlusion and truncation; do not invent invisible extent. Parent-pole references point to another reviewed draft in the same image, not a verified asset ID. Unknown parent assignments should remain unassigned.

## Material attributes

The insulator attribute is `glass`, `porcelain`, `polymer`, `unknown`, or `null`. `null` means not reviewed; `unknown` means reviewed but unresolved. Colour, shape and a language-model score alone are not sufficient evidence. Record resolution, ambiguity, obstruction or the available product/asset documentation.

The material diagnostic panel uses frozen SigLIP2 on **automatic detector crops**, with tight and contextual views. It ranks three material descriptions and three alternative descriptions (metal fitting, equipment box and background). Scores are cosine similarities, not probabilities. All final diagnostic materials stay `unknown`: this release has no labelled UK calibration set. A top-ranked hypothesis can help prioritise review but is not a confirmed material.

The native-pixel gate (short side at least 16 pixels and area at least 512 pixels) is a preregistered engineering heuristic, not proof that retained crops contain enough detail. Upscaling a crop does not add physical evidence. Context disagreement is another warning, not a calibrated error bound.

## Steelwork and pole-top review

DINO metal-crossarm/bracket queries propose possible structural members. The wording does not verify steel composition or assembly boundaries. A steelwork draft requires written material/assembly evidence before it can be marked ready for a second review. The evidence field itself is not expert certification.

Automatic pole-top search windows come from the preceding experiment's predicted pole boxes. Adopting one preserves its derived origin. Correct it, optionally associate it with a pole draft, and keep `entity_kind: inspection_region`. Do not import these boxes as detector ground truth or assign them the parent pole's confidence.

## Review workflow

1. Inspect the entire original image, including areas with no model proposals.
2. Reject false positives, correct class and extent, and add all visible omitted objects.
3. Separate components from regions and complete material/steelwork evidence or retain uncertainty.
4. Record reviewer, notes, visibility and parent relationships. Save and request an independent second review when appropriate.
5. Record verified asset/flight identities outside this prototype before defining a new train/development/test split.

Drafts are stored separately from model outputs. All exports retain `training_approved: false` and `expert_validated: false`. A workflow status never proves accuracy. This tool has no training endpoint. The v2 draft store is not silently migrated or overwritten; v3 drafts use their own schema and directory.

## Evaluation boundary

These photographs have been inspected through multiple models. They cannot be presented as an untouched holdout. Future evaluation must include independently identified assets, close and distant views, occlusion, negatives, and failures of the detector crop stage.

Measure localisation per class, material classification on correct reference crops, and end-to-end localisation plus material on automatic crops separately. Report rejection coverage and error among accepted cases together. Pole-top regions require an agreed region definition and region-coverage evaluation, not a borrowed detection confidence.
