# UK component annotation guide: EPRI transfer experiment

27 August 2026. This is a development annotation protocol, not completed labelling or permission to train.

The export contains all 27 UK photographs, including 20 with no supervised output at score 0.25. Its seven machine proposals include one insulator. Reviewed objects are empty and material fields are null. Empty model output does not establish absence of objects. Source photographs, authors, licences and SHA256 hashes are retained. The five provisional visual-prompt reference boxes are separate, not expert ground truth.

## Units and attributes

| Field | Annotation unit | Exclusions |
|---|---|---|
| pole | Visible extent of one shaft | Do not include its whole equipment assembly |
| crossarm | One distinct main support beam | Do not merge beams, braces, cables or equipment boxes |
| insulator | A complete string or an independently mounted unit | Do not mix whole units with individual discs; exclude fittings |
| material | Independent instance attribute: glass, porcelain, polymer, unknown | Colour, shape or model score alone cannot establish composition |
| pole-top | Not a trained class in this experiment | Agree shaft tip, assembly or inspection region before annotating |
| steelwork | Requires a separate structural definition and review | Do not rename wooden or unknown crossarms as steelwork |

Boxes use original-image pixels [x1, y1, x2, y2], clipped to the original dimensions. Record occlusion and truncation. Do not invent invisible extent. Keep nested components of different classes. Original EPRI polygons and derived boxes remain unchanged; later taxonomy harmonisation requires a new dataset version.

## Review every image

1. Inspect the full source image and all visible components, including images with no proposals. Remove false proposals and add missed objects. Explicitly record an independently checked empty image when appropriate.
2. Correct instance boundaries and units. Record difficult overlaps instead of inventing extra objects.
3. Use unknown if the evidence cannot support a material decision; record the reason. Null means unreviewed, not reviewed-and-unknown.
4. Record reviewer, time, class, box, material and supporting evidence. Seek a second reviewer with electrical asset expertise for material decisions. Preserve disagreements.
5. Record verified asset or survey identifiers when available. Similar filenames, locations or appearance alone do not prove asset identity.

## Evaluation boundary

All 27 images have already been inspected by models and are development data. A formal UK evaluation needs new, independently identified assets spanning scale, viewpoint, occlusion and target-free images. Report localisation, material classification on reviewed crops, automatic-crop end-to-end performance and rejection coverage separately. Do not train from unchecked proposals or report glass/porcelain accuracy before independent material evidence exists. Defects and safety ratings are outside the current labels.

The later v3 review offers additional draft steelwork and derived pole-top fields; those do not retrospectively change this three-class experiment or its frozen labels.
