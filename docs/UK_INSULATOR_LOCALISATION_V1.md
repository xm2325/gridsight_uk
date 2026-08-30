# Prospective UK small-insulator localisation v1

## Question

Can fixed small-object inference techniques improve UK distribution-insulator localisation without adapting to, or selecting settings from, the acceptance images?

## Frozen acceptance boundary

- 8 Geograph images: 7 positives and 1 hard negative.
- 40 analyst-drawn visible-object boxes across 7 asset groups.
- Dave Hitchborne's two October 2007 distribution-pole images share one asset group.
- Images, source pages, authors, download URLs, licences, byte hashes, selection roles and exclusions are retained in the ignored source manifest.
- The boxes were frozen before model inference. They are analyst visible-object references, not expert-reviewed inspection ground truth.
- Transformer bushings, service enclosures, guy-wire strain apples, transmission-tower strings, railway equipment and building-mounted legacy equipment are outside this acceptance target.

## Predeclared arms

1. The existing EPRI component detector on the full image at 1280 pixels.
2. The same detector on the full image plus fixed 320-pixel tiles with 25% overlap.
3. The existing MPID three-material detector on the full image plus the same tiles, with all three outputs collapsed to generic insulator proposals for localisation only.
4. Fixed-priority proposal fusion: keep EPRI proposals first, then add only non-overlapping MPID proposals. Cross-model scores are never averaged or compared as probabilities.

Raw proposals are retained from score 0.01. Reporting points are fixed at scores 0.05 and 0.25 and IoU 0.30 and 0.50. The primary diagnostic is score 0.05 at IoU 0.30. There is one inference-only Roihu `gputest` job and no acceptance-driven retry, threshold tuning or training.

## Claim boundary

This is a prospective technique diagnostic on a small public UK cohort. It is not UK population accuracy, operational inspection validation, or a UK material benchmark. MPID class names are not reported as verified material on these images. Model scores are uncalibrated.
