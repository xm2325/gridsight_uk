# UK Insulator Localisation Adaptation v1

This is a bounded single-class localisation specialist. It is designed to recover small UK distribution-line insulators while leaving the existing pole and crossarm detector unchanged.

## Data boundary

- Training and development combine a deterministic EPRI subset with 10 already-consumed UK development assets.
- The UK development references contain 21 analyst visible-object boxes across seven positive assets plus three negative assets. They are not expert-reviewed inspection truth.
- Full images and deterministic context crops are repeated training views, not independent assets.
- A second, pixel-disjoint UK cohort was frozen before adapted-model inference: five accepted images, 14 reference boxes, five asset groups and one hard negative.
- The second cohort cannot select training data, a checkpoint, a threshold or a retry.

## Fixed experiment

The YOLOE-26m single-class specialist starts from the existing EPRI component checkpoint and trains for exactly 10 epochs on Roihu `gputest`. Library development fitness selects the checkpoint. Only after its hash and all inference choices are frozen are the baseline and adapted checkpoints each evaluated once on the second UK cohort.

The primary point remains score 0.05 and IoU 0.30, copied from the earlier localisation study. Full-frame and fixed 320-pixel tiled inference are retained. Raw region predictions, unmatched predictions, failures and the hard-negative false positives are preserved.

## Claim boundary

This can show whether small, source-preserved UK development supervision improves a prospective technique check. It cannot establish UK population accuracy, expert inspection performance, material identity, defect status or calibrated probabilities.
