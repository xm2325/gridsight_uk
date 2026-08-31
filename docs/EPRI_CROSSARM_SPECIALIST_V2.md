# EPRI crossarm specialist v2

31 August 2026. This is a bounded source-domain adaptation and frozen-transfer audit. It improves EPRI crossarm detection, but it does not establish or improve verified UK crossarm accuracy.

## Protocol and execution

- Source: the existing EPRI Distribution Inspection Imagery pilot, CC BY-SA 4.0, with original URLs, ETags, JPEG hashes and publisher polygons retained.
- Split: circuits 1, 2, 3 and 6 for training; circuit 4 for development; circuits 5 and 7 for one frozen evaluation.
- Training view: all 320 independent training images remain. The 78 crossarm-positive images receive two additional training-only repetitions; 242 crossarm-negative images remain. The 476 samples are still only 320 independent source images.
- Model: one-class YOLOE-26m crossarm specialist initialized from the earlier three-class EPRI checkpoint, 40 epochs at 1280 pixels.
- Development selection: maximum F1 over the predeclared thresholds, selecting 0.10. Evaluation and UK images were not used for checkpoint or threshold selection.
- Roihu job 959950 completed successfully in 10:25 on one NVIDIA GH200. Job 959947 failed in two seconds before Python/model execution because the Slurm wrapper enabled `nounset` while CSC's environment initializer read `PS1`; the repaired wrapper and full preflight were recorded before 959950.

## Verified result

The immutable result SHA-256 is `6561ce49bfe8cae93c2897bae049e870914b6e2458b68dd0edf7c0a830fa148f`; the selected checkpoint SHA-256 is `f204583ece771ba3a42dd33404122cb00f94736c56558bad49973be57bc7b75b`.

On the frozen EPRI evaluation circuits, crossarm AP50 increases from 0.312 to 0.442 and AP50–95 from 0.113 to 0.216. At the displayed operating points, the old model at 0.05 has precision 0.416, recall 0.370 and F1 0.392; the specialist at its development-selected 0.10 has precision 0.698, recall 0.370 and F1 0.484.

The result does **not** transfer at that operating point: the specialist emits zero crossarm proposals on all nine source-preserved UK v3 images. UK reference boxes were unavailable to the model and were not used for checkpoint or threshold selection. This checkpoint is therefore retained as a verified source-domain experiment and is not promoted into the current UK multi-component overlay.

The English visual audit is served under `report/crossarm_v2/index.html`. It shows the old and specialist outputs side by side for all nine UK images, including explicit zero-output states.

## Next evidence-safe step

Do not extend this run or tune a lower threshold on UK v3. Use the separate 27-image UK development pool to choose a domain-robust crossarm proposal method, or acquire a public source with compatible ground-level distribution-pole labels and explicit licence/provenance. Freeze that choice before any further UK v3 comparison. Grounding or pseudo-label candidates may support training and review, but they must remain marked as machine-generated rather than truth.
