# Crossarm Grounding DINO v1

This bounded inference experiment tests whether an open-vocabulary crossarm proposal stage can recover UK candidates after the supervised EPRI specialist failed to emit any UK proposals.

## Frozen protocol

- Four prompts, two post-processing variants and eight raw-score thresholds were selected on 80 EPRI circuit-4 development images with publisher polygon boxes.
- The selected arm was frozen before UK inference: `the top support beam of a wooden utility pole.`, pole-associated, raw threshold `0.30`.
- It was run once on 27 source-provenance UK development images with no crossarm ground truth.
- UK v3 was not accessed. UK development output did not select the prompt, threshold or geometry.
- The job used Roihu `gputest` on one GH200. No local model inference ran.

## Verified result

Slurm job `961904` completed. The result SHA-256 is `0c9424dc9571dc37fc54c6036413e4f803587c3690a70cf351b820d606a28421`; the frozen-choices SHA-256 is `2794a62b09f8b17ca7d5d396de8f7324b58d1e77177b34874e1217784213de31`.

The selected arm achieved EPRI precision `0.880`, recall `0.489` and F1 `0.629`. On the 27 UK development images it emitted three candidates on two images. UK accuracy is unavailable because these images have no reference boxes.

Visual failure audit: one compact box overlaps a plausible pole-top support region; a second follows the pole shaft; a third spans a fallen wooden pole rather than a crossarm. These observations are not reference annotations.

## Decision

Do not promote this arm to UK v3 or the main overlay. It demonstrates limited target-domain proposal coverage, but two of three displayed candidates have the wrong morphology and target-domain recall remains unknown. The next crossarm experiment needs target-domain morphological supervision or a public dataset closer to UK wooden-pole assemblies, while preserving a disjoint acceptance cohort.
