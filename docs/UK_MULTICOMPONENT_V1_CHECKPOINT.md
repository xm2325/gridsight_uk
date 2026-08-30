# UK Multi-component Inference v1 Status

## Current checkpoint

- Local branch: `codex/keen-capability-lab-20260828`
- Protocol commit: `29b2c1c` (`Freeze UK multi-component inference protocol`)
- Report-renderer commit: `e6751ac` (`Add pinned multi-component report renderer`)
- Roihu project: `/scratch/project_2012997/keen_ai`
- Slurm account/partition for any approved run: `project_2012997` / `gputest`

## Verified on Roihu — 2026-08-30

Only these previously absent protocol files were synchronised:

- `configs/uk_multicomponent_inference_v1.json`
- `scripts/roihu_uk_multicomponent_inference_v1.py`
- `scripts/uk_multicomponent_inference_v1.sbatch`

The existing remote dependencies matched the local files before verification:

- `scripts/material_head_v2_common.py`: `16a382c574185a3f68737b9ed7c3051d587606697dbe93dd5647027cd8d3f63f`
- `scripts/roihu_demo_ablation.py`: `508e2a3cacb632af7b96d78b589c78ae59e109f5685fc36e11c629b347798b20`

The Roihu `--verify-only` preflight completed successfully:

```json
{
  "status": "PROTOCOL_VERIFIED",
  "images": 9,
  "asset_groups": 9,
  "preserved_insulator_predictions": 9,
  "gradient_steps": 0,
  "v3_reference_boxes_accessed_or_used": false,
  "output_exists": false
}
```

The preflight pins the UK v3 source manifest and source images, the preserved v2
insulator result and raw predictions, the EPRI component detector, the SigLIP2
release, the material head and threshold source, and the Grounding DINO release.
It also confirms the frozen UK v3 analyst boxes and roles are unavailable to the
models.

## Claim boundary

- Pole and crossarm outputs will be model detections with raw, uncalibrated scores.
- Insulator boxes will reuse the preserved v2 predictions; no duplicate detector pass is allowed.
- Material may show a frozen-head diagnostic candidate, but the final UK result remains `unknown` without independent instance evidence.
- Steelwork output is a `steelwork candidate`; steel composition is not verified.
- Pole-top output is an unscored geometry search region, not a physical component detection.
- No multi-component accuracy is computed on this cohort because the necessary truth is absent.

## Not yet run

No `gridsight-uk-multicomponent-v1` Slurm job was present or submitted at this
checkpoint. The protocol permits one fixed-budget, inference-only gputest job
after explicit user authorisation. It permits no gradient steps, automatic retry,
threshold selection from UK v3, or follow-on tuning.
