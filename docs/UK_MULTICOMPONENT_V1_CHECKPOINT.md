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

## Completed fixed run — 2026-08-30

- Slurm job: `958020`
- State/exit: `COMPLETED`, `0:0`
- GPU: NVIDIA GH200 120GB
- Slurm elapsed time: 27 seconds
- Recorded inference elapsed time: 6.736 seconds
- Result SHA-256: `8955cad7b399c60680e27d68a93953ba1c04d775eb45e9b17656a5b9d2494ee9`
- Result status: `COMPLETE_UNSCORED_MULTICOMPONENT_DIAGNOSTIC`
- Gradient steps: 0
- Reference boxes accessed or used: false
- Multi-component accuracy computed: false
- Automatic retry or follow-on tuning: none

The strict English report builder verified the result pin, code snapshots,
frozen choices, material feature tensor, all 18 Grounding DINO raw tensors,
all nine record hashes, preserved v2 predictions and source-image hashes. It
rendered 36 panels and integrated the report into 8772 only after those checks.

### Observed outputs, not accuracy

- 21 pole detections across 5 of 9 images
- 5 crossarm detections on 1 of 9 images
- 14 insulator detections across 7 of 9 images
- 36 displayed steelwork candidates, with zero verified steelwork targets
- 8 of 14 material diagnostics emitted a specific diagnostic candidate, while all 14 final material outputs remained `unknown`
- 0 images contained both a pole and crossarm detection; all 9 pole-top outputs abstained

Visual review shows duplicated pole boxes, sparse or incorrect crossarm boxes,
and broad steelwork candidates that often cover a whole pole or background
structure. This is a verified real-output overlay, but it remains well short of
the supplied Keen AI result.

Browser QA verified the nine-option selector, Previous/Next navigation and
wraparound, loaded panels on the first and last images, English-only visible
text, the 8772 parent link, desktop two-column layout and zero console warnings
or errors. The browser report is:

`http://127.0.0.1:8772/report/multicomponent_v1/index.html`
