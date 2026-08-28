# GridSight capability lab

Source release: 28 August 2026. This branch adds the experiment runners, English review workbench, source audits and tests. It does not publish new model weights or datasets, deploy GitHub Pages, or establish production equivalence to Keen AI.

## What has actually run

| Experiment | Observed result | Boundary |
| --- | --- | --- |
| InsPLAD100 | Four inference arms on 100 real images, followed by separately recorded adaptation experiments | Historical runs; do not resubmit to recreate screenshots |
| EPRI components | Pole/crossarm/insulator detector: evaluation mAP50 rose from 0.11031 to 0.63368 | EPRI circuit-separated evaluation, not UK accuracy |
| UK capability diagnostic | 27 development images; 35 structural candidates at score 0.30; all 216 final material labels unknown | Unreviewed steelwork candidates; pole-top is an unscored derived region |
| UVInsDet public demo | Grounding DINO localises six selected main glass strings with IoU 0.942–0.986; frozen SigLIP2 references/text often disagree | Eight previously inspected demonstration inputs, not independent accuracy |
| UVInsDet supervised pilot | Fixed 10 epochs; 19 saved glass predictions and zero porcelain predictions above 0.05 | Failed pilot retained. Only ONE distinct porcelain training photo, even after weighting |

Roihu jobs 916391 (38 seconds) and 916435 (90 seconds, including training) completed successfully. Successful execution is not successful recognition. The two legacy adapted-detector outputs in the public demo predate the Roihu-only requirement; all subsequent model work was on Roihu. New material inference requires a CUDA allocation on `gputest` and rejects local CPU/MPS execution before importing torch.

## Source map

| Purpose | Entry points |
| --- | --- |
| Frozen InsPLAD diagnostic/adaptation | `prepare_insplad100.py`, `roihu_benchmark100.py`, `prepare_insplad_adaptation.py`, `roihu_insplad_train.py` |
| Component detection and evaluation | `prepare_keen_components.py`, `roihu_keen_components.py`, `keen_component_metrics.py` |
| English EPRI explorer | `build_keen_components_english.py`, `templates/keen_components_report_en.html` |
| UK structural/material diagnostics | `roihu_uk_capabilities.py`, `build_uk_capabilities_report.py`, `uk_capability_common.py` |
| Editable review with local draft API | `serve_uk_review.py`, `uk_review_common.py`, `templates/uk_component_review_v3.html` |
| Public material demo and failed pilot | `paper_material_demo.py`, `prepare_paper_supervised.py`, `roihu_paper_supervised.py`, `build_paper_material_report.py` |
| Reconstruct historical sample selection | `prepare_paper_selection.py` |
| New public material source audit | `audit_substation15.py`, `verify_substation15_samples.py`; see [audit](SUBSTATION15_AUDIT.md) |

Python entry points live under `scripts/`; protocols under `configs/`. Model/data acquisition scripts pin releases and verify hashes. Do not substitute similarly named weights. Each run retains its own source snapshot: the latest branch code is not claimed to be byte-identical to every historical runner.

## Run the source tests without a model

Python 3.11+ and Node.js are needed. The unit suite uses the standard library; image/report generation additionally needs Pillow, NumPy and Matplotlib. Optional mask audit verification uses OpenCV. Do not install a local inference stack just to run these checks.

```bash
python3 -m unittest discover -s tests -q
node --test tests/test_uk_review_ui.cjs
GRIDSIGHT_REVIEW_TEMPLATE=uk_component_review_v3.html node --test tests/test_uk_review_ui.cjs
```

One optional Python integration check skips when the completed EPRI report is not installed. The remaining checks cover split integrity, prediction matching, material abstention, draft validation, concurrent revisions and review UI edits. They do not rerun trained models.

## Existing local reports

The browser URLs are local services, not public GitHub links. A clean clone does **not** contain the saved runs required to render them.

| Port | Saved run | Presentation |
| --- | --- | --- |
| 8771 | `runs/keen_components/epri_components_en_20260827` | English EPRI/UK explorer; pole, crossarm and insulator filters |
| 8772 | `runs/uk_capabilities/v3_20260827` | English structural/material review, with local draft API |
| 8773 | `runs/paper_material_demo/v2_roihu_20260828` | English four-arm material demo, source polygons and failures |

If the corresponding service is not already running:

```bash
python3 -m http.server 8771 --bind 127.0.0.1 --directory runs/keen_components/epri_components_en_20260827
python3 scripts/serve_uk_review.py --port 8772 --run runs/uk_capabilities/v3_20260827
python3 -m http.server 8773 --bind 127.0.0.1 --directory runs/paper_material_demo/v2_roihu_20260828
```

Each command starts one foreground server; use a separate terminal. A generic static server cannot save 8772 drafts. Drafts remain separate from source predictions and are never automatically approved for training. Historical Chinese templates remain only for archive compatibility; 8771/8772 use their English successors.

## Reproduction versus resumption

Do not blindly execute all `.sbatch` files. First inspect existing manifests, completion receipts, checkpoints and Slurm state; several fixed runs have already completed. The scripts are research workflows with explicit prerequisites, not a turnkey installer. Existing run paths refuse incompatible state or validate completed artifacts. Preserve the original run before making a new protocol.

For UVInsDet, obtain the publisher's v1.0.0 ZIP from [Zenodo](https://zenodo.org/records/18197601), retain its licence/citation, then use `prepare_paper_selection.py` to reproduce the recorded 12-image selection. It verifies archive SHA256 and reproduces selection SHA256 `c2245a2f83cffd3a300e31cd9977307c873cc6247c7a0ac8201097dabf712014`. Preparation also needs the archived adapted checkpoints, pinned SigLIP2/Grounding DINO weights and publisher metadata at `runtime/target_sources/uvinsdet_zenodo.json`. Those artifacts are deliberately not in Git. Do not pretend a clean clone can recreate the trained baselines without them.

The Roihu jobs used CSC `python-pytorch/2.10`, project-local dependencies, offline verified model weights and the committed Slurm wrappers. `requirements-roihu-extra.txt` is additive; it is not a complete environment lock. Exact runtime versions and checkpoint hashes are recorded with the experiment outputs. Do not replace the cluster's CUDA/PyTorch stack with a local Apple environment.

## How the demo should improve next

1. **Material:** use the newly audited public source to obtain more distinct porcelain views. Preserve its polygon units and overlapping equipment/material labels. First compare a supervised crop classifier with the direct material detector under a fixed development protocol; include automatic-crop misses and abstentions. Neither raw cosine nor detector score is a calibrated material probability.
2. **Steelwork:** a crossarm is a role, while steel is a material. Keep wood crossarms, metal support members and enclosures distinct. Use member/assembly annotations for detection; use segmentation only if downstream metal area or corrosion is needed. The new dataset does not supply a steelwork class.
3. **Pole-top:** decide between shaft-tip keypoint and upper-assembly extent. The current geometry-derived region is useful navigation, not a verified detector. The new source does not supply pole-top labels.
4. **Presentation:** show original image, automatic component boxes, native crops, source evidence and failure examples separately. Preserve class toggles. Do not copy the screenshot's percentages or draw publisher polygons as model predictions.

The broader capability remains incomplete. A visually attractive single image is not proof of robust UK asset recognition. See [design and references](../KEEN_CAPABILITY_DESIGN_EN.md) for the architecture and acceptance boundaries.
