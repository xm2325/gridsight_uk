# GridSight-UK — UK overhead-line component vision under domain shift

A reproducible computer-vision portfolio project for electricity-transmission inspection. The repository intentionally separates **model capability**, **data/provenance quality**, and **claim readiness** rather than presenting a polished demo as production evidence.

## Capability lab — 28 August 2026

The separate research branch adds English component review, Roihu experiment runners, material-source audits and tests. See the [capability lab](docs/CAPABILITY_LAB.md) for verified outcomes, saved-run prerequisites and limitations, and the [new public material-source audit](docs/SUBSTATION15_AUDIT.md). These additions do not reopen the frozen UK holdout or claim production material recognition. Model weights, source datasets and human review drafts are not included in the source release.

## Closed-cycle evidence status — v3.9

The first development cycle is now **closed**. Model selection stopped before a two-image final holdout was unlocked; the final evaluation ran once in GitHub Actions and is now programmatically locked against automatic re-execution/tuning.

### Data used in the closed cycle

- training: **5 independent UK tower images / 55 component boxes**
- validation: **1 UK tower / 9 boxes**
- adaptive development showcase: **1 UK tower / 13 boxes** (`POS_2326530`)
- frozen final component holdout: **2 previously unseen towers / 22 boxes**, including **12 insulator references** (`POS_3437435`, England; `POS_7561805`, Scotland)
- final-holdout release SHA256: `5b7fddb7200523b861e4e0970e9ff0b54b0cde2426d9f461b71ec50c417efa9f`

Source images are hydrated from Geograph/Commons with recorded provenance and byte checks; third-party originals are not committed as ordinary repository assets.

## 2026 model stack evaluated

- **YOLOE-26n-seg** — open-vocabulary text and visual-prompt detection/segmentation
- **YOLO26n** — pretrained closed-set component fine-tuning/specialist experiments
- multi-source visual-prompt embeddings
- tower-relative component geometry priors
- validation-selected post-processing

Scratch Faster R-CNN and structured localisers are retained only as earlier baselines/failure diagnostics.

## What the experiments actually found

### 1. Tiny-data supervised localisation was not enough

A scratch detector trained on a handful of towers collapsed or produced very weak localisation. Later pretrained YOLO26 specialist/tiled runs were executable but still low precision/recall at this data scale. The project therefore did not treat a green training run as a successful detector.

### 2. YOLOE text prompting transferred better than a single visual reference

On the adaptive development tower, validation-selected YOLOE text prompting reached **6 TP / 12 FP / 0 FN at IoU≥0.30** (P=0.333, R=1.000, F1=0.500). A single-reference visual prompt failed to transfer, while a five-source aggregated visual embedding recovered some cross-tower robustness but did not beat text prompting.

### 3. A training-only geometry prior looked excellent on development — then failed the real holdout

A regularised tower-relative geometry prior learned from 30 training insulators reduced the adaptive-development output from **6 TP + 12 FP** to **3 TP + 0 FP** at IoU≥0.30 (P=1.000, R=0.500, F1=0.667). A symmetry-completion ablation was rejected because validation selected zero additions.

The complete method and operating point were then frozen **before** final-holdout inference.

### 4. One-shot frozen final holdout falsified the geometry assumption

GitHub Actions run `32103220833` evaluated the frozen method exactly once.

| Final holdout, pooled over 12 insulators | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw YOLOE text, IoU≥0.30 | 0.227 | **0.833** | **0.357** |
| Frozen text + geometry champion, IoU≥0.30 | **1.000** | 0.083 | 0.154 |
| Raw YOLOE text, IoU≥0.50 | 0.114 | **0.417** | **0.179** |
| Frozen text + geometry champion, IoU≥0.50 | 0.000 | 0.000 | 0.000 |

The geometry champion did **not** generalise. That negative result is preserved rather than retuned away.

See [`reports/v3_8_final_holdout_summary.json`](reports/v3_8_final_holdout_summary.json).

## Why did the geometry prior fail?

The post-final label-only audit found a real morphology shift:

- training geometry Mahalanobis d² q95: **6.567**, max **7.008**
- `POS_3437435` (England): **6/6** insulators exceed even the maximum training distance; median d² **39.629**
- its median width/tower-width is **0.251** vs **0.127** in training, while the relative shape ratio drops from **0.982** to **0.216**
- `POS_7561805` (Scotland) is much closer to training geometry: only **1/6** exceeds training q95

The England tower contains wide horizontal/strain-like insulators absent from the narrow training morphology. The open-vocabulary YOLOE model retained substantially more recall; the hand-engineered prior caused the stronger generalisation failure.

See [`reports/v3_9_morphology_shift.md`](reports/v3_9_morphology_shift.md).

## Evidence controls

- source metadata cannot become ground truth
- AI-assisted review is not represented as independent human consensus
- train/validation/development/final source identities are separated
- checkpoint, release and annotation identities are hashed
- generated masks are pseudo-labels until reviewed
- model/fused scores are not described as calibrated probabilities
- final-holdout inference is locked after the one-shot v3.8 run
- **no material, condition, corrosion, defect, failure or safety-risk performance is claimed**

The two final images still use assistant-provisional component references and the sample size is tiny; final metrics are therefore **portfolio evidence, not a production performance estimate**.

## Reproducibility

`.github/workflows/v23-online-yolo.yml` is now a **post-final audit workflow**. It verifies the evidence locks and runs the label-only morphology-shift audit; it deliberately cannot invoke the v3.8 final evaluator.

Important files:

- `data/final_holdout/final_holdout_freeze.json` — pre-inference holdout freeze
- `data/final_holdout/champion_freeze_v38.json` — frozen champion/config/thresholds
- `reports/v3_8_final_holdout_summary.json` — persisted one-shot final evidence
- `scripts/v39_post_final_morphology_audit.py` — post-final failure audit only
- `reports/v3_9_morphology_shift.md` — generalisation diagnosis

## Next cycle — v4

The old final holdout is **retired from model selection**. The next model-development cycle must:

1. acquire and annotate morphology-diverse tower designs, explicitly including horizontal/strain-insulator configurations;
2. separate tower/component morphology strata at split time;
3. use a **new preregistered holdout** that remains unseen during v4 development;
4. evaluate pretrained YOLO26/YOLOE and segmentation-assisted annotation without reusing the v3.8 final images for tuning;
5. report both coverage and localisation quality, not only attractive overlays.

The core lesson from v3.9 is deliberate: **foundation-model recall survived the domain shift better than a narrow structural prior.** The next improvement is data diversity and a new evaluation design, not another threshold tweak on the old holdout.
