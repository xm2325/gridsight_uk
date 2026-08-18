# GridSight-UK v2.3 — GitHub Actions SOTA execution pack

Minimal frozen execution repository for the online YOLO26 / YOLOE-26 bridge.

## Frozen data
- train: 3 source images / 33 boxes
- validation: 1 source image / 9 boxes
- test: 1 source image / 13 boxes (`POS_2326530`)
- total: 5 source images / 55 boxes

## PR-triggered workflow
`.github/workflows/v23-online-yolo.yml`:
1. downloads official `yolo26n.pt` and `yoloe-26n-seg.pt` and records SHA256;
2. runs YOLOE-26 zero-shot text-prompt inference on the fixed held-out test image;
3. runs a deliberately short pretrained YOLO26 CPU fine-tune smoke on the frozen 3/1/1 source split;
4. uploads checkpoints, predictions, overlays and metadata as Actions artifacts.

Generated masks are pseudo-labels until human QA. Model scores are not calibrated probabilities. No material/condition/corrosion/defect/safety-risk claim is made by this pack.

## Repository hygiene
The five CC-licensed source images are not committed. GitHub Actions downloads the exact canonical Commons bytes from `data/image_sources.json` and verifies SHA-1 + SHA-256 before any model job.
