# Public substation material-source audit

28 August 2026. **Data audit only; no model was trained or evaluated on this source.**

This statement describes the audit stage. A subsequent bounded material experiment has since completed; see [v1 results and its orientation-handling limitation](SUBSTATION_MATERIAL_V1.md). The source audit and its original mask checks remain unchanged.

Source: Gomes and colleagues, *A Semantically Annotated 15-Class Ground Truth Dataset for Substation Equipment*, Data 8(7), 118 (2023). [Paper](https://www.mdpi.com/2306-5729/8/7/118) · [Publisher dataset, CC BY 4.0](https://zenodo.org/records/7884270).

The archive contains 1,660 image annotation files from one Brazilian substation with multiple camera types and acquisition dates. These are not 1,660 independent assets or UK distribution poles. Object labels can overlap: porcelain components may also lie inside an equipment polygon. The separate porcelain masks retain this information.

## Verified archive and observed counts

- File: `substation-semantic-dataset.zip`, 1,844,470,522 bytes.
- Publisher MD5 verified: `ca897cb85a3b944af6a1355e83530c32`.
- Computed SHA256: `7b4bbf72b48e437b8571584c0b8bee394fa6e156a622c6bf8276c4a61c8eb424`.
- `Porcelain pin insulator`: **26,450 polygons in 1,413 images**.
- `Glass disc insulator`: **3,180 polygons in 739 images**.
- 52,886 polygons across all 16 labels, including background.

These are direct archive counts, not copied paper totals, verified material accuracy or unique-instance counts across photographs. Do not relabel pin/shed units as whole strings, or automatically convert unlabelled frames into steelwork truth. There is no polymer, steelwork or pole-top class.

## Coordinate and mask verification

The supplied `json2png.py` uses points directly as `(x, y)` for OpenCV polygons; the paper's coordinate description can be misread as `(y, x)`. The source script was read, not executed. Our independent verifier rasterised the original polygons with the publisher palette and checked three samples selected by annotation area before model inference:

| Original image | Dimensions | Exact mask checks | Transposed porcelain IoU |
| --- | --- | --- | --- |
| `FLIR0335_rgb.jpg` | 1280 × 960 | All 3 masks, zero differing pixels | 0.0521 |
| `WhatsApp Image 2021-07-21 at 12.01.42.jpeg` | 720 × 1280 | All 3 masks, zero differing pixels | 0.0128 |
| `FLIR6829_rgb_AdgweI4.jpg` | 1280 × 960 | All 3 masks, zero differing pixels | 0.0201 |

Use `(x, y)`, integer truncation as in the publisher converter, and separate 14-class / 15-class / porcelain masks. Exact raster agreement verifies format compatibility on these samples; it does not certify annotation correctness. The samples have now been visually inspected and are development examples. Two are heavily backlit; size alone does not ensure visible material texture. Do not enhance dark source pixels and call the result new material evidence.

Local audit artifacts are deliberately excluded from Git: `runtime/substation15_audit/audit.json` and `samples/{manifest,verification}.json`. The sample manifest SHA256 is `70703f35f79c4cfb3d3f1dd6243fe5284aee1a300cb924fb55b4b1ed9572eaa6`. The original archive is cached only on Roihu; it need not be downloaded again.

## Reproducible audit

Place the publisher ZIP at `data/external/substation15_cache/substation-semantic-dataset.zip`, then run:

```bash
python3 scripts/audit_substation15.py --samples
python3 scripts/verify_substation15_samples.py
```

The first command uses the standard library, checks archive integrity and reads exact members without running archive-provided code. The second uses NumPy, Pillow and OpenCV for raster comparisons only; it does not import or execute an ML model. Original image, JSON and mask bytes are retained with hashes.

Before any next training job, audit exact/near duplicates and capture groups, visibility, polygon units and class overlap. A filename/date split is only a proxy for asset independence. A bounded public-data experiment can test material learning here; it cannot establish independent UK accuracy. No additional experiment is launched by this document.
