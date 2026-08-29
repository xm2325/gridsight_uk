# UK insulator material protocol v1

29 August 2026. This freezes the target semantics before acquiring another material dataset or running another model. It narrows the next development cycle to insulator localisation and material recognition; it does not erase the broader component experiments or claim that the Keen AI result has been reached.

## Target output

The automatic pipeline must first localise one complete visible insulator assembly, then classify the dielectric body as **glass**, **porcelain / ceramic**, or **polymer / composite**. **Unknown** is an abstention, not a forced material class. Metal caps and fittings can be inside the component extent but do not define the dielectric material.

The primary localisation unit is one physically continuous assembly. An individual shed or disc is not silently interchangeable with a full string, and parallel strings are not merged. Sources using incompatible units are quarantined or evaluated as source-specific auxiliary tasks.

## Why the current supervised result is not yet a material model

The completed YOLO26 supervised adaptation collapsed InsPLAD glass and polymer categories into one `insulator` target. It is therefore a localisation baseline. It can improve proposal recall relative to an open-vocabulary prompt without demonstrating glass-versus-polymer recognition.

The completed frozen SigLIP2 head is a crop-classification baseline. Given publisher boxes it classified many material crops correctly, but its automatic proposal branch covered seven of eight glass references and zero of twelve porcelain references at IoU 0.50. More classifier steps cannot recover an assembly that was not localised. Both original results and failures stay frozen.

## MPID candidate audit

[MPID](https://github.com/phd-benel/MPID) is the strongest current public candidate because its publisher describes tight, manually corrected YOLO boxes and three material folders. A published table reports 4,807 images and 7,850 instances: 3,773 glass instances in 1,612 images, 3,172 porcelain instances in 2,549 images, and 905 composite instances in 646 images.

MPID is not one independent dataset. It merges CPLID (China), STN PLAD (Brazil), IDID (United States), the Danish pylon-components dataset, one Vietnamese Roboflow dataset, and four sources whose country is reported as unknown. The repository declares the images CC BY 4.0, while its code repository uses MIT. Before training, each upstream source and its licence chain must be retained, exact and perceptual duplicates must be audited, and the split must be source-grouped. A random MPID image split would overstate generalisation.

MPID is UAV/transmission-heavy and supplies no UK deployment truth. It can improve material diversity and localisation, but it cannot validate UK performance.

The stable Zenodo v1 archives have now been downloaded directly on Roihu and verified against all three publisher MD5 values. They contain 5,019 images and 8,367 boxes, all under `train`; the advertised `valid` and `test` paths are absent. Labels and image pairs are syntactically complete and all normalised coordinates are valid. The archive has 108 two-copy exact glass duplicate groups and 104 two-copy exact porcelain groups. Removing one image from every exact group yields exactly the paper's 1,612 glass and 2,549 porcelain image counts; composite already matches 646. Filename-origin grouping also indicates additional non-exact families, especially glass and composite, so exact deduplication alone is insufficient.

A deterministic 24-image stratified visual audit finds that most boxes cover one continuous assembly/string, but the archive includes tiny unresolved targets, truncated components, substantial fitting context and at least one visually ambiguous composite-labelled example. MPID is therefore a filtered training-source candidate, not a self-contained benchmark. Its flattened export does not retain an authoritative upstream-source field, while several upstream datasets are nearly material-exclusive; material and camera/background source can be confounded.

## Model comparison

The next bounded comparison should use the same source-grouped splits and fixed whole-assembly unit:

1. the existing material-agnostic YOLO26/YOLOE localisation baseline;
2. an MPID-trained three-material detector as a diagnostic direct-class baseline;
3. material-agnostic localisation followed by the existing frozen SigLIP2 head;
4. material-agnostic localisation followed by one supervised three-material crop head.

The two-stage method is the primary design because it separates a missed component from a wrong material. A direct three-class detector remains useful as a comparison and pseudo-label consensus source, not as automatic truth.

## UK evidence gate

The 27 existing UK qualitative images can support target-domain adaptation and failure mining but have no component/material truth. They cannot produce UK accuracy. UK evaluation requires material evidence tied to the source page or another independent record, a box for the complete assembly, and a freeze before inference. Assistant or model proposals remain development annotations until supported by that evidence.

The UK pool is allowed to grow. `acquire_uk_material_sources.py` starts a provenance-preserved candidate pool with six explicit Geograph material pages. Two were already consumed by earlier adaptive development and cannot become new evaluation images; two Mayles Lane photos share one asset/work group and must stay in the same split; one old telegraph-pole record is auxiliary rather than primary electricity-network evidence. The remaining records are still candidates until their original pixels and compatible whole-assembly boxes pass review. Every record retains the page URL, original image URL, author, licence, page/image hashes, short evidence excerpt and asset group.

`build_uk_source_pool.py` combines these records with the existing 27-image UK qualitative pool without copying or relabelling pixels. The resulting 33-record manifest explicitly separates provenance-only images from source-evidenced material candidates. It is an expandable pool, not a frozen test set.

The public pixel decision record is `UK_MATERIAL_SOURCE_PIXEL_AUDIT_V1.json`. Acquired originals and the generated 33-record manifest stay in the ignored data cache; the acquisition and manifest builders are source-released.

The report must keep localisation score and material acceptance separate. A single Keen-style percentage is permitted only after a target-labelled calibration set defines and validates that probability. Until then, the UI may show a material label plus separate localisation evidence, or `unknown` with a rejection reason.

## Stop conditions before a new job

Do not submit training merely because MPID is downloadable. First verify its archive inventory, upstream attribution, label syntax, class mapping, whole-assembly compatibility, source groups and duplicate graph. A bounded Roihu job may then be preregistered with fixed splits, epochs, thresholds and comparison arms. No completed experiment is rerun and no Apple model execution is allowed.
