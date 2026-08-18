# v4.0 pre-model assistant pixel review

**Review method:** `visual_assistant_single` on the exact JPEGs acquired by GitHub Actions.  
**Model-output exposure:** none before these role proposals.  
**Claim boundary:** these are provisional pixel-based review strata, not independent human consensus or engineering asset-class certification.

| Record | Country | Tower | Morphology / view | Component annotatability | Scale | Proposed v4 role | Review note |
|---|---|---|---|---|---|---|---|
| POS_7072688 | Scotland | yes | narrow suspension | low | small | reserve | tower clear; strings only weakly resolved |
| POS_7478407 | Scotland | yes | conventional suspension | not assessable | very small | reserve | component pixels too small for reliable GT |
| POS_7528296 | Scotland | yes | narrow suspension + clutter | low | small | reserve | foreground vegetation/clutter; low component resolution |
| POS_6610209 | Wales | yes | **complex mixed diagonal / strain / suspension** | high | large | **train** | close complex tower-head with mixed-angle strings; key morphology-diversity source |
| POS_4451265 | Wales | yes | multi-tower industrial scene | low | small | reserve | multiple towers and construction structures; primary-asset identity ambiguous |
| POS_5952661 | England | yes | suspension, oblique view | high | large | **train** | single tower with about six visible suspension strings |
| POS_8091164 | England | yes | **asymmetric three-arm suspension** | high | large | **new final holdout** | three very clear strings on an asymmetric head |
| POS_7945993 | England | yes | terminal/strain candidate | low | medium | reserve | unusual head but attachment/insulator pixels insufficient for reliable component GT |
| POS_4712773 | England | yes | conventional suspension | not assessable | very small | reserve | too distant for component GT |
| POS_8239540 | England | yes | **narrow compact suspension** | high | medium | **new final holdout** | six visible strings; clean narrow-tower morphology |
| POS_543992 | England | yes | conventional suspension, mid-scale | medium | medium | **validation** | six visible strings at moderate scale |
| POS_354803 | England | yes | canonical suspension, frontal | high | large | **validation** | six clear suspension strings; frontal canonical geometry |
| POS_7480474 | England | yes | suspension, oblique view | high | large | **train** | six visible strings; close oblique viewpoint |
| POS_1352733 | England | yes | low-angle partial tower | medium | large | **train** | strong low-angle/partial-tower view; useful viewpoint diversity |
| POS_8261555 | England | yes | narrow low-detail tower | not assessable | small | reserve | component pixels too small/obscured for reliable localisation GT |

## Split rationale

The previous v3 final holdout showed that a geometry prior learnt from morphologically similar suspension towers did not generalise to a wide horizontal/strain configuration. v4 therefore adds `POS_6610209` to training specifically because its tower head contains mixed-angle/diagonal/strain-like component geometry. It also adds low-angle and oblique views rather than only frontal towers.

The new final records were selected **before v4 model inference** because they are both component-annotatable while differing from each other: `POS_8091164` is an asymmetric three-arm design with three clear strings; `POS_8239540` is a narrow compact design with six clear strings. They are not selected because of model difficulty or model scores.

Far-scale, multi-tower and low-detail candidates are retained as reserve/stress sources instead of contaminating component-level final evaluation with annotation-visibility failures.
