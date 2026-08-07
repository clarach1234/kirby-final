# Task 3 — final biomarker extraction and calibration

## Source-E segmentation ensemble

Task 3 is deterministic post-processing of a separate A/V segmentation source
and optic-disc masks. Source E is an equal no-TTA ensemble of:

1. Task-2 topology-fine-tuned fold 0;
2. Task-2 baseline fold 1; and
3. Task-2 baseline fold 2.

The pipeline thresholds the source at 127/255, obtains an optic-disc/cup mask
from the hash-pinned SegFormer model, locates the largest disc contour, builds
the measurement zones, estimates vessel calibre from medial-axis distances,
and writes CRAE, CRVE, AVR, artery density, vein density, artery fractal
dimension, and vein fractal dimension.

## One global calibration rule

The same rule is applied to every preliminary case; no case-specific source
selection, manual correction, or preliminary label is used.

| Field | Final value |
|---|---|
| CRVE | raw Source-E value |
| AVR | 35% rank-preserving empirical-quantile calibration |
| CRAE | recomputed as `AVR × CRVE` |
| artery density | raw Source-E value |
| vein density | 35% rank-preserving empirical-quantile calibration |
| artery FD | raw Source-E value minus `0.013`, clipped to `[1.268001, 1.491455]` |
| vein FD | raw Source-E value plus `0.012`, clipped to `[1.294064, 1.470247]` |

For a field `x`, the calibrated quantile value is
`0.65 × x + 0.35 × Q_train(rank(x))`; therefore the ordering of the 50
preliminary predictions is preserved.

## Relevant code

- Biomarker extraction: [`get_biomarker_kaggle.py`](../get_biomarker_kaggle.py)
- Optic-disc inference: [`material_review/optic_disc_segformer.py`](../material_review/optic_disc_segformer.py)
- Original mask generation, overlays, and QC workflow: [`material_review/OPTIC_DISC_MASKS.md`](../material_review/OPTIC_DISC_MASKS.md)
- Calibration and final text serialization: `build_final_task3()` in [`material_review/pipeline.py`](../material_review/pipeline.py)
