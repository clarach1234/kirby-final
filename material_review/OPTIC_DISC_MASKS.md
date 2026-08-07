# Optic-disc masks used for Task 3

Task 3 does not train a separate two-image network. The biomarker extractor
reads two aligned inputs for each validation case:

1. an RGB Task-2 artery/vessel/vein prediction; and
2. a binary optic-disc mask.

The mask is used to locate the optic-disc centre and diameter and therefore to
define the retinal measurement zones. The submitted Task-3 artifact is the
resulting `g_NNN.txt`, not the mask itself.

## Segmentation model and post-processing

The preliminary validation masks were generated from the public
[`pamixsun/segformer_for_optic_disc_cup_segmentation`](https://huggingface.co/pamixsun/segformer_for_optic_disc_cup_segmentation)
checkpoint, a REFUGE-based SegFormer optic-disc/cup model released under
Apache-2.0. The exact model revision is pinned to
`e1698e9f52e24cb6a7b2fecab4688852b89f77ef`.

[`generate_optic_disc_masks.py`](generate_optic_disc_masks.py) performs the
same steps used for the preliminary set:

1. resize/preprocess each CFP with the model's `AutoImageProcessor`;
2. resize the predicted logits back to the original image size;
3. combine every non-background label (disc rim plus cup);
4. retain the largest 8-connected component and fill its external contour;
5. save a grayscale 0/255 PNG with the original filename and dimensions;
6. save a green-contour overlay and `qc.csv` for manual review.

The separate [`optic_disc_segformer.py`](optic_disc_segformer.py) remains the
minimal offline implementation used by the hash-pinned end-to-end review
container. This standalone script documents the original mask-generation and
quality-control workflow without changing that final pipeline.

## Installation

Create and activate a Python environment, then install:

```bash
python -m pip install -r material_review/requirements-optic-disc.txt
```

The first online run downloads the pinned model. For a CUDA installation,
install the PyTorch build appropriate for the local CUDA version before the
remaining requirements.

## Reproduce the GAVE2 validation masks

From the repository root, with the official validation CFPs stored as
`g_051.png` through `g_100.png`:

```bash
python material_review/generate_optic_disc_masks.py \
  --images data/validation/images \
  --output work/validation/optic_disc_masks \
  --overlays work/validation/optic_disc_overlays \
  --start-index 51 \
  --expected-count 50 \
  --device auto
```

The default QC status is `review` unless the generated mask:

- has the same width and height as its source CFP;
- contains only values 0 and 255;
- has exactly one connected foreground component; and
- has an area ratio between 0.001 and 0.05.

An `ok` result is a structural sanity check, not proof of anatomical
correctness. Every overlay should still be inspected visually.

For the 50 preliminary validation cases, the recorded area ratios ranged from
`0.008636` to `0.019403`; all 50 masks had one component, none was marked
`review`, and all overlays were manually checked without an obvious
mislocalization. Generated CFPs, masks, overlays, model weights, and challenge
data are intentionally excluded from this public repository.

## Generate Task-3 text files

After Task-2 inference and mask review, run the deterministic biomarker
extractor:

```bash
python get_biomarker_kaggle.py \
  --av-dir work/task2_E \
  --disc-dir work/validation/optic_disc_masks \
  --output-dir work/task3_raw \
  --threshold 127 \
  --start-index 51 \
  --expected-count 50
```

It writes one `g_051.txt` through `g_100.txt` file per case containing CRAE,
CRVE, AVR, artery/vein density, and artery/vein fractal dimension.
