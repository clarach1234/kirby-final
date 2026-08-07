# Task 1 — final CFP-only A/V segmentation

## Final channel composition

Task 1 uses **CFP and ROI only**. Early- and late-phase FFA are never passed
to this task.

```
R = public R2-V2 AV artery probability
G = mean(kirby RRWNet fold 0..4 vessel probabilities), threshold-calibrated at 0.40
B = public R2-V2 AV vein probability
```

The external R2-V2 branch is fixed to repository commit
`7f6a8ea7a51782b1e0f89723a9ec137ba0a29913`; its public `av.pth` checksum is
recorded in `material_review/weights_manifest.json`. It uses the upstream
CFP-only inference implementation with 8-view rotation/flip TTA.

## Local vessel specialist

| Item | Setting |
|---|---|
| Architecture | RRWNet |
| Input / output | CFP RGB / artery-vessel-vein logits |
| Base channels / recurrent iterations | 16 / 5 |
| Folds | 5 equal folds; 10 validation and 40 training cases each |
| Optimizer | Adam, weight decay 0 |
| Initial learning rate | `1e-4` |
| Loss | `RRLoss(BCE3Loss)` |
| Batch size / workers | 1 / 2 |
| AMP / seed | enabled / 77 |
| Inference TTA | none |

The selected local checkpoints are fold 0–4 at iterations **53160, 52760,
67560, 63800, 62840**, respectively. Their SHA-256 values are the entries
`task1/fold_0.pth` through `task1/fold_4.pth` in the weight manifest.

## Relevant code

- Training engine: [`train/train.py`](../train/train.py)
- Historical baseline/resume recipes: [`run_task1_fold0.sh`](../run_task1_fold0.sh), [`run_task1_fold1.sh`](../run_task1_fold1.sh)
- Five-fold local inference: [`get_predictions.py`](../get_predictions.py)
- Channel assembly and calibrated threshold shift: `run_all()` in [`material_review/pipeline.py`](../material_review/pipeline.py)
- External AV retrieval: [`material_review/fetch_r2v2.sh`](../material_review/fetch_r2v2.sh)

At packaging, values are encoded as `{0, 1, 255}` without changing the
evaluator's exact-zero or 0.5 threshold masks. This is a file-size encoding,
not an additional segmentation post-processing rule.
