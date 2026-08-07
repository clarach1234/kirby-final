# Task 2 — final multimodal vessel segmentation

## Input and selected models

Task 2 uses five channels: CFP RGB, the first channel of early FFA, and the
first channel of late FFA. Each selected model is CMRRWNet with base width 16
and one recurrent iteration. The final ensemble contains topology-fine-tuned
folds 0, 1, and 2 at best iterations **2880, 5508, and 6300**.

| Training setting | Value |
|---|---|
| Split | Legacy 4-fold compatibility split; selected folds 0–2 |
| Optimizer | Adam, weight decay 0 |
| Fine-tuning learning rate | `1e-5` |
| Loss | `RRLoss(OfficialScoreLoss)` |
| AMP / batch size / seed | enabled / 1 / 77 |
| Inference TTA | none |

`OfficialScoreLoss` combines A/V classification sensitivity, specificity, and
accuracy; vessel Dice; artery/vein centerline recall as a topology proxy;
auxiliary BCE; A/V-to-vessel consistency; background suppression; and A/V
mutual exclusivity. Its exact coefficients are in
[`train/official_loss.py`](../train/official_loss.py).

## Fixed final post-processing

For each fold prediction `P=(R,G,B)`, a boosted prediction is formed as
`(R, max(R,G,B), B)`. Raw folds are equally averaged into **Dc**; boosted folds
are equally averaged into **Db**.

1. Apply the conservative endpoint-only vein bridge to Db.
2. Assemble `R` and `B` from endpoint-bridged Db and `G` from Dc.
3. Derive a deterministic vein-support mask from Db and apply
   `B_final = B_current OR B_veinskel10`.

The final operation is additive only: it added **527,170** vein pixels across
the 50 cases, deleted zero vein pixels, and changed zero artery or vessel
bytes. Endpoint bridging selected 352 bridges and changed 812 pixels.

## Relevant code

- Model inference: [`get_predictions.py`](../get_predictions.py)
- Fold averaging: [`ensemble_predictions.py`](../ensemble_predictions.py)
- Endpoint bridge: `endpoint_bridge_vein()` in [`build_joint_exploration_submission.py`](../build_joint_exploration_submission.py)
- Vein support: `recover_vein_skeleton()` in [`build_experimental_submission.py`](../build_experimental_submission.py)
- Exact assembly: `run_all()` in [`material_review/pipeline.py`](../material_review/pipeline.py)
- Historical topology recipe: [`run_task2_topology_finetune_fold2.sh`](../run_task2_topology_finetune_fold2.sh)
