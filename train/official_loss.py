import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


class OfficialScoreLoss(nn.Module):
    """Differentiable, memory-conscious surrogate for the GAVE Task 2 score.

    Channel order is [artery, vessel tree, vein]. COR/INF are discrete graph
    metrics, so their 0.3 score share is approximated by artery/vein skeleton
    recall. The exact score must still be measured offline.
    """

    def __init__(self, bce_weight=0.20, consistency_weight=0.05, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.consistency_weight = consistency_weight
        self.eps = eps

    def _masked_mean(self, values, mask):
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _soft_dice(self, prediction, target, mask):
        prediction = prediction * mask
        target = target * mask
        intersection = (prediction * target).sum()
        return (2.0 * intersection + self.eps) / (
            prediction.sum() + target.sum() + self.eps
        )

    def forward(self, logits, vessels, mask, skeletons=None):
        roi = (mask[:, 0] > 0.5).to(logits.dtype)
        target = (vessels > 0.5).to(logits.dtype)
        probabilities = torch.sigmoid(logits)

        artery_logit, vessel_logit, vein_logit = (
            logits[:, 0], logits[:, 1], logits[:, 2]
        )
        artery, vessel, vein = (
            probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]
        )
        gt_artery, gt_vessel, gt_vein = (
            target[:, 0], target[:, 1], target[:, 2]
        )

        dice_loss = 1.0 - self._soft_dice(vessel, gt_vessel, roi)

        # A/V classification is meaningful only on unambiguous vessel pixels.
        class_mask = roi * gt_vessel * ((gt_artery + gt_vein) == 1).to(logits.dtype)
        artery_class_prob = torch.softmax(
            torch.stack((artery_logit, vein_logit), dim=1), dim=1
        )[:, 0]
        y = gt_artery
        tp = (artery_class_prob * y * class_mask).sum()
        tn = ((1.0 - artery_class_prob) * (1.0 - y) * class_mask).sum()
        fp = (artery_class_prob * (1.0 - y) * class_mask).sum()
        fn = ((1.0 - artery_class_prob) * y * class_mask).sum()
        sensitivity = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        accuracy = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        classification_loss = 1.0 - (
            0.3 * sensitivity + 0.3 * specificity + 0.4 * accuracy
        )

        # Low-memory topology proxy: encourage complete A/V centerline recall.
        if skeletons is None:
            topology_loss = logits.new_tensor(0.0)
        else:
            skeletons = (skeletons > 0.5).to(logits.dtype)
            recalls = []
            for pred_channel, skeleton_channel in (
                (artery, skeletons[:, 0]),
                (vein, skeletons[:, 2]),
            ):
                recalls.append(
                    self._masked_mean(pred_channel, skeleton_channel * roi)
                )
            topology_loss = 1.0 - torch.stack(recalls).mean()

        # Stable auxiliary supervision for all heads.
        av_known = roi * ((gt_artery + gt_vein) <= 1).to(logits.dtype)
        bce_a = self._masked_mean(
            F.binary_cross_entropy_with_logits(
                artery_logit, gt_artery, reduction='none'
            ),
            av_known,
        )
        bce_v = self._masked_mean(
            F.binary_cross_entropy_with_logits(
                vein_logit, gt_vein, reduction='none'
            ),
            av_known,
        )
        bce_vessel = self._masked_mean(
            F.binary_cross_entropy_with_logits(
                vessel_logit, gt_vessel, reduction='none'
            ),
            roi,
        )
        bce_loss = (bce_a + bce_v + bce_vessel) / 3.0

        av_union = 1.0 - (1.0 - artery) * (1.0 - vein)
        consistency_loss = self._masked_mean((av_union - vessel).abs(), roi)

        return (
            0.3 * classification_loss
            + 0.4 * dice_loss
            + 0.3 * topology_loss
            + self.bce_weight * bce_loss
            + self.consistency_weight * consistency_loss
        )

    def save_predicted(self, prediction, fname):
        vutils.save_image(self.process_predicted(prediction), fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())
