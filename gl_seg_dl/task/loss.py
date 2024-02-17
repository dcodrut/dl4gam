import torch
import torch.nn.functional as F


def dice_loss(preds, targets, axes=None, eps=1e-6):
    intersection = (preds * targets).sum(axis=axes)

    # compute the dice score(s)
    dice = (2. * intersection + eps) / (preds.sum(axes) + targets.sum(axes) + eps)

    return 1 - dice


class MaskedLoss(torch.nn.Module):
    def __init__(self, metric='L2'):
        super().__init__()
        self.metric = metric

    def forward(self, preds, targets, mask, samplewise=False):
        assert preds.shape == targets.shape and preds.shape == mask.shape

        # reduce all but the first (batch) dimension if needed
        axes = tuple(list(range(1, len(preds.shape)))) if samplewise else None

        # apply the mask
        mask = mask.type(preds.dtype)
        preds_masked = preds * mask
        targets_masked = targets * mask

        # reduce the loss to a single dimension or one per batch element (for dice it is already reduced)
        if self.metric in ('L2', 'L1', 'BCE'):
            # compute the loss
            reduction = 'none' if samplewise else 'sum'
            if self.metric == 'L2':
                loss = F.mse_loss(preds_masked, targets_masked, reduction=reduction)
            elif self.metric == 'L1':
                loss = F.l1_loss(preds_masked, targets_masked, reduction=reduction)
            elif self.metric == 'BCE':
                loss = F.binary_cross_entropy(preds_masked, targets_masked, reduction=reduction)

            loss = loss.sum(axes)
            den = mask.sum(axes)
            loss /= den
        elif self.metric == 'dice':
            loss = dice_loss(preds, targets, axes)
        else:
            raise NotImplementedError

        return loss
