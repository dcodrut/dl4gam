import torch
import torch.nn.functional as F


def focal_loss(preds, targets, gamma=2, alpha=0.25, reduction='mean'):
    # based on: https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    # compute the binary cross-entropy loss
    bce = F.binary_cross_entropy(preds, targets, reduction='none')

    # compute the focal loss
    preds_t = preds * targets + (1 - preds) * (1 - targets)
    loss = bce * ((1 - preds_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


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
        if self.metric in ('L2', 'L1', 'BCE', 'focal'):
            # compute the loss
            reduction = 'none' if samplewise else 'sum'
            if self.metric == 'L2':
                loss = F.mse_loss(preds_masked, targets_masked, reduction=reduction)
            elif self.metric == 'L1':
                loss = F.l1_loss(preds_masked, targets_masked, reduction=reduction)
            elif self.metric == 'BCE':
                loss = F.binary_cross_entropy(preds_masked, targets_masked, reduction=reduction)
            elif self.metric == 'focal':
                loss = focal_loss(preds_masked, targets_masked, reduction=reduction)

            loss = loss.sum(axes)
            den = mask.sum(axes)
            loss /= den
        elif self.metric == 'dice':
            loss = dice_loss(preds, targets, axes)
        else:
            raise NotImplementedError

        return loss
