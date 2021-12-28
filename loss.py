"""
This code was written by Open-MMLab.
https://github.com/open-mmlab/mmediting
"""

import torch
import torch.nn as nn

def charbonnier_loss(pred, target,weight=None,reduction='mean',sample_wise=False, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f"Supported ones are: ['none', 'mean', 'sum']")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)