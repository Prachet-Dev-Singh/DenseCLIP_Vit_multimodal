# segmentation/denseclip/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss (SILog).
    Based on Eigen et al. (2014) - Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.
    https://arxiv.org/abs/1406.2283
    """
    def __init__(self, lambd=0.5, eps=1e-6, reduction='mean'):
        super().__init__()
        self.lambd = lambd
        self.eps = eps
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean' or 'sum'.")

    def forward(self, prediction, target, mask=None):
        """
        Args:
            prediction (Tensor): Predicted depth map, shape [B, 1, H, W].
            target (Tensor): Ground truth depth map, shape [B, 1, H, W].
            mask (Tensor, optional): Boolean mask where True indicates valid pixels,
                                     shape [B, 1, H, W]. Defaults to all pixels valid.
        Returns:
            Tensor: Calculated SILog loss (scalar).
        """
        # Ensure positive depth values for log
        prediction_valid = torch.clamp(prediction, min=self.eps)
        target_valid = torch.clamp(target, min=self.eps)

        # Calculate log difference (element-wise)
        log_diff = torch.log(prediction_valid) - torch.log(target_valid)

        # Apply mask if provided
        if mask is not None:
            if mask.shape != log_diff.shape:
                 # Try broadcasting if mask is [B, H, W] -> [B, 1, H, W]
                 if mask.shape == log_diff.shape[::2] + log_diff.shape[2:]:
                      mask = mask.unsqueeze(1)
                 else:
                      raise ValueError(f"Mask shape {mask.shape} incompatible with log_diff shape {log_diff.shape}")
            # Set log_diff to zero where mask is False (invalid pixels)
            log_diff = torch.where(mask, log_diff, torch.zeros_like(log_diff))
            num_valid = torch.sum(mask).item()
            if num_valid == 0:
                # Avoid division by zero if no valid pixels
                return torch.tensor(0.0, device=prediction.device, requires_grad=prediction.requires_grad)
            T = num_valid # Number of valid pixels
        else:
            # Consider all pixels valid if no mask
            T = log_diff.numel()
            if T == 0: return torch.tensor(0.0, device=prediction.device, requires_grad=prediction.requires_grad)


        # Calculate variance term (mean of squares)
        term1 = torch.sum(log_diff ** 2) / T

        # Calculate mean term squared
        term2 = (torch.sum(log_diff) ** 2) / (T ** 2)

        # Calculate final loss
        loss = term1 - self.lambd * term2

        # Ensure loss is non-negative (sqrt requires non-negative input, although variance should be >= lambda*mean^2)
        # Taking sqrt is sometimes omitted, optimizing variance directly is also common.
        # Let's return the variance-based term directly, as sometimes done.
        # If sqrt is needed: loss = torch.sqrt(torch.clamp(loss, min=0) + self.eps)

        # Handle reduction (though typically SILog is mean over batch)
        # Note: The division by T already provides a form of mean reduction over pixels.
        # If batch mean is desired (default), no further action needed here.
        # if self.reduction == 'sum':
        #     loss = loss * prediction.shape[0] # Multiply by batch size? Or just return sum over pixels? Needs clarification.

        return loss