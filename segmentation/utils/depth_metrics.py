# segmentation/utils/depth_metrics.py

import torch
import numpy as np
import warnings

# ============================ #
# Metric Calculation Functions #
# ============================ #

@torch.no_grad() # Ensure no gradients are computed within metric functions
def compute_depth_errors(gt, pred, mask, min_depth=1e-3, max_depth=80.0, clamp_pred=True):
    """
    Computes standard depth estimation metrics between ground truth and prediction.

    Args:
        gt (torch.Tensor): Ground truth depth map. Shape [H, W] or [B, H, W].
        pred (torch.Tensor): Predicted depth map. Shape [H, W] or [B, H, W].
        mask (torch.Tensor): Boolean mask indicating valid pixels in GT.
                             Shape [H, W] or [B, H, W]. True for valid pixels.
        min_depth (float): Minimum depth value to consider for evaluation.
                           Pixels in GT less than this value are typically ignored.
        max_depth (float): Maximum depth value to consider for evaluation.
                           Pixels in GT greater than this value are typically ignored.
        clamp_pred (bool): Whether to clamp the prediction values to the [min_depth, max_depth]
                           range *before* calculating metrics. Common practice.

    Returns:
        dict: A dictionary containing the computed metrics (AbsRel, SqRel, RMSE, RMSE_log, a1, a2, a3),
              or None if no valid pixels are found after masking and range filtering.
    """
    # Ensure inputs are on the same device and float type
    gt = gt.float().to(pred.device)
    pred = pred.float()
    mask = mask.bool().to(pred.device) # Ensure mask is boolean

    # --- Flatten spatial dimensions if batch dim exists ---
    if gt.ndim == 3: # Batch dimension present
        gt = gt.view(gt.shape[0], -1)
        pred = pred.view(pred.shape[0], -1)
        mask = mask.view(mask.shape[0], -1)
    elif gt.ndim == 2: # Single image
        gt = gt.flatten()
        pred = pred.flatten()
        mask = mask.flatten()
    else:
        raise ValueError(f"Input tensors must have 2 or 3 dimensions (H, W) or (B, H, W). Got {gt.ndim}")

    # --- Select pixels based on initial validity mask ---
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    # --- Apply evaluation mask based on min/max depth range on GT ---
    eval_mask = (gt_valid >= min_depth) & (gt_valid <= max_depth)

    # --- Further filter based on the evaluation mask ---
    gt_eval = gt_valid[eval_mask]
    pred_eval = pred_valid[eval_mask]

    # Check if any pixels remain after filtering
    num_eval_pixels = gt_eval.numel()
    if num_eval_pixels == 0:
        warnings.warn("No valid pixels found for depth metric calculation after range filtering.", RuntimeWarning)
        return None # Return None if no pixels are valid for evaluation

    # --- Optionally clamp predictions to the valid range ---
    if clamp_pred:
        pred_eval = torch.clamp(pred_eval, min=min_depth, max=max_depth)

    # --- Calculate Metrics ---
    # Use torch operations for potential GPU acceleration

    # Threshold metric (Accuracy delta)
    thresh = torch.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))
    a1 = (thresh < 1.25).float().mean().item()    # Delta < 1.25
    a2 = (thresh < 1.25 ** 2).float().mean().item() # Delta < 1.25^2
    a3 = (thresh < 1.25 ** 3).float().mean().item() # Delta < 1.25^3

    # Error metrics
    diff = gt_eval - pred_eval
    diff_log = torch.log(gt_eval) - torch.log(pred_eval) # Uses clamped pred_eval if clamp_pred=True

    rmse = torch.sqrt(torch.mean(diff ** 2)).item()           # Root Mean Squared Error
    rmse_log = torch.sqrt(torch.mean(diff_log ** 2)).item()   # Root Mean Squared Log Error
    abs_rel = torch.mean(torch.abs(diff) / gt_eval).item()    # Absolute Relative Difference
    sq_rel = torch.mean(diff ** 2 / gt_eval).item()          # Squared Relative Difference

    return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log, a1=a1, a2=a2, a3=a3)


# =========================== #
# Metric Aggregation Helpers #
# =========================== #

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = [] # Optional: Store history

    def update(self, val, n=1):
        if val is None: return # Do not update if value is None
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        self.history.append(val)

    def get_average(self):
        return self.avg

    def get_history(self):
        return self.history


class DepthMetricsAggregator:
    """Helper class to aggregate depth metrics over multiple samples/batches."""
    def __init__(self, metrics_to_track=['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']):
        """
        Args:
            metrics_to_track (list[str]): List of metric keys expected from compute_depth_errors.
        """
        self.metrics_to_track = metrics_to_track
        self.meters = {m: AverageMeter() for m in self.metrics_to_track}
        self.valid_samples_processed = 0

    def update(self, metrics_dict):
        """
        Update the aggregator with metrics computed for a single sample or batch average.
        Args:
            metrics_dict (dict | None): Dictionary returned by compute_depth_errors or None.
        """
        if metrics_dict is None:
            return # Skip if compute_depth_errors returned None (e.g., no valid pixels)

        self.valid_samples_processed += 1
        for key in self.metrics_to_track:
            if key in metrics_dict:
                self.meters[key].update(metrics_dict[key])
            else:
                warnings.warn(f"Metric '{key}' not found in provided metrics_dict.", RuntimeWarning)

    def get_average_metrics(self):
        """
        Returns a dictionary containing the average value for each tracked metric.
        Returns None if no valid samples were processed.
        """
        if self.valid_samples_processed == 0:
            warnings.warn("Cannot get average depth metrics: No valid samples were processed.", RuntimeWarning)
            return None
        return {m: self.meters[m].get_average() for m in self.metrics_to_track}

    def reset(self):
        """Resets all average meters and the count of processed samples."""
        for meter in self.meters.values():
            meter.reset()
        self.valid_samples_processed = 0

    def __str__(self):
        """String representation of the current average metrics."""
        avg_metrics = self.get_average_metrics()
        if avg_metrics is None:
            return "Depth Metrics: No valid samples processed."
        # Format metrics for display (lower is better for errors, higher for accuracy)
        return (f"Depth Metrics (Avg over {self.valid_samples_processed} samples):\n"
                f"  AbsRel: {avg_metrics.get('abs_rel', float('nan')):.4f} | SqRel: {avg_metrics.get('sq_rel', float('nan')):.4f}\n"
                f"  RMSE: {avg_metrics.get('rmse', float('nan')):.4f} | RMSElog: {avg_metrics.get('rmse_log', float('nan')):.4f}\n"
                f"  d1 (a1): {avg_metrics.get('a1', float('nan')):.4f} | d2 (a2): {avg_metrics.get('a2', float('nan')):.4f} | d3 (a3): {avg_metrics.get('a3', float('nan')):.4f}")