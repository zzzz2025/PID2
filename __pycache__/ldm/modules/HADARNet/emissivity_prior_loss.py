import torch
import torch.nn as nn
from typing import Dict, Optional

class EmissivityPriorLoss(nn.Module):
    """Soft-constraint emissivity prior loss using semantic segmentation guidance.
    
    For each pixel, the predicted emissivity e must fall within the physically
   合理 range for its semantic class. Only pixels with high segmentation
    confidence (> threshold) are penalized, to avoid segmentation errors
    propagating into emissivity constraints.
    
    Loss formula:
        L_e_prior = mean_{i in valid} [ max(0, e_i - e_max)^2 + max(0, e_min - e_i)^2 ]
    
    This is a boundary-only penalty: pixels inside the valid range incur zero loss.
    Only out-of-range pixels are penalized, preserving per-pixel adaptivity within
    the physically-allowed envelope.
    """

    # Default emissivity ranges per class (min_e, max_e), reference: ASTER TES / HADAR
    DEFAULT_EMISSIVITY_RANGES: Dict[int, tuple] = {
        0: (0.0, 1.0),    # background / unknown: no constraint
        1: (0.08, 0.30),  # vehicle / metal body: low emissivity (highly reflective)
        2: (0.95, 0.99),  # human / skin: very high emissivity
        3: (0.85, 0.98),  # vegetation / trees: high emissivity
        4: (0.75, 0.90),  # road / asphalt: medium-high emissivity
        5: (0.80, 0.99),  # sky / water: high emissivity
    }

    def __init__(
        self,
        emissivity_ranges: Optional[Dict[int, tuple]] = None,
        confidence_threshold: float = 0.70,
        weight: float = 0.15,
    ):
        super().__init__()
        self.ranges = emissivity_ranges or self.DEFAULT_EMISSIVITY_RANGES
        self.conf_thresh = confidence_threshold
        self.weight = weight

    def forward(
        self,
        emissivity_pred: torch.Tensor,
        semantic_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute emissivity prior loss.
        
        Args:
            emissivity_pred: (B, 1, H, W) or (B, H, W) - TeVNet predicted emissivity e
            semantic_logits: (B, C, H, W) - SegFormer logits (C=number of classes)
        
        Returns:
            Scalar emissivity prior loss, scaled by self.weight
        """
        # Ensure emissivity_pred has channel dim
        if emissivity_pred.dim() == 3:
            emissivity_pred = emissivity_pred.unsqueeze(1)  # (B, 1, H, W)

        b, _, h, w = emissivity_pred.shape
        _, c, h_s, w_s = semantic_logits.shape

        # Resize semantic logits to match emissivity spatial dims if needed
        if h != h_s or w != w_s:
            semantic_logits = torch.nn.functional.interpolate(
                semantic_logits, size=(h, w), mode='bilinear', align_corners=False
            )

        # Get class probabilities and confidence
        probs = torch.softmax(semantic_logits, dim=1)  # (B, C, H, W)
        confidence, class_idx = probs.max(dim=1)       # (B, H, W), (B, H, W)

        # Build per-pixel (e_min, e_max) maps from class_idx
        e_min = torch.zeros((b, h, w), device=emissivity_pred.device, dtype=emissivity_pred.dtype)
        e_max = torch.ones((b, h, w), device=emissivity_pred.device, dtype=emissivity_pred.dtype)

        for cls_id, (lo, hi) in self.ranges.items():
            mask = (class_idx == cls_id)
            e_min = e_min.masked_fill(mask, lo)
            e_max = e_max.masked_fill(mask, hi)

        # Only compute loss for high-confidence pixels
        valid_mask = (confidence > self.conf_thresh).float()
        num_valid = valid_mask.sum() + 1e-8

        # Boundary-only soft constraint: only penalize out-of-range values
        excess = torch.clamp(emissivity_pred.squeeze(1) - e_max, min=0.0) ** 2
        deficit = torch.clamp(e_min - emissivity_pred.squeeze(1), min=0.0) ** 2
        loss_raw = (excess + deficit) * valid_mask

        return loss_raw.sum() / num_valid * self.weight

    def get_class_stats(
        self,
        emissivity_pred: torch.Tensor,
        semantic_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-class emissivity statistics for analysis (non-differentiable debug).
        
        Returns dict with mean, std, count, and constraint-violation rate per class.
        """
        if emissivity_pred.dim() == 3:
            emissivity_pred = emissivity_pred.unsqueeze(1)

        b, _, h, w = emissivity_pred.shape
        _, c, h_s, w_s = semantic_logits.shape

        if h != h_s or w != w_s:
            semantic_logits = torch.nn.functional.interpolate(
                semantic_logits, size=(h, w), mode='bilinear', align_corners=False
            )

        probs = torch.softmax(semantic_logits, dim=1)
        confidence, class_idx = probs.max(dim=1)

        stats = {}
        e = emissivity_pred.squeeze(1)

        for cls_id in self.ranges:
            mask = (class_idx == cls_id) & (confidence > self.conf_thresh)
            if mask.sum() == 0:
                stats[f'class_{cls_id}_mean'] = torch.tensor(0.0, device=e.device)
                stats[f'class_{cls_id}_std'] = torch.tensor(0.0, device=e.device)
                stats[f'class_{cls_id}_count'] = torch.tensor(0, device=e.device, dtype=torch.long)
            else:
                vals = e[mask]
                stats[f'class_{cls_id}_mean'] = vals.mean()
                stats[f'class_{cls_id}_std'] = vals.std()
                stats[f'class_{cls_id}_count'] = mask.sum()
                lo, hi = self.ranges[cls_id]
                viol = ((e[mask] < lo) | (e[mask] > hi)).float().mean()
                stats[f'class_{cls_id}_viol_rate'] = viol

        return stats