import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SemanticSegmentor(nn.Module):
    """Lightweight SegFormer-based semantic segmentor for emissivity prior guidance.
    
    Maps RGB image to 6-class semantic segmentation:
      0: background/unknown  -> no emissivity constraint
      1: vehicle/metal     -> e in [0.08, 0.3]
      2: human/skin        -> e in [0.95, 0.99]
      3: vegetation         -> e in [0.85, 0.98]
      4: road/asphalt       -> e in [0.75, 0.9]
      5: sky/water          -> e in [0.8, 0.99]
    """
    
    CLASS_NAMES = {
        0: 'background',
        1: 'vehicle',
        2: 'human',
        3: 'vegetation',
        4: 'road',
        5: 'sky_water',
    }
    
    # Emissivity ranges per class (min_e, max_e) for 8-14μm long-wave infrared
    EMISSIVITY_RANGES = {
        0: (0.0, 1.0),   # no constraint
        1: (0.08, 0.30),  # vehicle/metal - low emissivity
        2: (0.95, 0.99),  # human/skin - very high emissivity
        3: (0.85, 0.98),  # vegetation - high emissivity
        4: (0.75, 0.90),  # road/asphalt - medium-high emissivity
        5: (0.80, 0.99),  # sky/water - high emissivity
    }
    
    # Confidence threshold: only apply constraint if pixel confidence > this
    CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, encoder_name='mit_b0', pretrained=True):
        super().__init__()
        self.model = smp.create_model(
            'Unet',
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=6,
        )
        self.confidence_thresh = self.CONFIDENCE_THRESHOLD

    def forward(self, x):
        logits = self.model(x)
        return logits  # (B, 6, H, W)

    @staticmethod
    def get_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities over class dimension."""
        return torch.softmax(logits, dim=1)

    @staticmethod
    def get_confidence(logits: torch.Tensor) -> torch.Tensor:
        """Return max probability (confidence) per pixel."""
        probs = torch.softmax(logits, dim=1)
        confidence, _ = probs.max(dim=1)  # (B, H, W)
        return confidence

    @staticmethod
    def get_argmax_class(logits: torch.Tensor) -> torch.Tensor:
        """Return argmax class index per pixel."""
        return logits.argmax(dim=1).long()  # (B, H, W)