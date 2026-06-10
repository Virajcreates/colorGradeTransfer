"""
Feature extraction module for neural color transfer.
Uses pretrained VGG19 to extract style features (Gram matrices)
from source and reference images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Dict, Optional


# VGG19 layers used for style feature extraction
STYLE_LAYERS = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1',
}


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from a pretrained VGG19 network.
    Only loads the feature layers (no classifier) to save ~50MB VRAM.
    """

    def __init__(self, layer_indices: Optional[List[int]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load VGG19 features only — no classifier head
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        if layer_indices is None:
            layer_indices = [int(k) for k in STYLE_LAYERS.keys()]

        self.layer_indices = sorted(layer_indices)
        max_layer = max(layer_indices) + 1

        # Only keep layers up to the deepest one we need
        self.features = nn.Sequential(*list(vgg.children())[:max_layer])
        self.features.eval()
        self.features.to(self.device)

        # Freeze all parameters — no training needed
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def preprocess(self, img_rgb: np.ndarray) -> torch.Tensor:
        """
        Convert an RGB uint8 numpy array to a normalized tensor for VGG.

        Args:
            img_rgb: RGB uint8 numpy array (H, W, 3).

        Returns:
            Normalized float32 tensor (1, 3, H, W) on target device.
        """
        # uint8 [0, 255] → float32 [0, 1]
        tensor = torch.from_numpy(img_rgb).float() / 255.0
        # (H, W, C) → (C, H, W)
        tensor = tensor.permute(2, 0, 1)
        # Normalize with ImageNet stats
        tensor = self.normalize(tensor)
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def extract_features(self, img_rgb: np.ndarray) -> Dict[int, torch.Tensor]:
        """
        Extract feature maps at specified VGG layers.
        Uses no_grad for VRAM efficiency. AMP is intentionally disabled
        because half-precision features cause overflow in Gram matrix
        computation (values like -87..+87 squared overflow float16 range).

        Args:
            img_rgb: RGB uint8 numpy array (H, W, 3).

        Returns:
            Dict mapping layer index → feature map tensor (float32).
        """
        x = self.preprocess(img_rgb)
        features = {}

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer_indices:
                features[idx] = x.clone().float()  # Ensure float32

        return features

    @staticmethod
    def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix of a feature map.
        The Gram matrix captures style information (correlations between
        feature channels).

        Always operates in float32 to prevent overflow.

        Args:
            feature_map: Tensor of shape (1, C, H, W).

        Returns:
            Gram matrix of shape (C, C), normalized by the number of elements.
        """
        # Ensure float32 to prevent overflow in matrix multiplication
        feature_map = feature_map.float()
        b, c, h, w = feature_map.size()
        features = feature_map.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram / (c * h * w)

    @torch.no_grad()
    def extract_style_features(self, img_rgb: np.ndarray) -> Dict[int, torch.Tensor]:
        """
        Extract Gram matrices (style features) at each style layer.

        Args:
            img_rgb: RGB uint8 numpy array (H, W, 3).

        Returns:
            Dict mapping layer index → Gram matrix tensor (float32).
        """
        feature_maps = self.extract_features(img_rgb)
        gram_matrices = {}
        for idx, feat in feature_maps.items():
            gram_matrices[idx] = self.gram_matrix(feat)
        return gram_matrices

