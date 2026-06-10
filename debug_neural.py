"""Debug script for neural transfer NaN issue."""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from core.preprocess import bgr_to_rgb
from core.extractor import VGGFeatureExtractor

# Load images
src = cv2.resize(cv2.imread('test_output/source_warm.png'), (512, 512))
ref = cv2.resize(cv2.imread('test_output/reference_cool.png'), (512, 512))
source_rgb = bgr_to_rgb(src)
reference_rgb = bgr_to_rgb(ref)

device = 'cuda'
dev = torch.device(device)

extractor = VGGFeatureExtractor(device=dev)

# Force float32
extractor.features.float()

# Extract features and cast to float32
content_features = {k: v.float() for k, v in extractor.extract_features(source_rgb).items()}
style_grams = {k: v.float() for k, v in extractor.extract_style_features(reference_rgb).items()}

print("Content features dtypes:")
for k, v in content_features.items():
    print(f"  Layer {k}: {v.dtype}, shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

print("\nStyle Gram matrices:")
for k, v in style_grams.items():
    print(f"  Layer {k}: {v.dtype}, shape={v.shape}, range=[{v.min():.6f}, {v.max():.6f}]")

# Check for NaN in features
for k, v in content_features.items():
    if torch.isnan(v).any():
        print(f"  WARNING: NaN in content features layer {k}!")

for k, v in style_grams.items():
    if torch.isnan(v).any():
        print(f"  WARNING: NaN in style grams layer {k}!")

# Init output
output_tensor = extractor.preprocess(source_rgb).float().clone().requires_grad_(True)
print(f"\nOutput tensor: dtype={output_tensor.dtype}, range=[{output_tensor.min():.4f}, {output_tensor.max():.4f}]")

optimizer = torch.optim.Adam([output_tensor], lr=0.01)
content_layer = 19
style_layers = list(style_grams.keys())

# Run 5 iterations with detailed logging
for i in range(5):
    optimizer.zero_grad()

    x = output_tensor
    current_features = {}
    for idx, layer in enumerate(extractor.features):
        x = layer(x)
        if idx in extractor.layer_indices:
            current_features[idx] = x

    # Check current features
    for k, v in current_features.items():
        has_nan = torch.isnan(v).any().item()
        if has_nan:
            print(f"  Iter {i}, Layer {k}: NaN detected! dtype={v.dtype}")

    content_loss = F.mse_loss(current_features[content_layer], content_features[content_layer])

    style_loss = torch.tensor(0.0, device=dev)
    for layer_idx in style_layers:
        current_gram = extractor.gram_matrix(current_features[layer_idx])
        target_gram = style_grams[layer_idx]
        layer_loss = F.mse_loss(current_gram, target_gram)
        if torch.isnan(layer_loss):
            print(f"  Iter {i}: NaN in style loss for layer {layer_idx}")
            print(f"    current_gram range: [{current_gram.min():.4f}, {current_gram.max():.4f}], nan={torch.isnan(current_gram).any()}")
            print(f"    target_gram range: [{target_gram.min():.4f}, {target_gram.max():.4f}], nan={torch.isnan(target_gram).any()}")
        style_loss += layer_loss
    style_loss /= len(style_layers)

    total_loss = 1.0 * content_loss + 1e5 * style_loss
    print(f"  Iter {i}: content={content_loss.item():.6f}, style={style_loss.item():.8f}, total={total_loss.item():.4f}")

    total_loss.backward()

    # Check gradient
    if output_tensor.grad is not None:
        grad_max = output_tensor.grad.abs().max().item()
        grad_nan = torch.isnan(output_tensor.grad).any().item()
        print(f"    grad_max={grad_max:.6f}, grad_nan={grad_nan}")

    torch.nn.utils.clip_grad_norm_([output_tensor], max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        output_tensor.clamp_(-2.5, 2.5)

    print(f"    output range: [{output_tensor.min():.4f}, {output_tensor.max():.4f}]")
