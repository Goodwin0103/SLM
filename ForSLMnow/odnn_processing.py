from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from odnn_model import complex_pad_asymm


def build_spot_masks(layer_size: int, num_modes: int, focus_radius: int, energy_radius: int) -> np.ndarray:
    
    num_rows = int(np.floor(np.sqrt(num_modes)))
    num_cols = int(np.ceil(num_modes / num_rows))
    row_spacing = (layer_size - num_rows * 2 * focus_radius) / (num_rows + 1)
    col_spacing = (layer_size - num_cols * 2 * focus_radius) / (num_cols + 1)

    Y, X = np.ogrid[:layer_size, :layer_size]
    masks = []
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if len(masks) >= num_modes:
                break
            center_row = int(round((r - 1) * (2 * focus_radius + row_spacing) + row_spacing + focus_radius))
            center_col = int(round((c - 1) * (2 * focus_radius + col_spacing) + col_spacing + focus_radius))
            mask = (X - center_col) ** 2 + (Y - center_row) ** 2 <= energy_radius**2
            masks.append(mask)

    if len(masks) != num_modes:
        raise RuntimeError("Mismatch between generated spot masks and num_modes. Please adjust parameters.")
    return np.stack(masks, axis=0).astype(bool)



def detector_weights_from_intensity(intensity: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
   
    energies = np.array([float(intensity[mask].sum()) for mask in masks], dtype=np.float64)
    norm = np.linalg.norm(energies) + 1e-12
    weights = energies / norm
    return energies, weights


def pad_field_to_layer(field: torch.Tensor, target_size: int) -> torch.Tensor:
    
    if field.ndim < 2:
        raise ValueError("Expected at least 2 dimensions for the field tensor.")

    height, width = field.shape[-2:]
    dh = target_size - height
    dw = target_size - width

    if dh < 0 or dw < 0:
        raise ValueError("field_size 大于 layer_size，无法填充。")

    pt, pb = dh // 2, dh - dh // 2
    pl, pr = dw // 2, dw - dw // 2
    return complex_pad_asymm(field, pt, pb, pl, pr)


def pad_label_to_layer(label: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Pad a single-channel label tensor to the desired square size using zero padding.
    """
    if label.ndim != 3 or label.shape[0] != 1:
        raise ValueError("label must have shape (1, H, W).")

    _, height, width = label.shape
    dh = target_size - height
    dw = target_size - width
    if dh < 0 or dw < 0:
        raise ValueError("label size exceeds target_size; cannot pad.")

    pt, pb = dh // 2, dh - dh // 2
    pl, pr = dw // 2, dw - dw // 2
    return F.pad(label, (pl, pr, pt, pb))


def prepare_sample(image: torch.Tensor, label: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad input image and label tensors to the ODNN layer size.
    """
    padded_image = pad_field_to_layer(image.squeeze(0), target_size).unsqueeze(0)
    padded_label = pad_label_to_layer(label, target_size)
    return padded_image, padded_label
