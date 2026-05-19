from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.patches import Circle

from odnn_model import complex_crop, complex_pad
from ODNN_functions import generate_fields_ts
from odnn_processing import prepare_sample


def build_circular_roi_masks(
    height: int,
    width: int,
    num_spots: int,
    focus_radius: int,
    radius_scale: float = 1.2,
    generator: callable | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boolean masks for circular ROIs centred at evaluation regions.
    """
    radius = int(round(focus_radius * radius_scale))
    if generator is None:
        from ODNN_functions import create_labels
        generator = create_labels

    masks = []
    for k in range(num_spots):
        mask = generator(height, width, num_spots, radius, k + 1)
        masks.append(np.asarray(mask) > 0.5)
    stack = np.stack(masks, axis=0)
    union = np.any(stack, axis=0)
    return stack, union


def l2_normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (norms + eps)


def shift_complex_batch(batch: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    """
    Translate a batch of complex fields by (shift_y, shift_x) pixels with zero padding.
    Positive shift_y moves downward; positive shift_x moves right.
    """
    if shift_y == 0 and shift_x == 0:
        return batch

    _, _, height, width = batch.shape
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return torch.zeros_like(batch)

    real_imag = torch.view_as_real(batch)
    shifted = torch.zeros_like(real_imag)

    if shift_y >= 0:
        src_y = slice(0, height - shift_y)
        dst_y = slice(shift_y, height)
    else:
        src_y = slice(-shift_y, height)
        dst_y = slice(0, height + shift_y)

    if shift_x >= 0:
        src_x = slice(0, width - shift_x)
        dst_x = slice(shift_x, width)
    else:
        src_x = slice(-shift_x, width)
        dst_x = slice(0, width + shift_x)

    shifted[:, :, dst_y, dst_x, :] = real_imag[:, :, src_y, src_x, :]
    return torch.view_as_complex(shifted)


def sample_tensor_slices(tensor: torch.Tensor, kmax: int = 25) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        return tensor
    total = tensor.shape[0]
    count = min(total, kmax)
    indices = np.linspace(0, total - 1, count, dtype=int)
    return tensor[indices]


# =========================
# ROI mask helpers (FIXED)
# =========================
def _circle_masks_from_regions(
    shape: Tuple[int, int],
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    offset: Tuple[int, int] = (0, 0),
) -> list[np.ndarray]:
    """
    Build list of 2D boolean masks (H,W).
    shape can be (H,W), (B,H,W), (B,1,H,W) etc.
    """
    height, width = int(shape[-2]), int(shape[-1])
    yy, xx = np.ogrid[:height, :width]
    off_x, off_y = offset

    masks = []
    for x0, x1, y0, y1 in evaluation_regions:
        cx = int(round((x0 + x1) / 2.0 + off_x))
        cy = int(round((y0 + y1) / 2.0 + off_y))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        masks.append(mask)
    return masks


def sum_signal_energy_circle(intensity, evaluation_regions, radius, offset=(0, 0)):
    """
    Supports:
      - (H, W)
      - (B, H, W)
      - (B, 1, H, W)

    Returns:
      - float: summed energy over all detectors AND all batch samples
    """
    arr = np.asarray(intensity)

    # squeeze channel dim if present: (B,1,H,W) -> (B,H,W)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]

    if arr.ndim == 2:
        H, W = int(arr.shape[-2]), int(arr.shape[-1])
        masks = _circle_masks_from_regions((H, W), evaluation_regions, radius, offset)
        return float(sum(arr[mask].sum() for mask in masks))

    if arr.ndim == 3:
        B, H, W = arr.shape
        masks = _circle_masks_from_regions((H, W), evaluation_regions, radius, offset)
        total = 0.0
        for b in range(B):
            Ib = arr[b]
            total += sum(Ib[mask].sum() for mask in masks)
        return float(total)

    raise ValueError(f"Unsupported intensity shape: {arr.shape}")


def spot_energy_ratios_circle(
    intensity: np.ndarray,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    offset: Tuple[int, int] = (0, 0),
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IMPORTANT:
      - This function expects a SINGLE 2D map (H,W).
      - If you have batch, index it first: intensity = intensity[b] or intensity[b,0]
    """
    arr = np.asarray(intensity)

    # squeeze channel dim if present: (1,H,W) -> (H,W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"spot_energy_ratios_circle expects 2D (H,W), got {arr.shape}")

    masks = _circle_masks_from_regions(arr.shape, evaluation_regions, radius, offset)
    energies = np.array([float(arr[mask].sum()) for mask in masks], dtype=np.float64)
    total = float(arr.sum()) + eps
    ratios = energies / total
    return energies, ratios

import torch


@torch.no_grad()
def _make_circle_mask(h: int, w: int, r: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)).to(torch.float32)


@torch.no_grad()
def region_energies_circle(
    I_bhw: torch.Tensor,  # (B, H, W) float
    evaluation_regions: list[tuple[int, int, int, int]],
    detect_radius: int,
) -> torch.Tensor:
    """
    返回每个 ROI 的“绝对能量”(不归一化): (B, M)
    ROI 内用圆形 mask 积分（与你现有 detect_radius 口径一致）。
    """
    B, H, W = I_bhw.shape
    M = len(evaluation_regions)
    out = torch.zeros((B, M), device=I_bhw.device, dtype=torch.float32)

    for mi, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        patch = I_bhw[:, y0:y1, x0:x1]
        hh, ww = patch.shape[-2], patch.shape[-1]
        cmask = _make_circle_mask(hh, ww, float(detect_radius), device=I_bhw.device)
        out[:, mi] = (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))

    return out


@torch.no_grad()
def wavelength_energy_ratios_in_det(
    I_blhw: torch.Tensor,  # (B, L, H, W)
    evaluation_regions: list[tuple[int, int, int, int]],
    detect_radius: int,
    *,
    L: int,
    num_modes: int,
) -> torch.Tensor:
    """
    波长间能量比例（仅统计“所有模式检测 ROI”内的能量）:
      1) 每个波长：把该波长对应的 num_modes 个 ROI 能量求和 -> E_det[:, wl]
      2) 在波长维归一化 -> p_wl (B, L)

    evaluation_regions 的索引约定需与你主脚本一致：
      label_idx = mode_k * L + wl_idx
    """
    B, Lloc, H, W = I_blhw.shape
    if Lloc != L:
        raise ValueError(f"I_blhw has L={Lloc}, but expected L={L}")

    E_det = torch.zeros((B, L), device=I_blhw.device, dtype=torch.float32)

    for wl_idx in range(L):
        wl_regions = [evaluation_regions[k * L + wl_idx] for k in range(num_modes)]
        E_k = region_energies_circle(I_blhw[:, wl_idx].to(torch.float32), wl_regions, detect_radius)
        E_det[:, wl_idx] = E_k.sum(dim=1)

    p_wl = E_det / (E_det.sum(dim=1, keepdim=True) + 1e-12)
    return p_wl


def build_superposition_eval_context(
    num_samples: int,
    *,
    num_modes: int,
    field_size: int,
    layer_size: int,
    mmf_modes: torch.Tensor,
    mmf_label_data: torch.Tensor,
    batch_size: int,
    second_mode_half_range: bool = False,
    rng_seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Create datasets and bookkeeping tensors for evaluating on random superpositions.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive for superposition evaluation.")

    if rng is not None and rng_seed is not None:
        raise ValueError("Specify either rng or rng_seed, not both.")

    local_rng: Optional[np.random.Generator] = rng
    if local_rng is None and rng_seed is not None:
        local_rng = np.random.default_rng(rng_seed)

    if local_rng is not None:
        amplitudes = local_rng.random((num_samples, num_modes), dtype=np.float32)
    else:
        amplitudes = np.random.rand(num_samples, num_modes).astype(np.float32)

    norms = np.linalg.norm(amplitudes, axis=1, keepdims=True)
    zero_norm = norms <= 1e-12
    if np.any(zero_norm):
        amplitudes[zero_norm, 0] = 1.0
        norms = np.linalg.norm(amplitudes, axis=1, keepdims=True)
    amplitudes = amplitudes / (norms + 1e-12)

    if local_rng is not None:
        phases = local_rng.random((num_samples, num_modes), dtype=np.float32) * 2 * np.pi
    else:
        phases = np.random.rand(num_samples, num_modes).astype(np.float32) * 2 * np.pi

    phases[:, 0] = 0.0
    if second_mode_half_range and num_modes >= 2:
        if local_rng is not None:
            phases[:, 1] = local_rng.random(num_samples, dtype=np.float32) * np.pi
        else:
            phases[:, 1] = np.random.rand(num_samples).astype(np.float32) * np.pi
    phases[:, 0] = 0.0

    complex_weights = amplitudes * np.exp(1j * phases)

    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, mmf_modes, num_samples, num_modes, field_size
    ).to(torch.complex64)

    amp_ts = torch.from_numpy(amplitudes.astype(np.float32))
    amp_ts_energy = amp_ts ** 2
    label_maps = torch.tensordot(amp_ts_energy, mmf_label_data, dims=([1], [2])).to(torch.float32)
    label_maps = label_maps.unsqueeze(1)  # (N, 1, H, W)

    dataset_pairs = []
    for idx in range(num_samples):
        padded_image, padded_label = prepare_sample(image_data[idx], label_maps[idx], layer_size)
        dataset_pairs.append((padded_image, padded_label))

    images = torch.stack([sample[0] for sample in dataset_pairs], dim=0)
    labels = torch.stack([sample[1] for sample in dataset_pairs], dim=0)
    tensor_dataset = TensorDataset(images, labels)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    if num_modes > 1:
        amplitudes_phases = np.hstack((amplitudes, phases[:, 1:] / (2 * np.pi)))
    else:
        amplitudes_phases = amplitudes.copy()

    return {
        "dataset": dataset_pairs,
        "tensor_dataset": tensor_dataset,
        "loader": loader,
        "image_data": image_data,
        "amplitudes": amplitudes,
        "phases": phases,
        "amplitudes_phases": amplitudes_phases,
    }


def generate_superposition_sample(
    num_modes: int,
    field_size: int,
    layer_size: int,
    mmf_modes: torch.Tensor,
    mmf_label_data: torch.Tensor,
) -> Dict[str, Any]:
    """
    Sample random normalized amplitudes/phases and build padded tensors for inference.
    """
    amplitudes = np.random.rand(num_modes).astype(np.float32)
    if not np.any(amplitudes):
        amplitudes[0] = 1.0
    amplitudes = amplitudes / (np.linalg.norm(amplitudes) + 1e-12)

    phases = np.random.rand(num_modes).astype(np.float32) * 2 * np.pi
    complex_weights = amplitudes * np.exp(1j * phases)

    weights_ts = torch.from_numpy(complex_weights.astype(np.complex64)).unsqueeze(0)
    field_small = generate_fields_ts(weights_ts, mmf_modes, 1, num_modes, field_size)[0].detach().cpu()

    amp_ts = torch.from_numpy(amplitudes.astype(np.float32))
    label_map = (mmf_label_data * (amp_ts**2).view(1, 1, -1)).sum(dim=2)
    label_tensor = label_map.unsqueeze(0)

    padded_image, padded_label = prepare_sample(field_small, label_tensor, layer_size)

    return {
        "amplitudes": amplitudes,
        "phases": phases,
        "complex_weights": complex_weights,
        "field": field_small,
        "label_map": label_map,
        "padded_image": padded_image,
        "padded_label": padded_label,
    }


def infer_superposition_output(
    model: torch.nn.Module,
    padded_image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward the ODNN model with a single padded complex field and return the predicted intensity map.
    """
    model.eval()
    input_batch = padded_image.unsqueeze(0).to(device, dtype=torch.complex64)
    with torch.no_grad():
        output = model(input_batch)
    return output[0, 0].detach().cpu()


def _get_sample_from_dataset(dataset, idx: int):
    """
    Retrieve (image, label) from either a list of tuples or a TensorDataset.
    """
    if isinstance(dataset, list):
        return dataset[idx]
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[0][idx], dataset.tensors[1][idx]
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def save_prediction_diagnostics(
    model: torch.nn.Module,
    dataset,
    *,
    evaluation_regions,
    layer_size: int,
    detect_radius: int,
    num_samples: int,
    output_dir: Path,
    device: torch.device,
    tag: str,
) -> list[Path]:
    """
    Save side-by-side plots of label vs. prediction with an amplitude/energy bar graph.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    model.eval()
    sample_indices = list(range(min(num_samples, len(dataset))))

    for sample_idx in sample_indices:
        image, label = _get_sample_from_dataset(dataset, sample_idx)

        image_batch = image.unsqueeze(0).to(device, dtype=torch.complex64)
        label_map = label[0].detach().cpu().numpy()

        with torch.no_grad():
            pred = model(image_batch)

        # NOTE: if model returns (B,L,H,W), this will NOT work as-is.
        # Keep your diagnostics for single-wavelength use or adapt similarly to evaluation.
        pred_map = pred[0, 0].detach().cpu().numpy()

        def region_energy(arr: np.ndarray) -> np.ndarray:
            vals = []
            yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1]]
            radius_px = max(1, int(round(detect_radius / 2.0)))
            for (x0, x1, y0, y1) in evaluation_regions:
                cx = int(round((x0 + x1) / 2.0))
                cy = int(round((y0 + y1) / 2.0))
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
                vals.append(float(arr[mask].sum()))
            return np.asarray(vals, dtype=np.float64)

        label_weights = region_energy(label_map)
        pred_weights = region_energy(pred_map)

        def safe_normalize(weights: np.ndarray) -> np.ndarray:
            s = float(weights.sum())
            return weights / s if s > 0 else weights

        label_norm = safe_normalize(label_weights)
        pred_norm = safe_normalize(pred_weights)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        im0 = axes[0].imshow(label_map, cmap="inferno")
        axes[0].set_title("Label")
        axes[0].set_axis_off()
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pred_map, cmap="inferno")
        axes[1].set_title("Prediction")
        axes[1].set_axis_off()
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        diff_map = np.abs(pred_map - label_map)
        im2 = axes[2].imshow(diff_map, cmap="magma")
        axes[2].set_title("|Pred - Label|")
        axes[2].set_axis_off()
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        radius_px = max(1, int(round(detect_radius / 2.0)))
        for ax in (axes[0], axes[1], axes[2]):
            for (x0, x1, y0, y1) in evaluation_regions:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                circ = Circle(
                    (cx, cy),
                    radius=radius_px,
                    linewidth=0.8,
                    edgecolor="cyan",
                    facecolor="none",
                    linestyle=":",
                    alpha=0.9,
                )
                ax.add_patch(circ)

        x = np.arange(len(pred_weights))
        width = 0.35
        axes[3].bar(x - width / 2, label_norm, width=width, label="Label", color="tab:blue")
        axes[3].bar(x + width / 2, pred_norm, width=width, label="Pred", color="tab:orange")
        axes[3].set_xticks(x)
        axes[3].set_xticklabels([f"M{i+1}" for i in x], rotation=45, ha="right")
        axes[3].set_ylim(0, 1.05)
        axes[3].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        axes[3].legend()
        axes[3].set_title("Normalized detector amplitudes")

        fig.suptitle(f"Sample {sample_idx + 1} ({tag})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_path = output_dir / f"{tag}_sample{sample_idx:03d}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def compute_amp_relative_error_with_shift(
    model: torch.nn.Module,
    loader,
    *,
    shift_y_px: int,
    shift_x_px: int,
    evaluation_regions,
    pred_case: int,
    num_modes: int,
    eval_amplitudes: np.ndarray,
    eval_amplitudes_phases: np.ndarray,
    eval_phases: np.ndarray,
    phase_option: int,
    mmf_modes: torch.Tensor,
    field_size: int,
    image_test_data: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Evaluate amplitude-related metrics when the input field is shifted by (shift_y_px, shift_x_px).
    """
    model.eval()
    all_weights_pred: list[np.ndarray] = []
    all_weights_raw:  list[np.ndarray] = []   
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            shifted_images = shift_complex_batch(images, shift_y_px, shift_x_px)
            preds = model(shifted_images)
            preds_np = preds.detach().cpu().numpy()

            # NOTE: this assumes preds are (B,1,H,W). If you use MultiWL here, adapt similarly.
            for sample_idx in range(preds_np.shape[0]):
                intensity_map = preds_np[sample_idx, 0]
                weights = []
                for (x0, x1, y0, y1) in evaluation_regions:
                    cx = int(round((x0 + x1) / 2.0))
                    cy = int(round((y0 + y1) / 2.0))
                    radius_px = max(1, int(round((x1 - x0) / 2.0)))
                    yy, xx = np.ogrid[:intensity_map.shape[0], :intensity_map.shape[1]]
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
                    weights.append(float(intensity_map[mask].sum()))
                weights = np.asarray(weights, dtype=np.float64)

                norm_val = float(weights.sum())
                if norm_val > 0:
                    weights = weights / norm_val

                all_weights_pred.append(weights)

    metrics = compute_model_prediction_metrics(
        all_weights_pred,
        eval_amplitudes,
        eval_amplitudes_phases,
        eval_phases,
        phase_option,
        pred_case,
        num_modes,
        mmf_modes,
        field_size,
        image_test_data,
    )
    return metrics


# ==========================================================
# forward_full_intensity: original single-wavelength ODNN
# ==========================================================
def forward_full_intensity(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate inputs through the ODNN with padding retained to produce both cropped and
    full-field intensities.
    Assumes a single-wavelength ODNN-style model (layers have 2D phase, propagation uses 2D).
    """
    if not inputs.is_complex():
        raise ValueError("inputs must be complex64/complex128 tensors.")

    device = inputs.device
    batch, _, height, width = inputs.shape
    pad = int(model.propagation.pad_px)

    padded = complex_pad(inputs.squeeze(1), pad, pad)

    if hasattr(model, "pre_propagation") and model.pre_propagation is not None:
        pre = model.pre_propagation
        padded = pre._propagate(padded, pre.kz_pad, pre.z)

    for layer in model.layers:
        phase = torch.exp(1j * layer.phase.to(device, dtype=torch.float32)).to(torch.complex64)
        phase_big = torch.ones(height + 2 * pad, width + 2 * pad, dtype=torch.complex64, device=device)
        phase_big[pad:pad + height, pad:pad + width] = phase
        padded = padded * phase_big
        padded = layer._propagate(padded, layer.kz_pad, layer.z)

    prop = model.propagation
    padded = prop._propagate(padded, prop.kz_pad, prop.z)

    intensity_full = torch.abs(padded) ** 2
    cropped = complex_crop(padded, height, width, pad, pad)
    intensity_cropped = torch.abs(cropped) ** 2

    return intensity_cropped, intensity_full


# ==========================================================
# forward_full_intensity_multiwl: NEW for multi-wavelength
# ==========================================================
def forward_full_intensity_multiwl(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    wavelength_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-wavelength compatible forward.

    inputs: (B,1,H,W) complex
    model:  expects forward((B,L,H,W) complex) -> (B,L,H,W) float intensity

    returns:
      - intensity_cropped: (B,1,H,W) float
      - intensity_full:    (B,1,H+2p,W+2p) float
        (constructed by placing cropped intensity into center; outside=0)
    """
    if not inputs.is_complex():
        raise ValueError("inputs must be complex64/complex128 tensors.")

    B, C, H, W = inputs.shape
    if C != 1:
        raise ValueError(f"Expected inputs (B,1,H,W), got {tuple(inputs.shape)}")

    pad = int(model.propagation.pad_px)

    # infer L
    if hasattr(model, "propagation") and hasattr(model.propagation, "wavelengths"):
        L = int(model.propagation.wavelengths.numel())
    elif hasattr(model, "wavelengths"):
        L = int(model.wavelengths.numel())
    else:
        raise AttributeError("Cannot infer number of wavelengths L from model.")

    wl_idx = int(wavelength_idx)
    if not (0 <= wl_idx < L):
        raise ValueError(f"wavelength_idx={wl_idx} out of range for L={L}")

    x = inputs.repeat(1, L, 1, 1).contiguous()  # (B,L,H,W)

    with torch.no_grad():
        I_all = model(x)  # (B,L,H,W)

    if I_all.ndim != 4:
        raise ValueError(f"MultiWL model output must be (B,L,H,W), got {tuple(I_all.shape)}")

    I_crop = I_all[:, wl_idx:wl_idx + 1, :, :].contiguous()  # (B,1,H,W)

    I_full = torch.zeros((B, 1, H + 2 * pad, W + 2 * pad), dtype=I_crop.dtype, device=I_crop.device)
    I_full[:, :, pad:pad + H, pad:pad + W] = I_crop

    return I_crop, I_full


def build_roi_masks_with_centers(
    shape: Tuple[int, int],
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    center_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray, list[Tuple[int, int]]]:
    height, width = int(shape[-2]), int(shape[-1])
    yy, xx = np.ogrid[:height, :width]
    off_x, off_y = center_offset

    masks = []
    centers = []
    for x0, x1, y0, y1 in evaluation_regions:
        cx = int(round((x0 + x1) / 2.0 + off_x))
        cy = int(round((y0 + y1) / 2.0 + off_y))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        masks.append(mask)
        centers.append((cx, cy))
    union = np.any(np.stack(masks, axis=0), axis=0)
    return np.stack(masks, axis=0), union, centers


def spot_energy_and_snr(
    intensity: np.ndarray,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    spot_radius: int,
    *,
    ring_inner_pad: int = 3,
    ring_thickness: int = 8,
    union_mode: str = "global",
    center_offset: Tuple[int, int] = (0, 0),
) -> dict:
    if np.iscomplexobj(intensity):
        intensity = np.abs(intensity) ** 2
    intensity = np.asarray(intensity, dtype=np.float64)

    eps = 1e-12
    height, width = intensity.shape
    yy, xx = np.ogrid[:height, :width]

    sig_masks, union_mask, centers = build_roi_masks_with_centers(
        (height, width), evaluation_regions, spot_radius, center_offset
    )

    total_energy = float(intensity.sum())
    signal_energy = float(intensity[union_mask].sum())
    ratio_union = signal_energy / (total_energy + eps)

    energies = []
    snr_each_db = []
    for (cx, cy), sig_mask in zip(centers, sig_masks):
        rsq = (xx - cx) ** 2 + (yy - cy) ** 2
        r_in2 = (spot_radius + ring_inner_pad) ** 2
        r_out2 = (spot_radius + ring_inner_pad + ring_thickness) ** 2
        ring_mask = (rsq >= r_in2) & (rsq <= r_out2) & (~union_mask)

        energy = float(intensity[sig_mask].sum())
        energies.append(energy)

        if np.any(ring_mask):
            bg_mean = float(intensity[ring_mask].mean())
            bg_est = bg_mean * int(sig_mask.sum())
            snr_linear = energy / (bg_est + eps)
            snr_db = 10.0 * np.log10(max(snr_linear, eps))
        else:
            snr_db = float("nan")
        snr_each_db.append(snr_db)

    if union_mode == "global":
        bg_union = max(total_energy - signal_energy, 0.0)
    elif union_mode == "ring":
        ring_union = np.zeros_like(union_mask, dtype=bool)
        for (cx, cy) in centers:
            rsq = (xx - cx) ** 2 + (yy - cy) ** 2
            r_in2 = (spot_radius + ring_inner_pad) ** 2
            r_out2 = (spot_radius + ring_inner_pad + ring_thickness) ** 2
            ring_union |= (rsq >= r_in2) & (rsq <= r_out2)
        ring_union &= ~union_mask
        if np.any(ring_union):
            bg_mean_union = float(intensity[ring_union].mean())
            bg_union = bg_mean_union * int(union_mask.sum())
        else:
            bg_union = float("nan")
    else:
        raise ValueError("union_mode must be 'global' or 'ring'.")

    snr_union_db = (
        10.0 * np.log10(max(signal_energy / (bg_union + eps), eps)) if not np.isnan(bg_union) else float("nan")
    )

    return {
        "energies": np.array(energies, dtype=np.float64),
        "snr_each_db": np.array(snr_each_db, dtype=np.float64),
        "ratio_union": ratio_union,
        "snr_union_db": snr_union_db,
    }


def compute_model_prediction_metrics(
    weights_pred: Sequence[np.ndarray],
    amplitudes: np.ndarray,
    amplitudes_phases: np.ndarray,
    phases: np.ndarray,
    phase_option: int,
    pred_case: int,
    num_modes: int,
    mmf_modes: torch.Tensor,
    field_size: int,
    image_test_data: torch.Tensor,
) -> Dict[str, np.ndarray | float]:
    """
    Aggregate amplitude/phase metrics for a single trained model.
    """
    weights_array = np.asarray(weights_pred, dtype=np.float64)
    if weights_array.ndim == 1:
        weights_array = weights_array[None, :]

    num_samples = weights_array.shape[0]
    if num_samples == 0:
        return {
            "normalized_weights": weights_array,
            "complex_weights_pred": np.array([], dtype=np.complex64),
            "amplitudes_diff": np.array([], dtype=np.float64),
            "avg_amplitudes_diff": 0.0,
            "avg_relative_amp_err": 0.0,
            "image_data_pred": np.array([], dtype=np.complex64),
            "cc_recon_amp": np.array([], dtype=np.float64),
            "cc_recon_phase": np.array([], dtype=np.float64),
            "cc_real": np.array([], dtype=np.float64),
            "cc_imag": np.array([], dtype=np.float64),
        }

    if phase_option == 4:
        target_amp = amplitudes[:num_samples, :num_modes]
        phase_reference = phases[:num_samples, :num_modes]
    else:
        target_amp = amplitudes_phases[:num_samples, :num_modes]
        phase_reference = phases[:num_samples, :num_modes]

    target_energy = np.square(target_amp, dtype=np.float64)
    target_energy = target_energy / (np.sum(target_energy, axis=1, keepdims=True) + 1e-12)

    pred_energy = weights_array
    pred_energy = pred_energy / (np.sum(pred_energy, axis=1, keepdims=True) + 1e-12)
    amp_pred = np.sqrt(np.clip(pred_energy, 0.0, None))

    amp_target = l2_normalize_rows(target_amp.astype(np.float64))
    normalized_weights = amp_pred

    amplitudes_diff = np.abs(amp_target - amp_pred)
    avg_amp_err = float(np.mean(amplitudes_diff))
    relative_scale = 1.0 / np.sqrt(num_modes)
    avg_relative_amp_err = float(np.mean(amplitudes_diff / (relative_scale + 1e-12)))

    complex_weights_pred = (amp_pred * np.exp(1j * phase_reference)).astype(np.complex64, copy=False)

    complex_weights_tensor = torch.from_numpy(complex_weights_pred)
    reconstructed = generate_fields_ts(
        complex_weights_tensor,
        mmf_modes,
        num_samples,
        num_modes,
        field_size,
    ).cpu().numpy().squeeze()

    image_test_np = image_test_data.cpu().numpy().squeeze()

    cc_amp_list, cc_phase_list = [], []
    limit = min(image_test_np.shape[0], reconstructed.shape[0])
    for idx in range(limit):
        reference_flat = image_test_np[idx].ravel()
        predicted_flat = reconstructed[idx].ravel()
        cc_amp_list.append(np.corrcoef(np.abs(reference_flat), np.abs(predicted_flat))[0, 1])
        cc_phase_list.append(np.corrcoef(np.angle(reference_flat), np.angle(predicted_flat))[0, 1])

    cc_real = []
    cc_imag = []
    if phase_option == 4:
        ref_weights = amp_target * np.exp(1j * phases[:num_samples, :num_modes])
        for idx in range(min(ref_weights.shape[0], complex_weights_pred.shape[0])):
            ref_vec = ref_weights[idx]
            pred_vec = complex_weights_pred[idx]
            cc_real.append(np.corrcoef(np.real(ref_vec), np.real(pred_vec))[0, 1])
            cc_imag.append(np.corrcoef(np.imag(ref_vec), np.imag(pred_vec))[0, 1])

    return {
        "normalized_weights": normalized_weights,
        "complex_weights_pred": complex_weights_pred,
        "amplitudes_diff": amplitudes_diff,
        "avg_amplitudes_diff": avg_amp_err,
        "avg_relative_amp_err": avg_relative_amp_err,
        "image_data_pred": reconstructed,
        "cc_recon_amp": np.array(cc_amp_list, dtype=np.float64),
        "cc_recon_phase": np.array(cc_phase_list, dtype=np.float64),
        "cc_real": np.array(cc_real, dtype=np.float64),
        "cc_imag": np.array(cc_imag, dtype=np.float64),
    }


def evaluate_spot_metrics(
    model: torch.nn.Module,
    test_loader,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    *,
    detect_radius: int,
    device: torch.device,
    pred_case: int,
    num_modes: int,
    phase_option: int,
    amplitudes: np.ndarray,
    amplitudes_phases: np.ndarray,
    phases: np.ndarray,
    mmf_modes: torch.Tensor,
    field_size: int,
    image_test_data: torch.Tensor,
    snr_ring_inner_pad: int = 3,
    snr_ring_thickness: int = 8,
    snr_union_mode: str = "ring",     # "ring" 推荐；或 "global"
) -> Dict[str, np.ndarray | float]:
    """
    SNR is now computed via spot_energy_and_snr (ring-background estimator),
    on the cropped intensity Ic of each sample.
    """
    model.eval()
    eps = 1e-12
    r_sig = max(1, int(round(detect_radius / 2.0)))

    # ---- bookkeeping ----
    throughput_list: list[float] = []
    snr_ratio_full_list: list[float] = []   # 仍然保留旧的占比，用作对照
    snr_ratio_crop_list: list[float] = []

    snr_union_db_list: list[float] = []     # NEW: 来自 spot_energy_and_snr
    snr_each_db_batch: list[np.ndarray] = []  # NEW: 每个 ROI 的环形 SNR

    all_weights_pred: list[np.ndarray] = []
    all_weights_raw:  list[np.ndarray] = []

    for images, _ in test_loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)

        if hasattr(model, "propagation") and hasattr(model.propagation, "wavelengths"):
            I_crop_t, I_big_t = forward_full_intensity_multiwl(model, images, wavelength_idx=0)
        else:
            I_crop_t, I_big_t = forward_full_intensity(model, images)

        I_crop = I_crop_t.detach().cpu().numpy()
        I_big  = I_big_t.detach().cpu().numpy()

        pad = int(model.propagation.pad_px)

        if I_crop.ndim == 4 and I_crop.shape[1] == 1:
            I_crop = I_crop[:, 0, :, :]
        if I_big.ndim == 4 and I_big.shape[1] == 1:
            I_big = I_big[:, 0, :, :]

        for b in range(I_crop.shape[0]):
            Ic = I_crop[b]   # (H, W)
            Ib = I_big[b]    # (Hbig, Wbig)

            total_full = float(Ib.sum())
            total_crop = float(Ic.sum())

            # 旧的占比（保留）
            signal_full = sum_signal_energy_circle(Ib, evaluation_regions, r_sig, offset=(pad, pad))
            signal_crop = sum_signal_energy_circle(Ic, evaluation_regions, r_sig, offset=(0, 0))
            snr_ratio_full_list.append(signal_full / (total_full + eps))
            snr_ratio_crop_list.append(signal_crop / (total_crop + eps))
            throughput_list.append(total_crop / (total_full + eps))

            # ===== NEW: 真正的环形背景 SNR =====
            snr_info = spot_energy_and_snr(
                Ic,
                evaluation_regions,
                spot_radius=r_sig,
                ring_inner_pad=snr_ring_inner_pad,
                ring_thickness=snr_ring_thickness,
                union_mode=snr_union_mode,
                center_offset=(0, 0),
            )
            snr_union_db_list.append(float(snr_info["snr_union_db"]))
            snr_each_db_batch.append(np.asarray(snr_info["snr_each_db"], dtype=np.float64))

            # 检测器原始能量（用于 isolation / weights）
            masks_crop = _circle_masks_from_regions(Ic.shape, evaluation_regions, r_sig, offset=(0, 0))
            weights_raw = np.asarray([float(Ic[m].sum()) for m in masks_crop], dtype=np.float64)
            all_weights_raw.append(weights_raw)

            norm_val = float(weights_raw.sum())
            weights = weights_raw / norm_val if norm_val > 0 else weights_raw
            all_weights_pred.append(weights)

    metrics = compute_model_prediction_metrics(
        all_weights_pred,
        amplitudes,
        amplitudes_phases,
        phases,
        phase_option,
        pred_case,
        num_modes,
        mmf_modes,
        field_size,
        image_test_data,
    )

    # ---- 聚合 ----
    ratio_full      = float(np.mean(snr_ratio_full_list)) if snr_ratio_full_list else float("nan")
    ratio_crop      = float(np.mean(snr_ratio_crop_list)) if snr_ratio_crop_list else float("nan")
    throughput_mean = float(np.mean(throughput_list))      if throughput_list      else float("nan")

    # NEW: 用环形背景 SNR 替换原来的 dB 报表
    if snr_union_db_list:
        snr_db_full = float(np.nanmean(snr_union_db_list))
    else:
        snr_db_full = float("nan")

    if snr_each_db_batch:
        snr_each_db_mean_vec = np.nanmean(np.vstack(snr_each_db_batch), axis=0)
    else:
        snr_each_db_mean_vec = np.array([], dtype=np.float64)

    # ============================================================
    # Isolation & crosstalk matrix (unchanged)
    # ============================================================
    W = np.vstack(all_weights_raw) if all_weights_raw else np.zeros((0, num_modes))
    isolation_db_per_sample    = np.array([], dtype=np.float64)
    isolation_db_wc_per_sample = np.array([], dtype=np.float64)
    crosstalk_matrix           = np.zeros((num_modes, num_modes), dtype=np.float64)

    if W.shape[0] > 0 and W.shape[1] == num_modes:
        target_amp_full = (amplitudes if phase_option == 4 else amplitudes_phases)[: W.shape[0], :num_modes]
        target_idx = np.argmax(target_amp_full, axis=1)

        Et          = W[np.arange(W.shape[0]), target_idx]
        Eothers_sum = W.sum(axis=1) - Et
        mask_other  = np.ones_like(W, dtype=bool)
        mask_other[np.arange(W.shape[0]), target_idx] = False
        Emax_other  = (W * mask_other).max(axis=1)

        isolation_db_per_sample    = 10.0 * np.log10((Et + eps) / (Eothers_sum + eps))
        isolation_db_wc_per_sample = 10.0 * np.log10((Et + eps) / (Emax_other  + eps))

        W_row_norm = W / (W.sum(axis=1, keepdims=True) + eps)
        for m in range(num_modes):
            sel = (target_idx == m)
            if sel.any():
                crosstalk_matrix[m] = W_row_norm[sel].mean(axis=0)

    def _ratio_to_db(r):
        r = float(np.clip(r, eps, 1.0 - eps))
        return 10.0 * np.log10(r / (1.0 - r))

    snr_db_containment_full = _ratio_to_db(ratio_full) if np.isfinite(ratio_full) else float("nan")
    snr_db_containment_crop = _ratio_to_db(ratio_crop) if np.isfinite(ratio_crop) else float("nan")

    metrics.update(
        {
            # 兼容字段（继续输出）
            "snr_ratio_full": ratio_full,
            "snr_ratio_crop": ratio_crop,
            "throughput":     throughput_mean,

            # ===== 主报表：环形背景 SNR =====
            "snr_db_ring_union": snr_db_full,           # ring-bg union dB（每样本平均）
            "snr_each_db":       snr_each_db_mean_vec,  # 每个 ROI 的 ring-bg SNR
            "snr_union_db_per_sample": np.asarray(snr_union_db_list, dtype=np.float64),
            "snr_each_db_per_sample":  np.vstack(snr_each_db_batch) if snr_each_db_batch
                                       else np.zeros((0, len(evaluation_regions))),

            # ===== 旧 containment 也保留 =====
            "snr_db_containment_full": snr_db_containment_full,    # NEW: r=full(含 pad) 的 containment dB
            "snr_db_containment_crop": snr_db_containment_crop,    # NEW: r=crop 的 containment dB

            # 兼容旧字段名：让历史代码读 snr_db_full 还能拿到 ring-bg
            "snr_db_full": snr_db_full,

            "isolation_db":         isolation_db_per_sample,
            "isolation_db_worst":   isolation_db_wc_per_sample,
            "isolation_db_mean":    float(np.nanmean(isolation_db_per_sample))    if isolation_db_per_sample.size    else float("nan"),
            "isolation_db_wc_mean": float(np.nanmean(isolation_db_wc_per_sample)) if isolation_db_wc_per_sample.size else float("nan"),
            "crosstalk_matrix":     crosstalk_matrix,
        }
    )

    return metrics


def format_metric_report(
    num_modes: int,
    phase_option: int,
    pred_case: int,
    label: str,
    metrics: Dict[str, np.ndarray | float],
) -> str:
    """
    Build a concise, multi-line summary string for the provided metrics dict.
    """
    amp_err = float(metrics.get("avg_amplitudes_diff", float("nan")))
    rel_amp_err = float(metrics.get("avg_relative_amp_err", float("nan")))

    def _summary(name: str, values: np.ndarray | float) -> str:
        if values is None:
            return f"{name}=n/a"
        if isinstance(values, (float, int)):
            return f"{name}={float(values):.6f}"
        arr = np.asarray(values)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return f"{name}=n/a"
        return f"{name}={np.nanmean(arr):.6f}±{np.nanstd(arr):.6f}"

    sections = [
        f"{label}: modes={num_modes}, phase_opt={phase_option}, pred_case={pred_case}",
        f"  amp_err={amp_err:.6f}, amp_err_rel={rel_amp_err:.6f}",
    ]

    if "snr_ratio_full" in metrics:
        sections.append(
            "  "
            + ", ".join(
                [
                    _summary("snr_full", metrics.get("snr_ratio_full")),
                    _summary("snr_crop", metrics.get("snr_ratio_crop")),
                    _summary("throughput", metrics.get("throughput")),
                ]
            )
        )

    cc_fields = [
        ("cc_amp", metrics.get("cc_recon_amp")),
        ("cc_phase", metrics.get("cc_recon_phase")),
        ("cc_real", metrics.get("cc_real")),
        ("cc_imag", metrics.get("cc_imag")),
    ]

    snr_db_full = metrics.get("snr_db_full")
    snr_each_db = metrics.get("snr_each_db")
    if snr_db_full is not None and not (isinstance(snr_db_full, float) and np.isnan(snr_db_full)):
        snr_line_parts = [f"snr_union_db={float(snr_db_full):.3f}"]
        if isinstance(snr_each_db, np.ndarray) and snr_each_db.size:
            snr_line_parts.append(
                "snr_each_db=[" + ", ".join(f"{v:.2f}" for v in snr_each_db) + "]"
            )
        sections.append("  " + ", ".join(snr_line_parts))
    
    cc_line = ", ".join(_summary(name, values) for name, values in cc_fields if values is not None)
    if cc_line:
        sections.append("  " + cc_line)

    return "\n".join(sections)


# ==========================================================
# mask shift helpers (keep as your original)
# ==========================================================
def _shift_phase_bilinear(phase_2d: torch.Tensor, dx_mm: float, dy_mm: float, pixel_size_m: float) -> torch.Tensor:
    """
    把 2D 相位（弧度）在 x/y 方向平移 (dx_mm, dy_mm)，双线性插值。
    +x 向右，+y 向下；超界补相位 0（= 透明）。
    """
    with torch.no_grad():
        device = phase_2d.device
        H, W = phase_2d.shape
        dx_pix = (dx_mm * 1e-3) / pixel_size_m
        dy_pix = (dy_mm * 1e-3) / pixel_size_m
        sx = 2.0 * dx_pix / (W - 1)
        sy = 2.0 * dy_pix / (H - 1)
        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx - sx, gy - sy], dim=-1)[None]  # (1,H,W,2)
        out = F.grid_sample(
            phase_2d[None, None], grid, mode="bilinear",
            padding_mode="zeros", align_corners=True
        )
        return out[0, 0]


def apply_shift_to_model_masks(D2NN, dx_mm: float, dy_mm: float, pixel_size_m: float):
    """
    把模型中每一层的 phase 替换为“平移后的相位”，返回原相位列表，便于之后 restore。
    """
    originals = []
    for layer in D2NN.layers:
        p_orig = layer.phase.data.clone()
        originals.append(p_orig)

        p_shift = _shift_phase_bilinear(p_orig, dx_mm, dy_mm, pixel_size_m)
        layer.phase.data.copy_(p_shift)

    return originals


def restore_model_masks(D2NN, originals):
    for layer, p in zip(D2NN.layers, originals):
        layer.phase.data.copy_(p)

def evaluate_spot_metrics_multiwl_each(
    model: torch.nn.Module,
    test_loader,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    *,
    detect_radius: int,
    device: torch.device,
    pred_case: int,
    num_modes: int,
    phase_option: int,
    amplitudes: np.ndarray,
    amplitudes_phases: np.ndarray,
    phases: np.ndarray,
    mmf_modes: torch.Tensor,
    field_size: int,
    image_test_data: torch.Tensor,
) -> Dict[int, Dict[str, np.ndarray | float]]:
    """
    Return a metrics dict per wavelength index:
      {wl_idx: metrics_dict}

    Requires a MultiWL model with model.propagation.wavelengths (or model.wavelengths).
    """
    # infer wavelengths list
    if hasattr(model, "propagation") and hasattr(model.propagation, "wavelengths"):
        wls = model.propagation.wavelengths.detach().cpu().numpy().astype(np.float64)
    elif hasattr(model, "wavelengths"):
        wls = model.wavelengths.detach().cpu().numpy().astype(np.float64)
    else:
        raise AttributeError("MultiWL model must expose wavelengths via model.propagation.wavelengths or model.wavelengths")

    L = int(len(wls))
    results: Dict[int, Dict[str, np.ndarray | float]] = {}

    # run per wavelength by temporarily calling forward_full_intensity_multiwl with different idx
    # easiest: reuse evaluate_spot_metrics logic, but force wavelength_idx each time.
    # We do it by monkeypatching a tiny wrapper that looks MultiWL but always returns chosen wl.
    class _PickWL(torch.nn.Module):
        def __init__(self, m: torch.nn.Module, wl_idx: int):
            super().__init__()
            self.m = m
            self.wl_idx = int(wl_idx)
            # keep propagation attr so evaluate_spot_metrics branches to multiwl
            self.propagation = getattr(m, "propagation", None)

        def forward(self, x_blhw: torch.Tensor) -> torch.Tensor:
            # x_blhw is (B,L,H,W) complex in multiwl eval path
            I_all = self.m(x_blhw)  # (B,L,H,W) float
            return I_all[:, self.wl_idx : self.wl_idx + 1].contiguous()  # (B,1,H,W)

    # NOTE:
    # - evaluate_spot_metrics() will detect MultiWL by propagation.wavelengths and call forward_full_intensity_multiwl(),
    #   which repeats inputs to (B,L,H,W) and calls model(x).
    # - So here we wrap the model so that model(x) returns (B,1,H,W) for the chosen wl,
    #   and we also make the wrapper still look "multiwl" via .propagation.wavelengths existing.
    for wl_idx in range(L):
        picker = _PickWL(model, wl_idx).to(device)
        picker.eval()

        m = evaluate_spot_metrics(
            picker,
            test_loader,
            evaluation_regions,
            detect_radius=detect_radius,
            device=device,
            pred_case=pred_case,
            num_modes=num_modes,
            phase_option=phase_option,
            amplitudes=amplitudes,
            amplitudes_phases=amplitudes_phases,
            phases=phases,
            mmf_modes=mmf_modes,
            field_size=field_size,
            image_test_data=image_test_data,
        )
        # attach wavelength value for convenience
        m = dict(m)
        m["wavelength_m"] = float(wls[wl_idx])
        results[int(wl_idx)] = m

    return results

def save_prediction_diagnostics_multiwl_each(
    model: torch.nn.Module,
    dataset,
    *,
    evaluation_regions,
    layer_size: int,
    detect_radius: int,
    num_samples: int,
    output_dir: Path,
    device: torch.device,
    tag: str,
    wavelength_indices: Optional[Sequence[int]] = None,
) -> Dict[int, list[Path]]:
    """
    Save per-wavelength diagnostics for a MultiWL model.

    It will save, for each wavelength index:
      - label map
      - predicted intensity map (for that wavelength)
      - abs diff
      - normalized detector energies (label vs pred)

    Returns:
      {wl_idx: [list of saved png paths]}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # infer wavelengths and L
    if hasattr(model, "propagation") and hasattr(model.propagation, "wavelengths"):
        wls = model.propagation.wavelengths.detach().cpu().numpy().astype(np.float64)
    elif hasattr(model, "wavelengths"):
        wls = model.wavelengths.detach().cpu().numpy().astype(np.float64)
    else:
        raise AttributeError("MultiWL model must expose wavelengths via model.propagation.wavelengths or model.wavelengths")

    L = int(len(wls))
    if wavelength_indices is None:
        wl_indices = list(range(L))
    else:
        wl_indices = [int(i) for i in wavelength_indices]
        for i in wl_indices:
            if i < 0 or i >= L:
                raise ValueError(f"wavelength index {i} out of range for L={L}")

    sample_indices = list(range(min(num_samples, len(dataset))))

    def _get_sample(dataset_, idx_: int):
        # same behavior as your helper
        if isinstance(dataset_, list):
            return dataset_[idx_]
        if isinstance(dataset_, TensorDataset):
            return dataset_.tensors[0][idx_], dataset_.tensors[1][idx_]
        raise TypeError(f"Unsupported dataset type: {type(dataset_)}")

    def _region_energy(arr2d: np.ndarray) -> np.ndarray:
        yy, xx = np.ogrid[:arr2d.shape[0], :arr2d.shape[1]]
        radius_px = max(1, int(round(detect_radius / 2.0)))
        vals = []
        for (x0, x1, y0, y1) in evaluation_regions:
            cx = int(round((x0 + x1) / 2.0))
            cy = int(round((y0 + y1) / 2.0))
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
            vals.append(float(arr2d[mask].sum()))
        return np.asarray(vals, dtype=np.float64)

    def _safe_norm(v: np.ndarray) -> np.ndarray:
        s = float(v.sum())
        return v / s if s > 0 else v

    saved_paths: Dict[int, list[Path]] = {i: [] for i in wl_indices}

    for sample_idx in sample_indices:
        image, label = _get_sample(dataset, sample_idx)

        # image: (1,H,W) complex, label: (1,H,W) float
        image_b1 = image.unsqueeze(0).to(device, dtype=torch.complex64)  # (B=1,1,H,W)
        label_map = label[0].detach().cpu().numpy()

        # forward multiwl: (1,L,H,W)
        x_blhw = image_b1.repeat(1, L, 1, 1).contiguous()
        with torch.no_grad():
            I_blhw = model(x_blhw)  # (1,L,H,W) float

        I_lhw = I_blhw[0].detach().cpu().numpy()  # (L,H,W)

        # prepare label detector bars once
        label_weights = _region_energy(label_map)
        label_norm = _safe_norm(label_weights)

        for wl_idx in wl_indices:
            pred_map = I_lhw[wl_idx]
            pred_weights = _region_energy(pred_map)
            pred_norm = _safe_norm(pred_weights)

            wl_nm = wls[wl_idx] * 1e9

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            im0 = axes[0].imshow(label_map, cmap="inferno")
            axes[0].set_title("Label")
            axes[0].set_axis_off()
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(pred_map, cmap="inferno")
            axes[1].set_title(f"Prediction (wl_idx={wl_idx}, {wl_nm:.1f} nm)")
            axes[1].set_axis_off()
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            diff_map = np.abs(pred_map - label_map)
            im2 = axes[2].imshow(diff_map, cmap="magma")
            axes[2].set_title("|Pred - Label|")
            axes[2].set_axis_off()
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            radius_px = max(1, int(round(detect_radius / 2.0)))
            for ax in (axes[0], axes[1], axes[2]):
                for (x0, x1, y0, y1) in evaluation_regions:
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    circ = Circle(
                        (cx, cy),
                        radius=radius_px,
                        linewidth=0.8,
                        edgecolor="cyan",
                        facecolor="none",
                        linestyle=":",
                        alpha=0.9,
                    )
                    ax.add_patch(circ)

            x = np.arange(len(pred_norm))
            width = 0.35
            axes[3].bar(x - width / 2, label_norm, width=width, label="Label", color="tab:blue")
            axes[3].bar(x + width / 2, pred_norm, width=width, label="Pred", color="tab:orange")
            axes[3].set_xticks(x)
            axes[3].set_xticklabels([f"M{i+1}" for i in x], rotation=45, ha="right")
            axes[3].set_ylim(0, 1.05)
            axes[3].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            axes[3].legend()
            axes[3].set_title("Normalized detector energies")

            fig.suptitle(f"Sample {sample_idx:03d} | {tag} | wl_idx={wl_idx} ({wl_nm:.1f} nm)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))

            out_path = output_dir / f"{tag}_wl{wl_idx:02d}_sample{sample_idx:03d}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            saved_paths[int(wl_idx)].append(out_path)

    return saved_paths
