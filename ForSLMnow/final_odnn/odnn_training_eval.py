from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def _circle_masks_from_regions(
    shape: Tuple[int, int],
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    offset: Tuple[int, int] = (0, 0),
) -> list[np.ndarray]:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    off_x, off_y = offset

    masks = []
    for x0, x1, y0, y1 in evaluation_regions:
        cx = int(round((x0 + x1) / 2.0 + off_x))
        cy = int(round((y0 + y1) / 2.0 + off_y))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        masks.append(mask)
    return masks


def sum_signal_energy_circle(
    intensity: np.ndarray,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    offset: Tuple[int, int] = (0, 0),
) -> float:
    masks = _circle_masks_from_regions(intensity.shape, evaluation_regions, radius, offset)
    return float(sum(intensity[mask].sum() for mask in masks))


def spot_energy_ratios_circle(
    intensity: np.ndarray,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    offset: Tuple[int, int] = (0, 0),
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    masks = _circle_masks_from_regions(intensity.shape, evaluation_regions, radius, offset)
    energies = np.array([float(intensity[mask].sum()) for mask in masks], dtype=np.float64)
    total = float(intensity.sum()) + eps
    ratios = energies / total
    return energies, ratios


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

    phases[:, 0] = 0.0  # enforce deterministic global phase reference
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
    amp_ts_energy = amp_ts ** 2  # 标签改为能量/强度
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
    label_map = (mmf_label_data * (amp_ts**2).view(1, 1, -1)).sum(dim=2)  # 标签用能量权重
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

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            shifted_images = shift_complex_batch(images, shift_y_px, shift_x_px)
            preds = model(shifted_images)
            preds_np = preds.detach().cpu().numpy()

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


def forward_full_intensity(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate inputs through the ODNN with padding retained to produce both cropped and
    full-field intensities.
    """
    if not inputs.is_complex():
        raise ValueError("inputs must be complex64/complex128 tensors.")

    device = inputs.device
    batch, _, height, width = inputs.shape
    pad = int(model.propagation.pad_px)

    # Pad to match propagation canvas
    padded = complex_pad(inputs.squeeze(1), pad, pad)

    # Optional pre-propagation
    if hasattr(model, "pre_propagation") and model.pre_propagation is not None:
        pre = model.pre_propagation
        padded = pre._propagate(padded, pre.kz_pad, pre.z)

    # Diffraction layers on padded canvas
    for layer in model.layers:
        phase = torch.exp(1j * layer.phase.to(device, dtype=torch.float32)).to(torch.complex64)
        phase_big = torch.ones(height + 2 * pad, width + 2 * pad, dtype=torch.complex64, device=device)
        phase_big[pad : pad + height, pad : pad + width] = phase
        padded = padded * phase_big
        padded = layer._propagate(padded, layer.kz_pad, layer.z)

    prop = model.propagation
    padded = prop._propagate(padded, prop.kz_pad, prop.z)

    intensity_full = torch.abs(padded) ** 2
    cropped = complex_crop(padded, height, width, pad, pad)
    intensity_cropped = torch.abs(cropped) ** 2

    return intensity_cropped, intensity_full


def build_roi_masks_with_centers(
    shape: Tuple[int, int],
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    radius: int,
    center_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, np.ndarray, list[Tuple[int, int]]]:
    height, width = shape
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

    # Target amplitudes
    if phase_option == 4:
        target_amp = amplitudes[:num_samples, :num_modes]
        phase_reference = phases[:num_samples, :num_modes]
    else:
        target_amp = amplitudes_phases[:num_samples, :num_modes]
        phase_reference = phases[:num_samples, :num_modes]

    # 能量占比 -> 幅度占比（预测端开方）保持与标签幅度比对
    target_energy = np.square(target_amp, dtype=np.float64)
    target_energy = target_energy / (np.sum(target_energy, axis=1, keepdims=True) + 1e-12) #双保险归一化一下
    # amp_target = np.sqrt(np.clip(target_energy, 0.0, None)) # 都行你用energy去算也行，你之前用amp也行
    pred_energy = weights_array
    pred_energy = pred_energy / (np.sum(pred_energy, axis=1, keepdims=True) + 1e-12)
    amp_pred = np.sqrt(np.clip(pred_energy, 0.0, None))  
    
    amp_target = l2_normalize_rows(target_amp.astype(np.float64)) #双保险一下归一化
    normalized_weights = amp_pred  # 保持接口兼容，返回预测幅度占比

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
) -> Dict[str, np.ndarray | float]:
    """
    Evaluate spot energy ratios and reconstruction metrics for a trained model.
    """
    model.eval()
    eps = 1e-12
    r_sig = max(1, int(round(detect_radius / 2.0)))

    snr_ratio_full_list: list[float] = []
    snr_ratio_crop_list: list[float] = []
    throughput_list: list[float] = []
    ratio_each_full_batch: list[np.ndarray] = []

    all_weights_pred: list[np.ndarray] = []

    for images, _ in test_loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)

        I_crop_t, I_big_t = forward_full_intensity(model, images)
        I_crop = I_crop_t.detach().cpu().numpy()
        I_big = I_big_t.detach().cpu().numpy()

        pad = int(model.propagation.pad_px)

        for b in range(I_crop.shape[0]):
            Ic = I_crop[b]
            Ib = I_big[b]

            total_full = float(Ib.sum())
            total_crop = float(Ic.sum())

            signal_full = sum_signal_energy_circle(Ib, evaluation_regions, r_sig, offset=(pad, pad))
            signal_crop = sum_signal_energy_circle(Ic, evaluation_regions, r_sig, offset=(0, 0))
            #每个区域占没有裁剪也就是加上padding的整个大图的能量比例，是比例
            _, ratios_full_vec = spot_energy_ratios_circle(Ib, evaluation_regions, r_sig, offset=(pad, pad), eps=eps)
            ratio_each_full_batch.append(ratios_full_vec)

            snr_ratio_full_list.append(signal_full / (total_full + eps))
            snr_ratio_crop_list.append(signal_crop / (total_crop + eps))
            throughput_list.append(total_crop / (total_full + eps))
            #裁剪后，且只考虑检测那几个区域的能量的能量值
            masks_crop = _circle_masks_from_regions(Ic.shape, evaluation_regions, r_sig, offset=(0, 0))
            weights = np.asarray([float(Ic[m].sum()) for m in masks_crop], dtype=np.float64)

            norm_val = float(weights.sum())
            if norm_val > 0:
                weights = weights / norm_val

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

    ratio_full = float(np.mean(snr_ratio_full_list)) if snr_ratio_full_list else float("nan")
    ratio_crop = float(np.mean(snr_ratio_crop_list)) if snr_ratio_crop_list else float("nan")
    throughput_mean = float(np.mean(throughput_list)) if throughput_list else float("nan")

    if ratio_each_full_batch:
        ratio_each_full_mean_vec = np.mean(np.vstack(ratio_each_full_batch), axis=0)
    else:
        ratio_each_full_mean_vec = np.array([], dtype=np.float64)

    def ratio_to_db(r):
        r = np.clip(r, eps, 1.0 - eps)
        return 10.0 * np.log10(r / (1.0 - r))

    snr_db_from_ratio_full = ratio_to_db(ratio_full) if np.isfinite(ratio_full) else float("nan")
    snr_each_db_mean_vec = ratio_to_db(ratio_each_full_mean_vec) if ratio_each_full_mean_vec.size else np.array([])

    metrics.update(
        {
            "snr_ratio_full": ratio_full,
            "snr_ratio_crop": ratio_crop,
            "throughput": throughput_mean,
            "snr_db_full": snr_db_from_ratio_full,
            "snr_each_db": snr_each_db_mean_vec,
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
    cc_line = ", ".join(_summary(name, values) for name, values in cc_fields if values is not None)
    if cc_line:
        sections.append("  " + cc_line)

    return "\n".join(sections)



# 用来给多波长但是一个mask用的函数
def _shift_phase_bilinear(phase_2d: torch.Tensor, dx_mm: float, dy_mm: float, pixel_size_m: float) -> torch.Tensor:
    """
    把 2D 相位（弧度）在 x/y 方向平移 (dx_mm, dy_mm)，双线性插值。
    +x 向右，+y 向下；超界补相位 0（= 透明）。
    """
    with torch.no_grad():
        device = phase_2d.device
        H, W   = phase_2d.shape
        dx_pix = (dx_mm * 1e-3) / pixel_size_m
        dy_pix = (dy_mm * 1e-3) / pixel_size_m
        sx = 2.0 * dx_pix / (W - 1)
        sy = 2.0 * dy_pix / (H - 1)
        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack([gx - sx, gy - sy], dim=-1)[None]  # (1,H,W,2)
        out  = F.grid_sample(phase_2d[None,None], grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)
        return out[0,0]

def apply_shift_to_model_masks(D2NN, dx_mm: float, dy_mm: float, pixel_size_m: float):
    """
    把模型中每一层的 phase 替换为“平移后的相位”，返回原相位列表，便于之后 restore。
    """
    originals = []
    for layer in D2NN.layers:
        # 取出原相位（leaf tensor），clone 一份保存
        p_orig = layer.phase.data.clone()
        originals.append(p_orig)

        # 生成平移后的相位
        p_shift = _shift_phase_bilinear(p_orig, dx_mm, dy_mm, pixel_size_m)
        layer.phase.data.copy_(p_shift)  

    return originals

def restore_model_masks(D2NN, originals):

    for layer, p in zip(D2NN.layers, originals):
        layer.phase.data.copy_(p)
