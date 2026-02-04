from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from odnn_model import D2NNModel
from odnn_training_eval import evaluate_spot_metrics, spot_energy_ratios_circle


@dataclass(frozen=True)
class ModelGeometry:
    layer_size: int
    z_layers: float
    z_prop: float
    pixel_size: float
    z_input_to_first: float
    padding_ratio: float = 0.5


def scale_phase_masks_for_wavelength(
    phase_masks: Sequence[np.ndarray],
    base_wavelength: float,
    target_wavelength: float,
) -> list[np.ndarray]:
    """
    Rescale phase masks when the illumination wavelength changes.
    """
    if target_wavelength <= 0:
        raise ValueError("target_wavelength must be positive.")

    scale = float(base_wavelength / target_wavelength)
    scaled_masks = [np.mod(mask * scale, 2.0 * np.pi) for mask in phase_masks]
    return scaled_masks


def _instantiate_model(
    num_layers: int,
    wavelength: float,
    geometry: ModelGeometry,
    device: torch.device,
) -> D2NNModel:
    model = D2NNModel(
        num_layers=num_layers,
        layer_size=geometry.layer_size,
        z_layers=geometry.z_layers,
        z_prop=geometry.z_prop,
        pixel_size=geometry.pixel_size,
        wavelength=wavelength,
        device=device,
        padding_ratio=geometry.padding_ratio,
        z_input_to_first=geometry.z_input_to_first,
    ).to(device)
    return model


@torch.no_grad()
def _assign_phase_masks(model: D2NNModel, phase_masks: Sequence[np.ndarray], device: torch.device) -> None:
    for layer, mask in zip(model.layers, phase_masks):
        phase_tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).to(device=device)
        layer.phase.data.copy_(phase_tensor)


def _infer_target_region(label_map: np.ndarray, evaluation_regions) -> int:
    scores = [
        float(label_map[y0:y1, x0:x1].mean())
        for (x0, x1, y0, y1) in evaluation_regions
    ]
    return int(np.argmax(scores))


def compute_relative_amp_error_wavelength_sweep(
    phase_masks: Sequence[np.ndarray],
    *,
    base_wavelength: float,
    wavelength_list: Iterable[float],
    geometry: ModelGeometry,
    device: torch.device,
    test_loader,
    evaluation_regions,
    detect_radius: int,
    pred_case: int,
    num_modes: int,
    phase_option: int,
    amplitudes: np.ndarray,
    amplitudes_phases: np.ndarray,
    phases: np.ndarray,
    mmf_modes: torch.Tensor,
    field_size: int,
    image_test_data: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray | float]]]:

    errors: list[float] = []
    metrics_list: list[dict[str, np.ndarray | float]] = []

    wavelength_array = np.asarray(list(wavelength_list), dtype=np.float64)
    for lam in wavelength_array:
        scaled_masks = scale_phase_masks_for_wavelength(phase_masks, base_wavelength, lam)
        model = _instantiate_model(len(phase_masks), lam, geometry, device)
        _assign_phase_masks(model, scaled_masks, device)
        model.eval()

        metrics = evaluate_spot_metrics(
            model,
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

        errors.append(float(metrics.get("avg_relative_amp_err", float("nan"))))
        metrics_list.append(metrics)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return wavelength_array, np.asarray(errors, dtype=np.float64), metrics_list


def compute_mode_isolation_wavelength_sweep(
    phase_masks: Sequence[np.ndarray],
    *,
    base_wavelength: float,
    wavelength_list: Iterable[float],
    geometry: ModelGeometry,
    device: torch.device,
    test_loader,
    evaluation_regions,
    detect_radius: int,
) -> dict[str, np.ndarray]:
    """
    Evaluate average mode isolation (ratio + dB) across wavelengths.
    """
    wavelength_array = np.asarray(list(wavelength_list), dtype=np.float64)
    
    isolation_ratio_fraction_rows: list[np.ndarray] = []
    detection_fraction_rows: list[np.ndarray] = []

    for lam in wavelength_array:
        scaled_masks = scale_phase_masks_for_wavelength(phase_masks, base_wavelength, lam)
        model = _instantiate_model(len(phase_masks), lam, geometry, device)
        _assign_phase_masks(model, scaled_masks, device)
        model.eval()

        ratios_this_lambda: list[float] = []
        detection_rows_this_lambda: list[np.ndarray] = []

        for images, labels in test_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            outputs = model(images)
            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for sample_idx in range(outputs_np.shape[0]):
                intensity = outputs_np[sample_idx, 0]
                label_map = labels_np[sample_idx, 0]
                energies, ratios = spot_energy_ratios_circle(
                    intensity,
                    evaluation_regions,
                    detect_radius,
                    offset=(0, 0),
                    eps=1e-12,
                )
                target_idx = _infer_target_region(label_map, evaluation_regions)
                detection_sum = float(np.sum(energies)) + 1e-12
                detection_fractions = energies / detection_sum
                detection_rows_this_lambda.append(detection_fractions.astype(np.float64))
                target_fraction = float(detection_fractions[target_idx])
                ratios_this_lambda.append(target_fraction)

        isolation_ratio_fraction_rows.append(np.asarray(ratios_this_lambda, dtype=np.float64))
        detection_fraction_rows.append(np.vstack(detection_rows_this_lambda))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not isolation_ratio_fraction_rows:
        raise RuntimeError("No isolation ratios computed; check dataset or loader.")

    sample_count = None
    for row in isolation_ratio_fraction_rows:
        if sample_count is None:
            sample_count = row.shape[0]
        elif row.shape[0] != sample_count:
            raise ValueError("Inconsistent number of samples across wavelength sweep.")

    isolation_ratio_fraction = np.vstack(isolation_ratio_fraction_rows)
    isolation_ratios_percent = isolation_ratio_fraction * 100.0
    detection_fraction = np.stack(detection_fraction_rows, axis=0)

    denom = np.clip(1.0 - isolation_ratio_fraction, 1e-12, None)
    ratio_vs_rest = isolation_ratio_fraction / denom
    isolation_db = 10.0 * np.log10(np.clip(ratio_vs_rest, 1e-12, None))

    isolation_ratio_mean_percent = isolation_ratios_percent.mean(axis=1)
    isolation_db_mean = isolation_db.mean(axis=1)

    return {
        "wavelengths": wavelength_array,
        "isolation_ratio_fraction": isolation_ratio_fraction,
        "isolation_ratios": isolation_ratios_percent,
        "isolation_db": isolation_db,
        "isolation_ratio_mean": isolation_ratio_mean_percent,
        "isolation_db_mean": isolation_db_mean,
        "detection_fractions": detection_fraction,
    }
