from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.io import savemat

from odnn_training_eval import sample_tensor_slices


def save_masks_one_file_per_layer(
    phase_layers: list[np.ndarray] | list[torch.Tensor],
    out_dir: str | Path,
    *,
    base_name: str = "mask",
    save_degree: bool = False,
    use_xlsx: bool = True,
) -> None:
    """
    Persist per-layer phase masks to disk, one file per layer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, mask in enumerate(phase_layers, start=1):
        array = np.asarray(mask, dtype=np.float32)
        if save_degree:
            array = np.degrees(array)

        file_name = f"{base_name}_layer{index}"
        if use_xlsx:
            df = pd.DataFrame(array)
            df.to_excel(out_dir / f"{file_name}.xlsx", index=False, header=False, engine="openpyxl")
        else:
            np.savetxt(out_dir / f"{file_name}.csv", array, delimiter=",")


def save_to_mat_light(
    filepath: str | Path,
    phase_stack: np.ndarray,
    input_field: torch.Tensor,
    propagated_fields: list[torch.Tensor],
    *,
    kmax: int = 16,
    save_amplitude_only: bool = True,
) -> None:
    """
    Lightweight MAT writer used for quick inspection of ODNN snapshots.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    masks = np.asarray(phase_stack, dtype=np.float32)
    input_field = input_field.detach().cpu()
    if save_amplitude_only:
        input_payload = torch.abs(input_field).to(torch.float32).numpy()
    else:
        input_payload = input_field.numpy()

    prop_slices = None
    if propagated_fields:
        tensor = propagated_fields[0].detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        sampled = sample_tensor_slices(tensor, kmax)
        if save_amplitude_only:
            sampled = torch.abs(sampled).to(torch.float32)
        prop_slices = sampled.numpy()

    mdict = {"temp_model": masks, "temp_E": input_payload}
    if prop_slices is not None:
        mdict["propagated_slices"] = prop_slices

    savemat(filepath, mdict, do_compression=True)
    print(f"Saved (v5 light): {filepath}")


def save_to_mat_light_plus(
    filepath: str | Path,
    *,
    phase_stack: np.ndarray,
    input_field: np.ndarray,
    scans: Dict[str, Dict[str, np.ndarray]],
    camera_field: np.ndarray | None = None,
    sample_stacks_kmax: int = 20,
    save_amplitude_only: bool = False,
    meta: dict | None = None,
) -> None:
    """
    Extended MAT writer that preserves complex stacks and optional metadata.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {"temp_model": np.asarray(phase_stack, dtype=np.float32)}

    input_field = np.asarray(input_field)
    if save_amplitude_only:
        payload["E0_amp"] = np.abs(input_field)
    else:
        payload["E0"] = input_field

    for name, pack in scans.items():
        stack = np.asarray(pack["stack"])
        z_values = np.asarray(pack["z"])

        if sample_stacks_kmax is not None and stack.ndim == 3 and stack.shape[0] > sample_stacks_kmax:
            indices = np.linspace(0, stack.shape[0] - 1, sample_stacks_kmax, dtype=int)
            stack = stack[indices]
            z_values = z_values[indices]

        if save_amplitude_only:
            payload[f"{name}_amp"] = np.abs(stack)
        else:
            payload[name] = stack
        payload[f"{name}_z"] = z_values

    if camera_field is not None:
        camera_array = np.asarray(camera_field)
        if save_amplitude_only:
            payload["E_camera_amp"] = np.abs(camera_array)
        else:
            payload["E_camera"] = camera_array

    if meta is not None:
        payload["meta_json"] = json.dumps(meta, ensure_ascii=False)

    savemat(filepath, payload, do_compression=True)
    print(f"Saved (v5 plus): {filepath}")

