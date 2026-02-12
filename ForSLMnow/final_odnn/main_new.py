#%%
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.io import savemat
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from ODNN_functions import (
    create_evaluation_regions,
    generate_complex_weights,
    generate_fields_ts,
)
from odnn_generate_label import (
    compute_label_centers,
    compose_labels_from_patterns,
    generate_detector_patterns,
)
from odnn_io import load_complex_modes_from_mat
from odnn_processing import prepare_sample

# ✅ 你的模型文件里真实存在的类名
from odnn_multiwl_model import D2NNModelMultiWL

# ✅ ROI masks
from odnn_training_eval import build_circular_roi_masks

# ✅ 复用旧的 superposition 采样上下文（我们会把它的 label map 换成 y_vec）
from odnn_training_eval import build_superposition_eval_context

# ✅ NEW: 基于 odnn_training_visualization 扩展出来的 MultiWL 可视化
from odnn_training_visualization import (
    visualize_model_slices_multiwl,
    capture_eigenmode_propagation_multiwl,
)

# ✅ NEW: 用于 free_space_propagate
from odnn_model import complex_crop, complex_pad
from odnn_processing import pad_field_to_layer


# ----------------------------
# Reproducibility / device
# ----------------------------
SEED = 424242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using Device:", device)
else:
    device = torch.device("cpu")
    print("Using Device: CPU")


# ----------------------------
# Parameters
# ----------------------------
field_size = 25
layer_size = 110
num_modes = 5
circle_focus_radius = 5
circle_detectsize = 10
eigenmode_focus_radius = 12.5
eigenmode_detectsize = 15
focus_radius = circle_focus_radius
detectsize = circle_detectsize
batch_size = 16

evaluation_mode = "superposition"  # options: "eigenmode", "superposition"
num_superposition_eval_samples = 1000
num_superposition_visual_samples = 3
label_pattern_mode = "circle"  # options: "eigenmode", "circle"
superposition_eval_seed = 20240116
show_detection_overlap_debug = True

training_dataset_mode = "eigenmode"  # options: "eigenmode", "superposition"
num_superposition_train_samples = 100
superposition_train_seed = 20240115

num_layer_option = [2, 3, 4, 5, 6]

# SLM / propagation parameters
z_layers = 40e-6
pixel_size = 1e-6
z_prop = 120e-6
z_input_to_first = 40e-6

# ✅ 多波长
wavelengths = np.array([650e-9, 1568e-9, 1900e-9], dtype=np.float32)
base_wavelength_idx = 0
L = len(wavelengths)


# phase sampling option (和旧代码一致)
phase_option = 4

# training hyperparams
epochs = 1000
lr = 1.99
padding_ratio = 0.5
use_apodization = True
apodization_width = 10

# ----------------------------
# NEW: Exports (phasemask / slices 风格)
# ----------------------------
export_multiwl_slices = True
export_multiwl_snapshots = True
export_phase_png = True
phase_png_wrap_2pi = True

slice_sample_mode = "random"  # "random" or "fixed"
slice_fixed_index = 0
slice_seed = 20251121

z_step = 5e-6
slice_kmax = 20

# ----------------------------
# Utils
# ----------------------------
def export_phase_masks_multiwl_per_wavelength(
    model: D2NNModelMultiWL,
    *,
    out_dir: str | Path,
    tag: str,
    wavelengths: np.ndarray,
    save_png: bool = True,
    wrap_to_2pi: bool = True,
    dpi: int = 300,
    cmap: str = "twilight",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model.layers not found or empty; cannot export phase masks.")

    wls = np.asarray(wavelengths, dtype=np.float32)
    L_local = int(wls.shape[0])

    phi0_list: list[np.ndarray] = []
    lam0_list: list[float] = []

    for layer in layers:
        phi0_list.append(layer.phase.detach().cpu().numpy().astype(np.float32))
        lam0_list.append(float(layer.lam0.detach().cpu().item()))

    phase_phi0 = np.stack(phi0_list, axis=0)
    lam0_arr = np.asarray(lam0_list, dtype=np.float32)

    phase_scaled = (
        phase_phi0[:, None, :, :] * (lam0_arr[:, None, None, None] / wls[None, :, None, None])
    ).astype(np.float32)

    if wrap_to_2pi:
        phase_phi0_vis = np.remainder(phase_phi0, 2 * np.pi).astype(np.float32)
        phase_scaled_vis = np.remainder(phase_scaled, 2 * np.pi).astype(np.float32)
        vmin, vmax = 0.0, 2 * np.pi
    else:
        phase_phi0_vis = phase_phi0
        phase_scaled_vis = phase_scaled
        vmin, vmax = None, None

    npz_path = out_dir / f"phase_masks_allwl_{tag}.npz"
    np.savez(
        npz_path,
        phase_phi0=phase_phi0,
        phase_scaled_by_lambda=phase_scaled,
        phase_phi0_vis=phase_phi0_vis,
        phase_scaled_by_lambda_vis=phase_scaled_vis,
        wavelengths_m=wls.astype(np.float64),
        lam0_per_layer_m=lam0_arr.astype(np.float64),
    )

    mat_path = out_dir / f"phase_masks_allwl_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "phase_phi0": phase_phi0,
            "phase_scaled_by_lambda": phase_scaled,
            "phase_phi0_vis": phase_phi0_vis,
            "phase_scaled_by_lambda_vis": phase_scaled_vis,
            "wavelengths_m": wls.astype(np.float64),
            "lam0_per_layer_m": lam0_arr.astype(np.float64),
        },
    )

    split_dir = out_dir / "per_wavelength"
    split_dir.mkdir(parents=True, exist_ok=True)

    for li in range(L_local):
        wl_nm = float(wls[li] * 1e9)
        wl_tag = f"{wl_nm:.1f}".replace(".", "p")

        phase_li = phase_scaled[:, li, :, :]

        npz_li = split_dir / f"phase_masks_lambda{wl_tag}nm_{tag}.npz"
        np.savez(
            npz_li,
            phase_scaled=phase_li,
            wavelength_m=np.array([wls[li]], dtype=np.float64),
            lam0_per_layer_m=lam0_arr.astype(np.float64),
        )

        mat_li = split_dir / f"phase_masks_lambda{wl_tag}nm_{tag}.mat"
        savemat(
            str(mat_li),
            {
                "phase_scaled": phase_li,
                "wavelength_m": np.array([wls[li]], dtype=np.float64),
                "lam0_per_layer_m": lam0_arr.astype(np.float64),
            },
        )

    if save_png:
        png_root = out_dir / "png"
        png_root.mkdir(parents=True, exist_ok=True)

        phi0_dir = png_root / "phi0"
        phi0_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in range(phase_phi0_vis.shape[0]):
            arr = phase_phi0_vis[layer_idx]
            fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("phase (rad)")
            ax.set_title(f"{tag} | phi0 | layer {layer_idx+1}/{phase_phi0_vis.shape[0]}" + (" | wrapped" if wrap_to_2pi else ""))
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(phi0_dir / f"{tag}_phi0_layer{layer_idx+1:02d}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        per_wl_dir = png_root / "per_wavelength"
        per_wl_dir.mkdir(parents=True, exist_ok=True)

        for li in range(L_local):
            wl_nm = float(wls[li] * 1e9)
            wl_tag = f"{wl_nm:.1f}".replace(".", "p")
            wl_dir = per_wl_dir / f"lambda_{wl_tag}nm"
            wl_dir.mkdir(parents=True, exist_ok=True)

            for layer_idx in range(phase_scaled_vis.shape[0]):
                arr = phase_scaled_vis[layer_idx, li]
                fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("phase (rad)")
                ax.set_title(
                    f"{tag} | λ={wl_nm:.1f} nm (idx {li}) | layer {layer_idx+1}/{phase_scaled_vis.shape[0]}"
                    + (" | wrapped" if wrap_to_2pi else "")
                )
                ax.set_axis_off()
                fig.tight_layout()
                fig.savefig(wl_dir / f"{tag}_lambda{wl_tag}nm_layer{layer_idx+1:02d}.png", dpi=dpi, bbox_inches="tight")
                plt.close(fig)

        print(f"✔ Saved phase mask PNGs -> {png_root}")

    print(f"✔ Saved phase masks (all wavelengths) -> {npz_path}")
    print(f"✔ Saved phase masks (all wavelengths) MAT -> {mat_path}")
    print(f"✔ Saved per-wavelength phase masks -> {split_dir}")

    return {
        "npz_path": str(npz_path),
        "mat_path": str(mat_path),
        "split_dir": str(split_dir),
        "png_dir": str(out_dir / "png") if save_png else "",
    }

def save_phase_masks_grid_png(
    phase_stack_nhw: np.ndarray,
    *,
    save_path: str | Path,
    title: str,
    cmap: str = "hsv",
    wrap_to_2pi: bool = True,
    dpi: int = 300,
):
    phase = np.asarray(phase_stack_nhw, dtype=np.float32)
    if phase.ndim != 3:
        raise ValueError(f"phase_stack_nhw must be (N,H,W), got {phase.shape}")

    if wrap_to_2pi:
        phase = np.remainder(phase, 2 * np.pi)
        vmin, vmax = 0.0, 2 * np.pi
    else:
        vmin, vmax = None, None

    N = phase.shape[0]
    fig_w = max(10, 2.6 * N)
    fig, axes = plt.subplots(1, N, figsize=(fig_w, 3.0), squeeze=False)
    axes = axes[0]

    for i in range(N):
        ax = axes[i]
        im = ax.imshow(phase[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Layer {i+1}", fontsize=10)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Phase (rad)", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_phase_masks_grid_wl_x_layer_png(
    phase_scaled_by_lambda_lLHW: np.ndarray,
    wavelengths_m: np.ndarray,
    *,
    save_path: str | Path,
    title: str,
    cmap: str = "hsv",
    wrapped: bool = True,
    dpi: int = 300,
):
    arr = np.asarray(phase_scaled_by_lambda_lLHW, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected (N_layers,L,H,W), got {arr.shape}")

    N_layers, L_local, _, _ = arr.shape
    wls = np.asarray(wavelengths_m, dtype=np.float64).reshape(-1)
    if wls.shape[0] != L_local:
        raise ValueError(f"wavelengths_m length {wls.shape[0]} != L {L_local}")

    if wrapped:
        vmin, vmax = 0.0, 2 * np.pi
    else:
        vmin, vmax = None, None

    fig_w = max(10, 2.2 * N_layers) + 0.9
    fig_h = max(3.5, 2.0 * L_local)
    fig, axes = plt.subplots(L_local, N_layers, figsize=(fig_w, fig_h), squeeze=False)

    im_ref = None
    for li in range(L_local):
        wl_nm = float(wls[li] * 1e9)
        for ni in range(N_layers):
            ax = axes[li, ni]
            im = ax.imshow(arr[ni, li], cmap=cmap, vmin=vmin, vmax=vmax)
            if im_ref is None:
                im_ref = im
            ax.set_axis_off()

            if li == 0:
                ax.set_title(f"Layer {ni+1}", fontsize=10)
            if ni == 0:
                ax.text(
                    -0.06, 0.5, f"λ={wl_nm:.1f} nm",
                    transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=10
                )

    fig.suptitle(title, fontsize=12)

    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    cax = fig.add_axes([0.94, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label("Phase (rad)", fontsize=10)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_mode_context(base_modes: np.ndarray, num_modes: int) -> dict:
    if base_modes.shape[2] < num_modes:
        raise ValueError(
            f"Requested {num_modes} modes, but source file only has {base_modes.shape[2]}."
        )
    mmf_data = base_modes[:, :, :num_modes].transpose(2, 0, 1)
    mmf_data_amp_norm = (np.abs(mmf_data) - np.min(np.abs(mmf_data))) / (
        np.max(np.abs(mmf_data)) - np.min(np.abs(mmf_data))
    )
    mmf_data = mmf_data_amp_norm * np.exp(1j * np.angle(mmf_data))

    if phase_option in [1, 2, 3, 5]:
        base_amplitudes_local, base_phases_local = generate_complex_weights(
            1000, num_modes, phase_option
        )
    elif phase_option == 4:
        base_amplitudes_local = np.eye(num_modes, dtype=np.float32)
        base_phases_local = np.eye(num_modes, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported phase_option: {phase_option}")

    return {
        "mmf_data_np": mmf_data,
        "mmf_data_ts": torch.from_numpy(mmf_data),
        "base_amplitudes": base_amplitudes_local,
        "base_phases": base_phases_local,
    }


def amplitudes_to_yvec(amplitudes: np.ndarray) -> torch.Tensor:
    """amplitudes: (N,M) -> y_vec: (N,M) 能量比例"""
    amp = torch.from_numpy(amplitudes.astype(np.float32))
    e = amp ** 2
    return e / (e.sum(dim=1, keepdim=True) + 1e-12)


# ============================================================
# ✅ NEW: 自由空间角谱传播 + 按波长生成物理真实标签
# ============================================================

def free_space_propagate(
    field_hw: torch.Tensor,
    wavelength: float,
    z_total: float,
    pixel_size_m: float,
    target_size: int,
    padding_ratio_val: float = 0.5,
    dev: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    对单个 2D 复数场做自由空间角谱传播，返回输出平面的强度 (H, W)。
    """
    H, W = field_hw.shape
    if H != target_size or W != target_size:
        E = pad_field_to_layer(field_hw.unsqueeze(0), target_size).squeeze(0)
    else:
        E = field_hw.clone()
    E = E.to(device=dev, dtype=torch.complex64)

    pad_px = int(round(target_size * padding_ratio_val))
    N_pad = target_size + 2 * pad_px

    E_pad = complex_pad(E, pad_px, pad_px)

    fx = torch.fft.fftfreq(N_pad, d=pixel_size_m, device=dev)
    fy = torch.fft.fftfreq(N_pad, d=pixel_size_m, device=dev)
    fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

    k = 2 * math.pi / wavelength
    kz_sq = k**2 - (2 * math.pi * fxx)**2 - (2 * math.pi * fyy)**2

    kz = torch.zeros_like(kz_sq)
    propagating = kz_sq > 0
    kz[propagating] = torch.sqrt(kz_sq[propagating])

    H_transfer = torch.exp(1j * kz * z_total)
    H_transfer[~propagating] = 0.0

    E_freq = torch.fft.fft2(E_pad)
    E_out_pad = torch.fft.ifft2(E_freq * H_transfer)

    E_out = complex_crop(E_out_pad, target_size, target_size, pad_px, pad_px)
    I_out = torch.abs(E_out) ** 2
    return I_out


def compute_per_wavelength_labels(
    amplitudes: np.ndarray,
    mmf_modes: torch.Tensor,
    wls: np.ndarray,
    masks: torch.Tensor,
    ls: int,
    px: float,
    z_tot: float,
    dev: torch.device,
    pad_ratio: float = 0.5,
) -> torch.Tensor:
    """
    为每个样本、每个波长生成物理真实的能量比例标签。

    amplitudes: (N, M)
    mmf_modes:  (M, H_field, W_field) complex
    wls:        (L,) wavelengths in meters
    masks:      (M_roi, H, W) on device
    ls:         layer_size
    px:         pixel_size
    z_tot:      总自由空间传播距离
    dev:        device
    pad_ratio:  padding_ratio

    Returns: (N, L, M_roi) float tensor on CPU
    """
    N, M = amplitudes.shape
    L_wl = len(wls)
    M_roi = masks.shape[0]

    # Step 1 & 2: 计算每个模式在每个波长下的 ROI 能量分布
    E_ref = torch.zeros(M, L_wl, M_roi, dtype=torch.float32, device=dev)

    print(f"  Computing per-wavelength labels (z_total={z_tot*1e6:.1f} μm) ...")
    with torch.no_grad():
        for m in range(M):
            mode_field = mmf_modes[m]
            for li in range(L_wl):
                lam = float(wls[li])
                I_out = free_space_propagate(
                    field_hw=mode_field,
                    wavelength=lam,
                    z_total=z_tot,
                    pixel_size_m=px,
                    target_size=ls,
                    padding_ratio_val=pad_ratio,
                    dev=dev,
                )
                for k in range(M_roi):
                    E_ref[m, li, k] = (I_out * masks[k]).sum()

    # debug 打印
    for m in range(M):
        for li in range(L_wl):
            wl_nm = wls[li] * 1e9
            ratios = E_ref[m, li] / (E_ref[m, li].sum() + 1e-12)
            print(f"    Mode {m}, λ={wl_nm:.1f}nm: ROI ratios = "
                  f"{ratios.cpu().numpy().round(4)}")

    # Step 3: 对每个样本，按振幅的平方加权（不相干叠加）
    amp_sq = torch.from_numpy(
        (amplitudes ** 2).astype(np.float32)
    ).to(dev)

    # einsum: (N,M) x (M,L,K) -> (N,L,K)
    y_energy = torch.einsum('nm, mlk -> nlk', amp_sq, E_ref)

    # 归一化为比例
    y_vec = y_energy / (y_energy.sum(dim=2, keepdim=True) + 1e-12)

    return y_vec.cpu()


def intensity_to_roi_energies(I_blhw: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
    """
    I_blhw: (B,L,H,W) float intensity
    roi_masks: (M,H,W) float
    return: (B,L,M) energies
    """
    I_blhw = I_blhw.to(torch.float32)
    roi_masks = roi_masks.to(torch.float32)
    return (I_blhw.unsqueeze(2) * roi_masks.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))


def evaluate_ratio_metrics(pred_ratio: torch.Tensor, y_true: torch.Tensor) -> dict:
    pred = pred_ratio.detach().cpu().float()
    true = y_true.detach().cpu().float()

    err = pred - true
    mae = float(err.abs().mean().item())
    rmse = float(torch.sqrt((err ** 2).mean()).item())
    cos = float(F.cosine_similarity(pred, true, dim=1).mean().item())

    p = pred.flatten()
    t = true.flatten()
    p0 = p - p.mean()
    t0 = t - t.mean()
    corr = float((p0 @ t0 / (torch.sqrt((p0 @ p0) + 1e-12) * torch.sqrt((t0 @ t0) + 1e-12))).item())

    return {"mae": mae, "rmse": rmse, "cosine": cos, "pearson": corr}


def save_regression_diagnostics(
    *,
    model: D2NNModelMultiWL,
    dataset: TensorDataset,
    roi_masks: torch.Tensor,
    evaluation_regions: list[tuple[int,int,int,int]],
    output_dir: Path,
    device: torch.device,
    tag: str,
    wavelengths: np.ndarray,
    num_samples: int = 3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    L_local = int(len(wavelengths))
    N = len(dataset)
    take = min(num_samples, N)
    idxs = list(range(take))

    with torch.no_grad():
        for idx in idxs:
            img, y = dataset[idx]
            img = img.to(device, dtype=torch.complex64)[None, ...]
            y = y.to(device, dtype=torch.float32)[None, ...]

            x = img.repeat(1, L_local, 1, 1).contiguous()
            I_blhw = model(x)

            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks)
            pred_ratio = pred_energy / (pred_energy.sum(dim=2, keepdim=True) + 1e-12)

            # --- SUM over wavelengths
            I_sum = I_blhw.sum(dim=1)[0].detach().cpu().numpy()
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
            im = ax.imshow(I_sum, cmap="inferno")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{tag} | sample#{idx} | output intensity (sum over λ)")
            ax.set_axis_off()

            circle_radius = focus_radius
            for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
                color = plt.cm.tab20(idx_region % 20)
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                circ = Circle((cx, cy), radius=circle_radius, linewidth=1.0, edgecolor=color, linestyle="--", fill=False)
                ax.add_patch(circ)
                ax.text(
                    x0 + 1, y0 + 4, f"M{idx_region + 1}",
                    color=color, fontsize=8, weight="bold",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4, edgecolor="none"),
                )

            fig.tight_layout()
            fig.savefig(output_dir / f"{tag}_sample{idx:04d}_intensity_sum.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- per wavelength intensity + ratio bars
            for li in range(L_local):
                wl_nm = float(wavelengths[li] * 1e9)

                I_li = I_blhw[0, li].detach().cpu().numpy()
                figI, axI = plt.subplots(1, 1, figsize=(5.5, 5))
                im2 = axI.imshow(I_li, cmap="inferno")
                figI.colorbar(im2, ax=axI, fraction=0.046, pad=0.04)
                axI.set_title(f"{tag} | sample#{idx} | output intensity (λ idx={li}, {wl_nm:.1f} nm)")
                axI.set_axis_off()

                circle_radius = focus_radius
                for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
                    color = plt.cm.tab20(idx_region % 20)
                    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=color, facecolor="none")
                    axI.add_patch(rect)
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    circ = Circle((cx, cy), radius=circle_radius, linewidth=1.0, edgecolor=color, linestyle="--", fill=False)
                    axI.add_patch(circ)

                figI.tight_layout()
                figI.savefig(output_dir / f"{tag}_sample{idx:04d}_intensity_l{li}_{wl_nm:.1f}nm.png",
                             dpi=300, bbox_inches="tight")
                plt.close(figI)

                y_np = y[0, li].detach().cpu().numpy()
                p_np = pred_ratio[0, li].detach().cpu().numpy()

                fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3.2))
                x_axis = np.arange(len(y_np))
                ax2.bar(x_axis - 0.15, y_np, width=0.3, label="true")
                ax2.bar(x_axis + 0.15, p_np, width=0.3, label="pred")
                ax2.set_xticks(x_axis)
                ax2.set_xticklabels([f"M{i+1}" for i in range(len(y_np))])
                ax2.set_ylim(0, 1.0)
                ax2.grid(True, alpha=0.3)
                ax2.set_title(f"{tag} | sample#{idx} | ratio true vs pred (λ idx={li}, {wl_nm:.1f} nm)")
                ax2.legend()
                fig2.tight_layout()
                fig2.savefig(output_dir / f"{tag}_sample{idx:04d}_ratio_l{li}_{wl_nm:.1f}nm.png",
                             dpi=300, bbox_inches="tight")
                plt.close(fig2)


def build_uniform_fractions(n: int, *, include_endpoints: bool = False) -> tuple[float, ...]:
    if n <= 0:
        return ()
    if include_endpoints:
        vals = np.linspace(0.0, 1.0, n, dtype=np.float64)
        return tuple(float(x) for x in vals)
    vals = np.linspace(1.0/(n+1), n/(n+1), n, dtype=np.float64)
    return tuple(float(x) for x in vals)

# ----------------------------
# Load eigenmode data
# ----------------------------
eigenmodes_OM4 = load_complex_modes_from_mat(
    "mmf_103modes_25_PD_1.15.mat",
    key="modes_field"
)
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]


# ----------------------------
# Generate label layout (用于 detector 布局可视化)
# ----------------------------
pred_case = 1
label_size = layer_size
if pred_case != 1:
    raise ValueError("This script assumes pred_case == 1.")

num_detector = num_modes
detector_focus_radius = focus_radius
detector_detectsize = detectsize

if label_pattern_mode == "eigenmode":
    pattern_stack = np.transpose(np.abs(MMF_data), (1, 2, 0))
    pattern_h, pattern_w, _ = pattern_stack.shape
    if pattern_h > label_size or pattern_w > label_size:
        raise ValueError(
            f"Eigenmode pattern size ({pattern_h}x{pattern_w}) exceeds label canvas {label_size}."
        )
    layout_radius = math.ceil(max(pattern_h, pattern_w) / 2)
    detector_focus_radius = eigenmode_focus_radius
    detector_detectsize = eigenmode_detectsize
elif label_pattern_mode == "circle":
    circle_radius = circle_focus_radius
    pattern_size = circle_radius * 2
    if pattern_size % 2 == 0:
        pattern_size += 1
    pattern_stack = generate_detector_patterns(pattern_size, pattern_size, num_detector, shape="circle")
    layout_radius = circle_radius
    detector_focus_radius = circle_radius
    detector_detectsize = circle_detectsize
else:
    raise ValueError(f"Unknown label_pattern_mode: {label_pattern_mode}")

centers, _, _ = compute_label_centers(label_size, label_size, num_detector, layout_radius)
mode_label_maps = [
    compose_labels_from_patterns(
        label_size,
        label_size,
        pattern_stack,
        centers,
        Index=i + 1,
        visualize=False,
    )
    for i in range(num_detector)
]
MMF_Label_data = torch.from_numpy(np.stack(mode_label_maps, axis=2).astype(np.float32))

focus_radius = detector_focus_radius
detectsize = detector_detectsize


# ----------------------------
# Detection regions (debug)
# ----------------------------
evaluation_regions = create_evaluation_regions(layer_size, layer_size, num_detector, focus_radius, detectsize)
print("Detection Regions:", evaluation_regions)

if show_detection_overlap_debug:
    detection_debug_dir = Path("results/detection_region_debug")
    detection_debug_dir.mkdir(parents=True, exist_ok=True)
    overlap_map = np.zeros((layer_size, layer_size), dtype=np.float32)
    for (x0, x1, y0, y1) in evaluation_regions:
        overlap_map[y0:y1, x0:x1] += 1.0
    overlap_pixels = int(np.count_nonzero(overlap_map > 1.0 + 1e-6))
    max_overlap = float(overlap_map.max()) if overlap_map.size else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.zeros((layer_size, layer_size), dtype=np.float32), cmap="Greys")
    axes[0].set_title("Detector layout")
    axes[0].set_axis_off()

    circle_radius = focus_radius
    for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        color = plt.cm.tab20(idx_region % 20)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=color, facecolor="none")
        axes[0].add_patch(rect)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        circ = Circle((cx, cy), radius=circle_radius, linewidth=1.0, edgecolor=color, linestyle="--", fill=False)
        axes[0].add_patch(circ)

    im1 = axes[1].imshow(overlap_map, cmap="viridis")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Detector coverage count (overlap map)")
    axes[1].set_axis_off()

    overlap_plot_path = detection_debug_dir / f"detection_overlap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(overlap_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if overlap_pixels > 0:
        print(f"⚠ Detection regions overlap detected: {overlap_pixels} pixels have >1 coverage (max {max_overlap:.1f}).")
    else:
        print("✔ No overlap detected between evaluation regions.")
    print(f"✔ Detection region debug plot saved -> {overlap_plot_path}")


# ----------------------------
# ROI masks for RegressionDetector (M,H,W)
# ----------------------------
roi_stack, _ = build_circular_roi_masks(
    height=layer_size,
    width=layer_size,
    num_spots=num_modes,
    focus_radius=int(focus_radius),
    radius_scale=1.0,
)
roi_masks = torch.tensor(roi_stack, dtype=torch.float32, device=device)
print("roi_masks:", tuple(roi_masks.shape))


# ----------------------------
# ✅ CHANGED: Build training & test datasets (标签按波长不同)
# 函数签名增加 num_layers_current 参数
# ----------------------------
def build_eigenmode_dataset(num_layers_current: int) -> tuple[TensorDataset, TensorDataset, dict]:
    if phase_option == 4:
        num_samples = num_modes
        amplitudes = base_amplitudes[:num_samples]
        phases = base_phases[:num_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_samples = amplitudes.shape[0]

    # ✅ 计算总自由空间传播距离
    z_total = z_input_to_first + (num_layers_current - 1) * z_layers + z_prop
    print(f"  [eigenmode] z_total for {num_layers_current} layers: {z_total*1e6:.1f} μm")

    # ✅ 为每个波长生成物理真实标签（替换旧的 amplitudes_to_yvec + repeat）
    y_vec = compute_per_wavelength_labels(
        amplitudes=amplitudes,
        mmf_modes=MMF_data_ts,
        wls=wavelengths,
        masks=roi_masks,
        ls=layer_size,
        px=pixel_size,
        z_tot=z_total,
        dev=device,
        pad_ratio=padding_ratio,
    )  # (N, L, M)

    complex_weights = amplitudes * np.exp(1j * phases)
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_samples, num_modes, field_size
    ).to(torch.complex64)

    dummy_label = torch.zeros([1, layer_size, layer_size], dtype=torch.float32)
    images_prepared = []
    for i in range(num_samples):
        img_i, _ = prepare_sample(image_data[i], dummy_label, layer_size)
        images_prepared.append(img_i)
    image_tensor = torch.stack(images_prepared, dim=0)

    ds = TensorDataset(image_tensor, y_vec)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, ds, meta


def build_superposition_dataset(
    num_samples: int,
    rng_seed: int,
    num_layers_current: int,
) -> tuple[TensorDataset, dict]:
    ctx = build_superposition_eval_context(
        num_samples,
        num_modes=num_modes,
        field_size=field_size,
        layer_size=layer_size,
        mmf_modes=MMF_data_ts,
        mmf_label_data=MMF_Label_data,
        batch_size=batch_size,
        second_mode_half_range=True,
        rng_seed=rng_seed,
    )
    tensor_dataset: TensorDataset = ctx["tensor_dataset"]
    images = tensor_dataset.tensors[0]
    amplitudes = ctx["amplitudes"]
    phases = ctx["phases"]

    # ✅ 计算总自由空间传播距离
    z_total = z_input_to_first + (num_layers_current - 1) * z_layers + z_prop
    print(f"  [superposition] z_total for {num_layers_current} layers: {z_total*1e6:.1f} μm")

    # ✅ 为每个波长生成物理真实标签
    y_vec = compute_per_wavelength_labels(
        amplitudes=amplitudes,
        mmf_modes=MMF_data_ts,
        wls=wavelengths,
        masks=roi_masks,
        ls=layer_size,
        px=pixel_size,
        z_tot=z_total,
        dev=device,
        pad_ratio=padding_ratio,
    )  # (N, L, M)

    ds = TensorDataset(images, y_vec)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, meta


# ============================================================
# ✅ CHANGED: 数据集构建移入循环内部（删除了循环外的构建代码）
# ============================================================

all_training_summaries: list[dict] = []
model_metrics: list[dict] = []

for num_layer in num_layer_option:
    print(f"\n{'='*60}")
    print(f"Training D2NNModelMultiWL with {num_layer} layers...")
    print(f"{'='*60}\n")

    # ✅ 每个层数重新构建数据集（z_total 不同 → 标签不同）
    if training_dataset_mode == "eigenmode":
        train_ds, _, train_meta = build_eigenmode_dataset(
            num_layers_current=num_layer
        )
    elif training_dataset_mode == "superposition":
        train_ds, train_meta = build_superposition_dataset(
            num_superposition_train_samples,
            superposition_train_seed,
            num_layers_current=num_layer,
        )
    else:
        raise ValueError(f"Unknown training_dataset_mode: {training_dataset_mode}")

    if evaluation_mode == "eigenmode":
        test_ds, _, test_meta = build_eigenmode_dataset(
            num_layers_current=num_layer
        )
    elif evaluation_mode == "superposition":
        test_ds, test_meta = build_superposition_dataset(
            num_superposition_eval_samples,
            superposition_eval_seed,
            num_layers_current=num_layer,
        )
    else:
        raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

    # 重建 dataloader
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("train images:", tuple(train_ds.tensors[0].shape), train_ds.tensors[0].dtype)
    print("train labels:", tuple(train_ds.tensors[1].shape), train_ds.tensors[1].dtype)
    print("test  images:", tuple(test_ds.tensors[0].shape), test_ds.tensors[0].dtype)
    print("test  labels:", tuple(test_ds.tensors[1].shape), test_ds.tensors[1].dtype)

    # ✅ debug: 确认不同波长标签确实不同
    print("\n  Label check (sample 0):")
    for li in range(L):
        wl_nm = wavelengths[li] * 1e9
        print(f"    λ={wl_nm:.1f}nm: {train_ds.tensors[1][0, li].numpy().round(4)}")

    # ---- 后续代码完全不变 ----

    model = D2NNModelMultiWL(
        num_layers=num_layer,
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelengths=wavelengths,
        device=device,
        padding_ratio=padding_ratio,
        z_input_to_first=float(z_input_to_first),
        base_wavelength_idx=base_wavelength_idx,
    ).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    losses = []
    epoch_durations: list[float] = []
    training_start_time = time.time()

    # sanity check
    model.train()
    with torch.no_grad():
        images0, y0 = next(iter(train_loader))
        images0 = images0.to(device, dtype=torch.complex64)
        x0 = images0.repeat(1, L, 1, 1).contiguous()
        I0 = model(x0)
        pred0 = intensity_to_roi_energies(I0, roi_masks)
        print(
            "Sanity check x:", tuple(x0.shape), x0.dtype,
            "| y:", tuple(y0.shape), y0.dtype,
            "| I:", tuple(I0.shape), I0.dtype,
            "| pred_energy:", tuple(pred0.shape), pred0.dtype
        )

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0

        for images, y in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            x = images.repeat(1, L, 1, 1).contiguous()

            optimizer.zero_grad(set_to_none=True)

            I_blhw = model(x)
            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks)
            pred_ratio  = pred_energy / (pred_energy.sum(dim=2, keepdim=True) + 1e-12)

            loss = F.mse_loss(pred_ratio, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()

        avg_loss = epoch_loss / max(1, len(train_loader))
        losses.append(avg_loss)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.12f}, "
                f"Epoch Time: {epoch_duration:.2f} seconds"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_training_time = time.time() - training_start_time
    print(
        f"Total training time for {num_layer}-layer model: {total_training_time:.2f} seconds "
        f"(~{total_training_time / 60:.2f} minutes)"
    )

    # ----------------------------
    # Save training curves
    # ----------------------------
    training_output_dir = Path("results/training_analysis")
    training_output_dir.mkdir(parents=True, exist_ok=True)

    epochs_array = np.arange(1, epochs + 1, dtype=np.int32)
    cumulative_epoch_times = np.cumsum(epoch_durations)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots()
    ax.plot(epochs_array, losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"MultiWL ROI-Ratio Regression Loss ({num_layer} layers)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    loss_plot_path = training_output_dir / f"loss_curve_multiwl_layers{num_layer}_m{num_modes}_ls{layer_size}_{timestamp_tag}.png"
    fig.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_time, ax_time = plt.subplots()
    ax_time.plot(epochs_array, cumulative_epoch_times, label="Cumulative Time")
    ax_time.set_xlabel("Epoch")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title(f"Cumulative Training Time ({num_layer} layers)")
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_time.legend()
    time_plot_path = training_output_dir / f"epoch_time_multiwl_layers{num_layer}_m{num_modes}_ls{layer_size}_{timestamp_tag}.png"
    fig_time.savefig(time_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig_time)

    mat_path = training_output_dir / f"training_curves_multiwl_layers{num_layer}_m{num_modes}_ls{layer_size}_{timestamp_tag}.mat"
    savemat(
        str(mat_path),
        {
            "epochs": epochs_array,
            "losses": np.array(losses, dtype=np.float64),
            "epoch_durations": np.array(epoch_durations, dtype=np.float64),
            "cumulative_epoch_times": np.array(cumulative_epoch_times, dtype=np.float64),
            "total_training_time": np.array([total_training_time], dtype=np.float64),
            "num_layers": np.array([num_layer], dtype=np.int32),
            "wavelengths": wavelengths.astype(np.float64),
            "base_wavelength_idx": np.array([base_wavelength_idx], dtype=np.int32),
            "z_input_to_first": np.array([z_input_to_first], dtype=np.float64),
        },
    )

    print(f"✔ Saved training loss plot -> {loss_plot_path}")
    print(f"✔ Saved cumulative time plot -> {time_plot_path}")
    print(f"✔ Saved training log data (.mat) -> {mat_path}")

    # ----------------------------
    # Save checkpoint
    # ----------------------------
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "meta": {
            "num_layers": int(num_layer),
            "layer_size": layer_size,
            "z_layers": float(z_layers),
            "z_prop": float(z_prop),
            "pixel_size": float(pixel_size),
            "wavelengths": wavelengths.tolist(),
            "base_wavelength_idx": int(base_wavelength_idx),
            "padding_ratio": float(padding_ratio),
            "field_size": int(field_size),
            "num_modes": int(num_modes),
            "z_input_to_first": float(z_input_to_first),
        }
    }
    save_path = os.path.join(ckpt_dir, f"odnn_multiwl_{int(num_layer)}layers_m{num_modes}_ls{layer_size}.pth")
    torch.save(ckpt, save_path)
    print("✔ Saved model ->", save_path)

    # ----------------------------
    # Export phase masks
    # ----------------------------
    phase_dir = Path("results/phase_masks_multiwl") / f"L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    tag = f"multiwl_L{num_layer}_m{num_modes}_ls{layer_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    export_info = export_phase_masks_multiwl_per_wavelength(
        model,
        out_dir=phase_dir,
        tag=tag,
        wavelengths=wavelengths,
        save_png=export_phase_png,
        wrap_to_2pi=phase_png_wrap_2pi,
        dpi=300,
        cmap="twilight",
    )

    data = np.load(export_info["npz_path"])
    phase_phi0_vis = data["phase_phi0_vis"]
    phase_scaled_vis = data["phase_scaled_by_lambda_vis"]
    wls_m = data["wavelengths_m"]

    save_phase_masks_grid_png(
        phase_phi0_vis,
        save_path=phase_dir / f"grid_phi0_layers{num_layer}_{tag}.png",
        title=f"Phase Masks (phi0) - {num_layer} Layers",
        cmap="hsv",
        wrap_to_2pi=False,
        dpi=300,
    )

    save_phase_masks_grid_wl_x_layer_png(
        phase_scaled_vis,
        wls_m,
        save_path=phase_dir / f"grid_wavelength_x_layer_layers{num_layer}_{tag}.png",
        title=f"Phase Masks - {num_layer} Layers (rows=λ, cols=layer)",
        cmap="hsv",
        wrapped=True,
        dpi=300,
    )

    print(f"✔ Phase export + grids saved -> {phase_dir}")

    # ============================================================
    # Export MultiWL propagation slices + key snapshots
    # ============================================================
    if export_multiwl_slices:
        if slice_sample_mode == "random":
            rng = np.random.default_rng(slice_seed)
            sample_idx = int(rng.integers(low=0, high=len(test_ds)))
        else:
            sample_idx = int(slice_fixed_index) % len(test_ds)

        input_E_1hw = test_ds.tensors[0][sample_idx].to(device, dtype=torch.complex64)

        slices_dir = Path("results/propagation_slices_multiwl") / f"L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        scans, camera_field = visualize_model_slices_multiwl(
            model,
            input_field=input_E_1hw,
            output_dir=slices_dir,
            sample_tag=f"multiwl_L{num_layer}_s{sample_idx:04d}",
            z_input_to_first=float(z_input_to_first),
            z_layers=float(z_layers),
            z_prop_plus=float(z_prop),
            z_step=float(z_step),
            pixel_size=float(pixel_size),
            wavelengths=wavelengths,
            kmax=int(slice_kmax),
            cmap="inferno",
        )
        np.savez(slices_dir / "camera_field_multiwl.npz", camera_field=camera_field)
        print(f"✔ Saved MultiWL slices -> {slices_dir}")

    if export_multiwl_snapshots:
        snap_dir = Path("results/propagation_snapshots_multiwl") / f"L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        eigenmode_index = min(2, MMF_data_ts.shape[0] - 1)

        fractions_per_segment = 10
        dense = np.linspace(
            1.0 / (fractions_per_segment + 1),
            fractions_per_segment / (fractions_per_segment + 1),
            fractions_per_segment,
            dtype=np.float64,
        )
        dense = tuple(float(x) for x in dense)

        num_layers_local = len(model.layers)
        fractions_between_layers = tuple(dense for _ in range(num_layers_local))
        output_fractions = dense

        # ✅ 修改：为每个波长单独生成 PNG
        for li, wl in enumerate(wavelengths):
            wl_nm = float(wl * 1e9)
            wl_tag = f"{wl_nm:.1f}".replace(".", "p")
            
            print(f"\n  Generating snapshots for λ={wl_nm:.1f} nm (idx {li})...")
            
            summary = capture_eigenmode_propagation_multiwl(
                model,
                eigenmode_field=MMF_data_ts[eigenmode_index],
                mode_index=int(eigenmode_index),
                layer_size=int(layer_size),
                z_input_to_first=float(z_input_to_first),
                z_layers=float(z_layers),
                z_prop=float(z_prop),
                pixel_size=float(pixel_size),
                wavelengths=wavelengths,
                output_dir=snap_dir,
                tag=f"multiwl_L{num_layer}_lambda{wl_tag}nm",
                base_wavelength_idx=int(li),  # ← 关键：使用当前波长索引
                fractions_between_layers=fractions_between_layers,
                output_fractions=output_fractions,
            )
            print(f"  ✔ Saved λ={wl_nm:.1f}nm PNG -> {summary['fig_path']}")
            print(f"  ✔ Saved λ={wl_nm:.1f}nm MAT -> {summary['mat_path']}")
        
        print(f"\n✔ All wavelength snapshots saved to -> {snap_dir}")


    # ----------------------------
    # Evaluate (ratio regression metrics)
    # ----------------------------
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for images, y in test_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            x = images.repeat(1, L, 1, 1).contiguous()
            I_blhw = model(x)
            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks)
            pred_ratio = pred_energy / (pred_energy.sum(dim=2, keepdim=True) + 1e-12)

            preds.append(pred_ratio.detach().cpu())
            trues.append(y.detach().cpu())

    pred_all = torch.cat(preds, dim=0)        # (N,L,M)
    true_all = torch.cat(trues, dim=0)        # (N,L,M)

    metrics = evaluate_ratio_metrics(
        pred_all.reshape(-1, num_modes),
        true_all.reshape(-1, num_modes),
    )
    metrics["num_layers"] = int(num_layer)
    metrics["evaluation_mode"] = evaluation_mode
    model_metrics.append(metrics)

    print(
        f"[Metrics | {num_layer} layers] "
        f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
        f"Cosine={metrics['cosine']:.6f}, Pearson={metrics['pearson']:.6f}"
    )

    # ----------------------------
    # Save diagnostics figures
    # ----------------------------
    diag_dir = Path("results/prediction_viz") / f"multiwl_L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_regression_diagnostics(
        model=model,
        dataset=test_ds,
        roi_masks=roi_masks,
        evaluation_regions=evaluation_regions,
        output_dir=diag_dir,
        device=device,
        tag=f"multiwl_L{num_layer}",
        wavelengths=wavelengths,
        num_samples=num_superposition_visual_samples if evaluation_mode == "superposition" else min(5, num_modes),
    )
    print(f"✔ Saved regression diagnostics -> {diag_dir}")

    all_training_summaries.append(
        {
            "num_layers": int(num_layer),
            "total_time": float(total_training_time),
            "loss_plot": str(loss_plot_path),
            "time_plot": str(time_plot_path),
            "mat_path": str(mat_path),
            "ckpt_path": str(save_path),
            "diagnostics_dir": str(diag_dir),
            "metrics": metrics,
        }
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Done.")


#%% Metrics vs. layer count (save plot + mat)
if model_metrics:
    metrics_dir = Path("results/metrics_analysis")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    layer_counts = np.asarray([m["num_layers"] for m in model_metrics], dtype=np.int32)
    mae = np.asarray([m["mae"] for m in model_metrics], dtype=np.float64)
    rmse = np.asarray([m["rmse"] for m in model_metrics], dtype=np.float64)
    cosine = np.asarray([m["cosine"] for m in model_metrics], dtype=np.float64)
    pearson = np.asarray([m["pearson"] for m in model_metrics], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    axes[0].plot(layer_counts, mae, marker="o")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layer_counts, rmse, marker="o", color="tab:orange")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(layer_counts, cosine, marker="o", color="tab:green")
    axes[2].set_ylabel("Cosine")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(layer_counts, pearson, marker="o", color="tab:purple")
    axes[3].set_xlabel("Number of layers")
    axes[3].set_ylabel("Pearson")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"Regression metrics vs. layer count ({evaluation_mode})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    metrics_plot_path = metrics_dir / f"metrics_vs_layers_multiwl_{metrics_tag}.png"
    fig.savefig(metrics_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics_mat_path = metrics_dir / f"metrics_vs_layers_multiwl_{metrics_tag}.mat"
    savemat(
        str(metrics_mat_path),
        {
            "layers": layer_counts.astype(np.float64),
            "mae": mae,
            "rmse": rmse,
            "cosine": cosine,
            "pearson": pearson,
        },
    )

    print(f"✔ Metrics vs. layers plot saved -> {metrics_plot_path}")
    print(f"✔ Metrics vs. layers data (.mat) -> {metrics_mat_path}")
#%%
