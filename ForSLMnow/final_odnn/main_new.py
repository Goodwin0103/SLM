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
wavelengths = np.array([1550e-9, 1568e-9, 1585e-9], dtype=np.float32)
base_wavelength_idx = 1
L = len(wavelengths)

# phase sampling option (和旧代码一致)
phase_option = 4

# training hyperparams
epochs = 1000
lr = 1.99
padding_ratio = 0.5
use_apodization = True  # 你的 D2NNModelMultiWL 里没用到这个参数；保留不影响
apodization_width = 10  # 同上


# ----------------------------
# Utils
# ----------------------------
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


def intensity_to_roi_energies(I_blhw: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
    """
    I_blhw: (B,L,H,W) float intensity
    roi_masks: (M,H,W) float
    return: (B,L,M) energies

    说明：不用 einsum，避免在 CUDA + deterministic 下触发 CuBLAS 非确定性路径。
    """
    I_blhw = I_blhw.to(torch.float32)
    roi_masks = roi_masks.to(torch.float32)

    # (B,L,1,H,W) * (1,1,M,H,W) -> (B,L,M,H,W) -> sum over H,W -> (B,L,M)
    return (I_blhw.unsqueeze(2) * roi_masks.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))

def evaluate_ratio_metrics(pred_ratio: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    pred_ratio, y_true: (N,M) float32
    输出：rmse/mae/cosine/corr 等
    """
    pred = pred_ratio.detach().cpu().float()
    true = y_true.detach().cpu().float()

    err = pred - true
    mae = float(err.abs().mean().item())
    rmse = float(torch.sqrt((err ** 2).mean()).item())

    cos = F.cosine_similarity(pred, true, dim=1).mean().item()

    p = pred.flatten()
    t = true.flatten()
    p0 = p - p.mean()
    t0 = t - t.mean()
    corr = float((p0 @ t0 / (torch.sqrt((p0 @ p0) + 1e-12) * torch.sqrt((t0 @ t0) + 1e-12))).item())

    return {"mae": mae, "rmse": rmse, "cosine": float(cos), "pearson": corr}


def save_regression_diagnostics(
    *,
    model: D2NNModelMultiWL,
    dataset: TensorDataset,
    roi_masks: torch.Tensor,                 # (M,H,W) on device
    evaluation_regions: list[tuple[int,int,int,int]],
    output_dir: Path,
    device: torch.device,
    tag: str,
    wavelengths: np.ndarray,                 # ✅ NEW: pass wavelengths
    num_samples: int = 3,
):
    """
    保存每个样本：
    - 输出面强度图：sum over λ 一张（总览）
    - 输出面强度图：每个 λ 单独一张（区分波长）
    - 真实/预测比例柱状图：每个 λ 单独一张（区分波长）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    L_local = int(len(wavelengths))
    N = len(dataset)
    take = min(num_samples, N)
    idxs = list(range(take))

    with torch.no_grad():
        for idx in idxs:
            img, y = dataset[idx]
            img = img.to(device, dtype=torch.complex64)[None, ...]  # (1,1,H,W)
            y = y.to(device, dtype=torch.float32)[None, ...]        # (1,L,M)

            # (1,L,H,W)
            x = img.repeat(1, L_local, 1, 1).contiguous()

            # (1,L,H,W) intensity
            I_blhw = model(x)

            # (1,L,M) energies -> ratios
            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks)
            pred_ratio = pred_energy / (pred_energy.sum(dim=2, keepdim=True) + 1e-12)

            # ----------------------------
            # Figure A: intensity SUM over wavelengths (overview)
            # ----------------------------
            I_sum = I_blhw.sum(dim=1)[0].detach().cpu().numpy()      # (H,W)

            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
            im = ax.imshow(I_sum, cmap="inferno")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{tag} | sample#{idx} | output intensity (sum over λ)")
            ax.set_axis_off()

            circle_radius = focus_radius
            for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
                color = plt.cm.tab20(idx_region % 20)
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=1.0, edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                circ = Circle((cx, cy), radius=circle_radius,
                              linewidth=1.0, edgecolor=color, linestyle="--", fill=False)
                ax.add_patch(circ)
                ax.text(
                    x0 + 1, y0 + 4, f"M{idx_region + 1}",
                    color=color, fontsize=8, weight="bold",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                              alpha=0.4, edgecolor="none"),
                )

            fig.tight_layout()
            fig_path = output_dir / f"{tag}_sample{idx:04d}_intensity_sum.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # ----------------------------
            # Figure B/C: per-wavelength intensity + per-wavelength ratio bars
            # ----------------------------
            for li in range(L_local):
                wl_nm = float(wavelengths[li] * 1e9)

                # ---- intensity @ wavelength li
                I_li = I_blhw[0, li].detach().cpu().numpy()

                figI, axI = plt.subplots(1, 1, figsize=(5.5, 5))
                im2 = axI.imshow(I_li, cmap="inferno")
                figI.colorbar(im2, ax=axI, fraction=0.046, pad=0.04)
                axI.set_title(f"{tag} | sample#{idx} | output intensity (λ idx={li}, {wl_nm:.1f} nm)")
                axI.set_axis_off()

                circle_radius = focus_radius
                for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
                    color = plt.cm.tab20(idx_region % 20)
                    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1.0, edgecolor=color, facecolor="none")
                    axI.add_patch(rect)
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    circ = Circle((cx, cy), radius=circle_radius,
                                  linewidth=1.0, edgecolor=color, linestyle="--", fill=False)
                    axI.add_patch(circ)

                figI.tight_layout()
                figI_path = output_dir / f"{tag}_sample{idx:04d}_intensity_l{li}_{wl_nm:.1f}nm.png"
                figI.savefig(figI_path, dpi=300, bbox_inches="tight")
                plt.close(figI)

                # ---- ratio bars @ wavelength li
                y_np = y[0, li].detach().cpu().numpy()                  # (M,)
                p_np = pred_ratio[0, li].detach().cpu().numpy()         # (M,)

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
                fig2_path = output_dir / f"{tag}_sample{idx:04d}_ratio_l{li}_{wl_nm:.1f}nm.png"
                fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
                plt.close(fig2)

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
# Build training & test datasets (NEW: label is y_vec not label_map)
# ----------------------------
def build_eigenmode_dataset() -> tuple[TensorDataset, TensorDataset, dict]:
    if phase_option == 4:
        num_samples = num_modes
        amplitudes = base_amplitudes[:num_samples]
        phases = base_phases[:num_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_samples = amplitudes.shape[0]

    # labels: (N,L,M)
    y_vec = amplitudes_to_yvec(amplitudes)                 # (N,M)
    y_vec = y_vec[:, None, :].repeat(1, L, 1).contiguous() # (N,L,M)

    # fields
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
    image_tensor = torch.stack(images_prepared, dim=0)  # (N,1,H,W) complex

    ds = TensorDataset(image_tensor, y_vec)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, ds, meta


def build_superposition_dataset(num_samples: int, rng_seed: int) -> tuple[TensorDataset, dict]:
    """
    复用 build_superposition_eval_context 生成的 image_data / amplitudes / phases，
    但把 label_map 替换成 y_vec（能量比例向量）。
    """
    ctx = build_superposition_eval_context(
        num_samples,
        num_modes=num_modes,
        field_size=field_size,
        layer_size=layer_size,
        mmf_modes=MMF_data_ts,
        mmf_label_data=MMF_Label_data,   # 这里只是为了让它跑通生成 dataset，我们不使用它的 label_map
        batch_size=batch_size,
        second_mode_half_range=True,
        rng_seed=rng_seed,
    )
    tensor_dataset: TensorDataset = ctx["tensor_dataset"]
    images = tensor_dataset.tensors[0]         # (N,1,H,W) complex
    amplitudes = ctx["amplitudes"]             # (N,M) numpy
    phases = ctx["phases"]                     # (N,M) numpy

    y_vec = amplitudes_to_yvec(amplitudes)                 # (N,M)
    y_vec = y_vec[:, None, :].repeat(1, L, 1).contiguous() # (N,L,M)

    ds = TensorDataset(images, y_vec)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, meta


if training_dataset_mode == "eigenmode":
    train_ds, _, train_meta = build_eigenmode_dataset()
elif training_dataset_mode == "superposition":
    train_ds, train_meta = build_superposition_dataset(num_superposition_train_samples, superposition_train_seed)
else:
    raise ValueError(f"Unknown training_dataset_mode: {training_dataset_mode}")

if evaluation_mode == "eigenmode":
    test_ds, _, test_meta = build_eigenmode_dataset()
elif evaluation_mode == "superposition":
    test_ds, test_meta = build_superposition_dataset(num_superposition_eval_samples, superposition_eval_seed)
else:
    raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")


# ----------------------------
# Dataloaders
# ----------------------------
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print("train images:", tuple(train_ds.tensors[0].shape), train_ds.tensors[0].dtype)
print("train labels:", tuple(train_ds.tensors[1].shape), train_ds.tensors[1].dtype)
print("test  images:", tuple(test_ds.tensors[0].shape), test_ds.tensors[0].dtype)
print("test  labels:", tuple(test_ds.tensors[1].shape), test_ds.tensors[1].dtype)


# ----------------------------
# Train & evaluate models
# ----------------------------
all_training_summaries: list[dict] = []
model_metrics: list[dict] = []

for num_layer in num_layer_option:
    print(f"\nTraining D2NNModelMultiWL with {num_layer} layers...\n")

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
        x0 = images0.repeat(1, L, 1, 1).contiguous()  # (B,L,H,W)
        I0 = model(x0)                                # (B,L,H,W)
        pred0 = intensity_to_roi_energies(I0, roi_masks)  # (B,L,M)
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
            images = images.to(device, dtype=torch.complex64, non_blocking=True)  # (B,1,H,W)
            y = y.to(device, dtype=torch.float32, non_blocking=True)              # (B,L,M)

            x = images.repeat(1, L, 1, 1).contiguous()                            # (B,L,H,W)

            optimizer.zero_grad(set_to_none=True)

            I_blhw = model(x)                                                     # (B,L,H,W)
            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks)            # (B,L,M)
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
    # Evaluate (ratio regression metrics)
    # ----------------------------
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for images, y in test_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            x = images.repeat(1, L, 1, 1).contiguous()                 # (B,L,H,W)
            I_blhw = model(x)                                          # (B,L,H,W)
            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks) # (B,L,M)
            pred_ratio = pred_energy / (pred_energy.sum(dim=2, keepdim=True) + 1e-12)

            preds.append(pred_ratio.detach().cpu())
            trues.append(y.detach().cpu())

    pred_all = torch.cat(preds, dim=0)        # (N,L,M)
    true_all = torch.cat(trues, dim=0)        # (N,L,M)

    metrics = evaluate_ratio_metrics(
        pred_all.reshape(-1, num_modes),      # (N*L, M)
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
        wavelengths=wavelengths,  # ✅ NEW
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
