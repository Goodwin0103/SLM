#%%
import math
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import time
from datetime import datetime
from pathlib import Path

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

# MultiWL model
from odnn_multiwl_model import D2NNModelMultiWL

# superposition sampler (提供 images/labels/amplitudes/phases)
from odnn_training_eval import build_superposition_eval_context


# ============================================================
# Reproducibility / device
# ============================================================
SEED = 424242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device:", device)


# ============================================================
# Parameters
# ============================================================
field_size = 25
layer_size = 110
num_modes = 5

circle_focus_radius = 5
circle_detectsize = 10  # 你原来当作“直径/窗口”，评估用半径 = detectsize//2
focus_radius = circle_focus_radius
detectsize = circle_detectsize

batch_size = 16

evaluation_mode = "superposition"      # "eigenmode" or "superposition"
training_dataset_mode = "eigenmode"    # "eigenmode" or "superposition"

num_superposition_eval_samples = 1000
num_superposition_train_samples = 100
superposition_eval_seed = 20240116
superposition_train_seed = 20240115

num_layer_option = [2, 3, 4, 5, 6]

# geometry / propagation params (metadata)
z_layers = 40e-6
pixel_size = 1e-6
z_prop = 120e-6
z_input_to_first = 40e-6

# wavelengths (MultiWL)
# wavelengths = np.array([1550e-9, 1568e-9], dtype=np.float32)
wavelengths = np.array([650e-9, 1568e-9], dtype=np.float32)
base_wavelength_idx = 0
L = int(len(wavelengths))

# data options
phase_option = 4
label_pattern_mode = "circle"  # "circle" or "eigenmode"
show_detection_overlap_debug = True

# train hyperparams
epochs = 400
lr = 1.99
padding_ratio = 0.5

# output root
RUN_ROOT = Path("results/650_1568_spatial_label")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# prediction viz samples
num_pred_diag_samples = 3
num_superposition_visual_samples = 2


# ============================================================
# Helpers (metrics / plotting / saving)
# ============================================================
def _safe_norm_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = v.sum(axis=-1, keepdims=True)
    return v / (s + eps)

def _per_sample_corrcoef(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.sqrt((a0 * a0).sum() + eps) * np.sqrt((b0 * b0).sum() + eps))
    return float((a0 * b0).sum() / denom)

def save_checkpoint_multiwl(model, out_path: Path, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(ckpt, str(out_path))

def extract_phase_masks_multiwl(model: D2NNModelMultiWL) -> list[np.ndarray]:
    masks = []
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "phase"):
            ph = layer.phase.detach().cpu().numpy()
            masks.append(np.remainder(ph, 2 * np.pi))
    return masks

def save_training_curves(
    *,
    losses: list[float],
    epoch_durations: list[float],
    out_dir: Path,
    tag: str,
    num_layers: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs_arr = np.arange(1, len(losses) + 1, dtype=np.int32)
    cum_times = np.cumsum(np.asarray(epoch_durations, dtype=np.float64))
    total_time = float(cum_times[-1]) if len(cum_times) else 0.0

    fig, ax = plt.subplots()
    ax.plot(epochs_arr, np.asarray(losses, dtype=np.float64), label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"MultiWL (spatial label) Training Loss ({num_layers} layers)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    loss_plot_path = out_dir / f"loss_curve_layers{num_layers}_{tag}.png"
    fig.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs_arr, cum_times, label="Cumulative Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Cumulative Training Time ({num_layers} layers)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    time_plot_path = out_dir / f"epoch_time_layers{num_layers}_{tag}.png"
    fig.savefig(time_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    mat_path = out_dir / f"training_curves_layers{num_layers}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "epochs": epochs_arr.astype(np.float64),
            "losses": np.asarray(losses, dtype=np.float64),
            "epoch_durations": np.asarray(epoch_durations, dtype=np.float64),
            "cumulative_epoch_times": cum_times,
            "total_training_time": np.asarray([total_time], dtype=np.float64),
            "num_layers": np.asarray([num_layers], dtype=np.int32),
        },
    )
    return {"loss_plot": loss_plot_path, "time_plot": time_plot_path, "mat": mat_path, "total_time": total_time}


def build_spatial_label_from_amplitudes(
    amplitudes: np.ndarray,
    mmf_label_data: torch.Tensor,  # (H,W,M) float32
) -> torch.Tensor:
    """
    legacy 空间 label：
      label(x,y) = Σ_k (amp_k^2 * pattern_k(x,y))
    return: (N,1,H,W) float32
    """
    amp = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))                # (N,M)
    energy = amp ** 2                                                              # (N,M)
    lbl = (energy[:, None, None, :] * mmf_label_data.unsqueeze(0)).sum(dim=3)      # (N,H,W)
    return lbl.unsqueeze(1).contiguous().to(torch.float32)


def _make_circle_mask(h: int, w: int, r: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
    return mask.to(torch.float32)


@torch.no_grad()
def region_energy_fractions(
    I_bhw: torch.Tensor,                  # (B,H,W) float
    evaluation_regions: list[tuple[int, int, int, int]],
    detect_radius: int,                   # 圆半径 px
) -> torch.Tensor:
    """
    在每个 region 的方框内套一个圆形 mask 积分能量，得到 (B,M) 能量比例（归一化到和=1）。
    """
    B, H, W = I_bhw.shape
    M = len(evaluation_regions)
    out = torch.zeros((B, M), device=I_bhw.device, dtype=torch.float32)

    for mi, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        patch = I_bhw[:, y0:y1, x0:x1]  # (B,hh,ww)
        hh, ww = patch.shape[-2], patch.shape[-1]
        cmask = _make_circle_mask(hh, ww, float(detect_radius), device=I_bhw.device)  # (hh,ww)
        out[:, mi] = (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))

    out = out / (out.sum(dim=1, keepdim=True) + 1e-12)
    return out


@torch.no_grad()
def evaluate_spot_metrics_regions_multiwl(
    model: D2NNModelMultiWL,
    loader: DataLoader,
    *,
    device: torch.device,
    evaluation_regions,
    detect_radius: int,
    wl_idx: int,
    L: int,
) -> dict:
    """
    legacy 风格（region-based）指标：
      true: amp -> energy_frac = amp^2 / sum -> amp_frac_true = sqrt(energy_frac)
      pred: output intensity -> region energy frac -> amp_frac_pred = sqrt(...)
    """
    model.eval()

    pred_amp_list, true_amp_list = [], []

    for images, label_img, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)     # (B,1,H,W)
        amp = amp.to(device, dtype=torch.float32, non_blocking=True)            # (B,M)

        if images.ndim == 3:
            images = images.unsqueeze(1)

        amp2 = amp ** 2
        true_energy_frac = amp2 / (amp2.sum(dim=1, keepdim=True) + 1e-12)
        true_amp_frac = torch.sqrt(true_energy_frac + 1e-12)
        true_amp_list.append(true_amp_frac.detach().cpu())

        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x)                             # (B,L,H,W)
        I_bhw = I_blhw[:, wl_idx].to(torch.float32)   # (B,H,W)

        pred_energy_frac = region_energy_fractions(
            I_bhw,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius,
        )                                             # (B,M)
        pred_amp_frac = torch.sqrt(pred_energy_frac + 1e-12)
        pred_amp_list.append(pred_amp_frac.detach().cpu())

    pred = torch.cat(pred_amp_list, dim=0).numpy()
    true = torch.cat(true_amp_list, dim=0).numpy()

    diff = pred - true
    abs_diff = np.abs(diff)
    rel = abs_diff / (np.abs(true) + 1e-12)
    cc = np.asarray([_per_sample_corrcoef(pred[i], true[i]) for i in range(pred.shape[0])], dtype=np.float64)

    return {
        "avg_amplitudes_diff": float(abs_diff.mean()),
        "avg_relative_amp_err": float(rel.mean()),
        "cc_recon_amp": cc,
        "amplitudes_diff": diff,
    }


@torch.no_grad()
def save_prediction_diagnostics_multiwl_spatial(
    model: D2NNModelMultiWL,
    dataset: TensorDataset,
    *,
    wavelengths: np.ndarray,
    evaluation_regions,
    detect_radius: int,
    sample_indices: list[int],
    out_dir: Path,
    device: torch.device,
    tag: str,
):
    """
    诊断图（空间 label 版）：
      - 输入 |E|^2
      - label 强度图
      - 每波长预测强度图
      - 每波长：region energy ratio (true vs pred) 柱状图
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    Lloc = int(len(wavelengths))

    saved = []
    for si in sample_indices:
        x, label_img, amp = dataset[si]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(device=device, dtype=torch.complex64).unsqueeze(0)          # (1,1,H,W)
        label_img = label_img.to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,1,H,W)
        amp = amp.to(device=device, dtype=torch.float32).unsqueeze(0)        # (1,M)

        xin = x.repeat(1, Lloc, 1, 1).contiguous()
        I_pred = model(xin)                                                  # (1,L,H,W)

        I_in = (torch.abs(x[0, 0]) ** 2).detach().cpu().numpy()
        I_lbl = label_img[0, 0].detach().cpu().numpy()

        # true ratios from amplitudes (match eval)
        amp2 = (amp[0] ** 2).detach().cpu().numpy()
        true_energy_frac = _safe_norm_np(amp2)

        fig = plt.figure(figsize=(4 * (Lloc + 2), 8))
        gs = fig.add_gridspec(2, Lloc + 2, height_ratios=[1.0, 1.0])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(I_in, cmap="inferno")
        ax0.set_title("Input |E|^2")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(I_lbl, cmap="inferno")
        ax1.set_title("Label (spatial)")
        ax1.axis("off")

        # draw detector regions overlay on label (optional)
        for (x0, x1, y0, y1) in evaluation_regions:
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.8, edgecolor="cyan", facecolor="none", alpha=0.9)
            ax1.add_patch(rect)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            ax1.add_patch(Circle((cx, cy), radius=detect_radius, linewidth=0.8, edgecolor="cyan", linestyle="--", fill=False, alpha=0.9))

        for li in range(Lloc):
            axI = fig.add_subplot(gs[0, li + 2])
            I_li = I_pred[0, li].detach().cpu().numpy()
            axI.imshow(I_li, cmap="inferno")
            axI.set_title(f"Pred I (λ={wavelengths[li]*1e9:.1f} nm)")
            axI.axis("off")

            # region ratios pred
            I_bhw = I_pred[:, li].to(torch.float32)  # (1,H,W)
            pred_energy_frac = region_energy_fractions(I_bhw, evaluation_regions, detect_radius=detect_radius)[0].detach().cpu().numpy()

            axb = fig.add_subplot(gs[1, li + 2])
            idx = np.arange(true_energy_frac.shape[0])
            axb.bar(idx - 0.15, true_energy_frac, width=0.3, label="true")
            axb.bar(idx + 0.15, pred_energy_frac, width=0.3, label="pred")
            axb.set_ylim(0, 1.0)
            axb.grid(True, alpha=0.3)
            axb.set_title(f"Region energy ratio (λ_idx={li})")
            if li == 0:
                axb.legend()

        # bottom-left two empty slots
        fig.add_subplot(gs[1, 0]).axis("off")
        fig.add_subplot(gs[1, 1]).axis("off")

        fig.tight_layout()
        out_path = out_dir / f"{tag}_sample{si:04d}.png"
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


# ============================================================
# Mode context + labels (legacy-style detector patterns)
# ============================================================
def build_mode_context(base_modes: np.ndarray, num_modes: int) -> dict:
    if base_modes.shape[2] < num_modes:
        raise ValueError("Requested modes exceed file modes.")
    mmf_data = base_modes[:, :, :num_modes].transpose(2, 0, 1)

    mmf_data_amp_norm = (np.abs(mmf_data) - np.min(np.abs(mmf_data))) / (
        (np.max(np.abs(mmf_data)) - np.min(np.abs(mmf_data))) + 1e-12
    )
    mmf_data = mmf_data_amp_norm * np.exp(1j * np.angle(mmf_data))

    if phase_option in [1, 2, 3, 5]:
        base_amplitudes_local, base_phases_local = generate_complex_weights(1000, num_modes, phase_option)
    elif phase_option == 4:
        base_amplitudes_local = np.eye(num_modes, dtype=np.float32)
        base_phases_local = np.eye(num_modes, dtype=np.float32)
    else:
        raise ValueError("Unsupported phase_option")

    return {
        "mmf_data_np": mmf_data,
        "mmf_data_ts": torch.from_numpy(mmf_data),
        "base_amplitudes": base_amplitudes_local,
        "base_phases": base_phases_local,
    }


# ============================================================
# Load eigenmodes
# ============================================================
eigenmodes_OM4 = load_complex_modes_from_mat("mmf_103modes_25_PD_1.15.mat", key="modes_field")
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]


# ============================================================
# Build detector layout / evaluation regions (legacy-style)
# ============================================================
label_size = layer_size
num_detector = num_modes

if label_pattern_mode == "eigenmode":
    pattern_stack = np.transpose(np.abs(MMF_data), (1, 2, 0))
    pattern_h, pattern_w, _ = pattern_stack.shape
    layout_radius = math.ceil(max(pattern_h, pattern_w) / 2)
elif label_pattern_mode == "circle":
    circle_radius = circle_focus_radius
    pattern_size = circle_radius * 2
    if pattern_size % 2 == 0:
        pattern_size += 1
    pattern_stack = generate_detector_patterns(pattern_size, pattern_size, num_detector, shape="circle")
    layout_radius = circle_radius
else:
    raise ValueError("Unknown label_pattern_mode")

centers, _, _ = compute_label_centers(label_size, label_size, num_detector, layout_radius)
mode_label_maps = [
    compose_labels_from_patterns(label_size, label_size, pattern_stack, centers, Index=i + 1, visualize=False)
    for i in range(num_detector)
]
MMF_Label_data = torch.from_numpy(np.stack(mode_label_maps, axis=2).astype(np.float32))  # (H,W,M)

evaluation_regions = create_evaluation_regions(layer_size, layer_size, num_detector, focus_radius, detectsize)
print("Detection Regions:", evaluation_regions)

# overlap debug
if show_detection_overlap_debug:
    detection_debug_dir = RUN_ROOT / "detection_region_debug"
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

    detect_radius_eval = int(detectsize // 2)
    for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        color = plt.cm.tab20(idx_region % 20)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=color, facecolor="none")
        axes[0].add_patch(rect)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        axes[0].add_patch(Circle((cx, cy), radius=detect_radius_eval, linewidth=1.0, edgecolor=color, linestyle="--", fill=False))

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


# ============================================================
# Dataset builders (spatial label + amplitudes)
# ============================================================
def build_eigenmode_dataset() -> tuple[TensorDataset, dict]:
    if phase_option == 4:
        num_samples = num_modes
        amplitudes = base_amplitudes[:num_samples]
        phases = base_phases[:num_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_samples = amplitudes.shape[0]

    # input fields
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

    # spatial label (legacy)
    label_img = build_spatial_label_from_amplitudes(amplitudes, MMF_Label_data).cpu()  # (N,1,H,W)

    # amplitudes tensor
    amp_tensor = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))            # (N,M)

    ds = TensorDataset(image_tensor, label_img, amp_tensor)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, meta


def build_superposition_dataset(num_samples: int, rng_seed: int) -> tuple[TensorDataset, dict]:
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
    images = tensor_dataset.tensors[0]     # (N,1,H,W) complex
    label_img = tensor_dataset.tensors[1]  # (N,1,H,W) float (legacy spatial label)
    amplitudes = ctx["amplitudes"]
    phases = ctx["phases"]

    amp_tensor = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))  # (N,M)
    ds = TensorDataset(images, label_img, amp_tensor)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, meta


# ============================================================
# Train/Eval loop
# ============================================================
all_losses: list[list[float]] = []
metrics_by_wl: dict[int, list[dict]] = {int(li): [] for li in range(L)}

detect_radius_eval = int(detectsize // 2)

for num_layer in num_layer_option:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*70}\nTraining D2NNModelMultiWL (spatial label) with {num_layer} layers\n{'='*70}")

    # datasets (labels 不再依赖 num_layer；和 legacy 更一致)
    if training_dataset_mode == "eigenmode":
        train_ds, train_meta = build_eigenmode_dataset()
    elif training_dataset_mode == "superposition":
        train_ds, train_meta = build_superposition_dataset(num_superposition_train_samples, superposition_train_seed)
    else:
        raise ValueError("Unknown training_dataset_mode")

    if evaluation_mode == "eigenmode":
        test_ds, test_meta = build_eigenmode_dataset()
    elif evaluation_mode == "superposition":
        test_ds, test_meta = build_superposition_dataset(num_superposition_eval_samples, superposition_eval_seed)
    else:
        raise ValueError("Unknown evaluation_mode")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # model
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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    # train (loss = spatial label MSE, replicated across wavelengths)
    losses: list[float] = []
    epoch_durations: list[float] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for images, label_img, amp in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)      # (B,1,H,W)
            label_img = label_img.to(device, dtype=torch.float32, non_blocking=True)  # (B,1,H,W)

            if images.ndim == 3:
                images = images.unsqueeze(1)

            x = images.repeat(1, L, 1, 1).contiguous()             # (B,L,H,W)
            label_blhw = label_img.repeat(1, L, 1, 1).contiguous()  # (B,L,H,W)

            optimizer.zero_grad(set_to_none=True)
            I_blhw = model(x)                                      # (B,L,H,W)
            loss = F.mse_loss(I_blhw, label_blhw)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        losses.append(avg_loss)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_durations.append(time.time() - epoch_t0)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] loss={avg_loss:.10f}")

    total_time = time.time() - t0
    all_losses.append(losses)
    print(f"Training done: {num_layer} layers, time={total_time:.2f}s")

    # ===== 输出 1：training_analysis =====
    training_output_dir = RUN_ROOT / "training_analysis"
    train_logs = save_training_curves(
        losses=losses,
        epoch_durations=epoch_durations,
        out_dir=training_output_dir,
        tag=f"spatial_m{num_modes}_ls{layer_size}_L{num_layer}_{run_tag}",
        num_layers=num_layer,
    )
    print(f"✔ Saved training loss plot -> {train_logs['loss_plot']}")
    print(f"✔ Saved cumulative time plot -> {train_logs['time_plot']}")
    print(f"✔ Saved training log data (.mat) -> {train_logs['mat']}")

    # ===== 输出 2：checkpoint =====
    ckpt_dir = RUN_ROOT / "checkpoints"
    ckpt_path = ckpt_dir / f"multiwl_spatial_{num_layer}layers_m{num_modes}_ls{layer_size}.pth"
    save_checkpoint_multiwl(
        model,
        ckpt_path,
        meta={
            "num_layers": int(num_layer),
            "layer_size": int(layer_size),
            "z_layers": float(z_layers),
            "z_prop": float(z_prop),
            "pixel_size": float(pixel_size),
            "wavelengths": wavelengths.astype(np.float32),
            "padding_ratio": float(padding_ratio),
            "field_size": int(field_size),
            "num_modes": int(num_modes),
            "z_input_to_first": float(z_input_to_first),
            "base_wavelength_idx": int(base_wavelength_idx),
            "epochs": int(epochs),
            "lr": float(lr),
            "training_dataset_mode": training_dataset_mode,
            "evaluation_mode": evaluation_mode,
            "label_type": "spatial_label_like_legacy",
            "total_training_time_sec": float(total_time),
        },
    )
    print("✔ Saved model ->", ckpt_path)

    # ===== 输出 3：phase masks =====
    phase_masks = extract_phase_masks_multiwl(model)
    if phase_masks:
        pm_dir = RUN_ROOT / "phase_masks" / f"L{num_layer}_{run_tag}"
        pm_dir.mkdir(parents=True, exist_ok=True)
        pm_mat = pm_dir / "phase_masks.mat"
        savemat(str(pm_mat), {"phase_masks": np.stack(phase_masks, axis=0).astype(np.float32)})
        print(f"✔ Saved phase masks (.mat) -> {pm_mat}")
    else:
        print("⚠ Phase masks not saved (model.layers[*].phase not found).")

    # ===== 输出 4：prediction_viz（空间 label 对比）=====
    diag_dir = RUN_ROOT / "prediction_viz_spatial" / f"L{num_layer}_{run_tag}"
    n_vis = min(num_pred_diag_samples, len(test_ds))
    diag_paths = save_prediction_diagnostics_multiwl_spatial(
        model,
        test_ds,
        wavelengths=wavelengths,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius_eval,
        sample_indices=list(range(n_vis)),
        out_dir=diag_dir,
        device=device,
        tag=f"main_L{num_layer}",
    )
    if diag_paths:
        print(f"✔ Saved prediction diagnostics ({len(diag_paths)} samples) -> {diag_paths[0].parent}")

    # ===== 输出 5：superposition 可视化（同一个函数，选 2 个样本即可）=====
    if evaluation_mode == "superposition":
        super_dir = RUN_ROOT / "results_superposition_spatial" / f"L{num_layer}_{run_tag}"
        n_sup = min(num_superposition_visual_samples, len(test_ds))
        sup_paths = save_prediction_diagnostics_multiwl_spatial(
            model,
            test_ds,
            wavelengths=wavelengths,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius_eval,
            sample_indices=list(range(n_sup)),
            out_dir=super_dir,
            device=device,
            tag=f"superposition_L{num_layer}",
        )
        if sup_paths:
            print(f"✔ Saved superposition visuals ({len(sup_paths)}) -> {sup_paths[0].parent}")

    # ========================================================
    # EVAL: 每个波长用 evaluation_regions 算 legacy 风格指标
    # ========================================================
    for li in range(L):
        metrics = evaluate_spot_metrics_regions_multiwl(
            model,
            test_loader,
            device=device,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius_eval,
            wl_idx=li,
            L=L,
        )
        cc_mean = float(np.nanmean(metrics["cc_recon_amp"]))
        cc_std = float(np.nanstd(metrics["cc_recon_amp"]))

        print(
            f"[Region-metrics | {num_layer} layers | λ_idx={li} | λ={wavelengths[li]*1e9:.1f} nm] "
            f"avg_amp_error={metrics['avg_amplitudes_diff']:.6f}, "
            f"avg_relative_amp_error={metrics['avg_relative_amp_err']:.6f}, "
            f"cc_amp_mean±std={cc_mean:.6f}±{cc_std:.6f}"
        )

        metrics_by_wl[int(li)].append({"num_layers": int(num_layer), **metrics})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("All done.")


# ============================================================
# Metrics vs. layer count (按波长分别输出) + MAT/NPZ
# ============================================================
metrics_dir = RUN_ROOT / "metrics_analysis_regions"
metrics_dir.mkdir(parents=True, exist_ok=True)
tag = datetime.now().strftime("%Y%m%d_%H%M%S")

for li in range(L):
    mlist = metrics_by_wl.get(int(li), [])
    if not mlist:
        print(f"No metrics for λ_idx={li}, skip plotting/saving.")
        continue

    layer_counts = np.asarray([m["num_layers"] for m in mlist], dtype=np.int32)
    amp_err = np.asarray([m["avg_amplitudes_diff"] for m in mlist], dtype=np.float64)
    amp_err_rel = np.asarray([m["avg_relative_amp_err"] for m in mlist], dtype=np.float64)

    cc_amp_mean = np.asarray([float(np.nanmean(m["cc_recon_amp"])) for m in mlist], dtype=np.float64)
    cc_amp_std = np.asarray([float(np.nanstd(m["cc_recon_amp"])) for m in mlist], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(layer_counts, amp_err, marker="o")
    axes[0].set_ylabel("avg_amp_error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layer_counts, amp_err_rel, marker="o", color="tab:orange")
    axes[1].set_ylabel("avg_relative_amp_error")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(layer_counts, cc_amp_mean, yerr=cc_amp_std, marker="o", capsize=4, color="tab:green")
    axes[2].set_ylabel("cc_amp (mean±std)")
    axes[2].set_xlabel("num_layers")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.0, 1.01)

    fig.suptitle(f"Region metrics vs num_layers | λ_idx={li} | λ={wavelengths[li]*1e9:.1f} nm")
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    fig_path = metrics_dir / f"metrics_vs_layers_regions_wlidx{li}_{tag}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Metrics plot saved -> {fig_path}")

    mat_path = metrics_dir / f"metrics_regions_wlidx{li}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "layer_counts": layer_counts,
            "avg_amp_error": amp_err,
            "avg_relative_amp_error": amp_err_rel,
            "cc_amp_mean": cc_amp_mean,
            "cc_amp_std": cc_amp_std,
            "cc_amp_all": np.array([m["cc_recon_amp"] for m in mlist], dtype=object),
            "all_losses": np.array(all_losses, dtype=object),
            "wl_idx": np.array([li], dtype=np.int32),
            "wl_nm": np.array([wavelengths[li] * 1e9], dtype=np.float32),
            "wavelengths": wavelengths.astype(np.float32),
            "evaluation_mode": np.array([evaluation_mode], dtype=object),
            "training_dataset_mode": np.array([training_dataset_mode], dtype=object),
            "detect_radius_eval": np.array([detect_radius_eval], dtype=np.int32),
            "label_type": np.array(["spatial_label_like_legacy"], dtype=object),
        },
    )
    print(f"✔ Metrics MAT saved -> {mat_path}")

    npz_path = metrics_dir / f"metrics_regions_wlidx{li}_{tag}.npz"
    np.savez(
        str(npz_path),
        layer_counts=layer_counts,
        avg_amp_error=amp_err,
        avg_relative_amp_error=amp_err_rel,
        cc_amp_mean=cc_amp_mean,
        cc_amp_std=cc_amp_std,
        cc_amp_all=np.array([m["cc_recon_amp"] for m in mlist], dtype=object),
        all_losses=np.array(all_losses, dtype=object),
        wl_idx=np.array([li], dtype=np.int32),
        wl_nm=np.array([wavelengths[li] * 1e9], dtype=np.float32),
        wavelengths=wavelengths.astype(np.float32),
        evaluation_mode=np.array([evaluation_mode], dtype=object),
        training_dataset_mode=np.array([training_dataset_mode], dtype=object),
        detect_radius_eval=np.array([detect_radius_eval], dtype=np.int32),
        label_type=np.array(["spatial_label_like_legacy"], dtype=object),
        allow_pickle=True,
    )
    print(f"✔ Metrics NPZ saved -> {npz_path}")
