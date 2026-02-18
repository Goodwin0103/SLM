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

# 你的 MultiWL 模型
from odnn_multiwl_model import D2NNModelMultiWL

# ROI masks + superposition sampler
from odnn_training_eval import build_circular_roi_masks, build_superposition_eval_context


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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using Device:", device)
else:
    device = torch.device("cpu")
    print("Using Device: CPU")


# ============================================================
# Parameters
# ============================================================
field_size = 25
layer_size = 110
num_modes = 5

circle_focus_radius = 5
circle_detectsize = 10
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

# propagation params
z_layers = 40e-6
pixel_size = 1e-6
z_prop = 120e-6
z_input_to_first = 40e-6

# wavelengths (MultiWL)
wavelengths = np.array([1550e-9], dtype=np.float32)
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


# ============================================================
# Utils: legacy-like metrics for MultiWL (ROI based)
# ============================================================
def _safe_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.sum(dim=-1, keepdim=True) + eps)

def _per_sample_corrcoef(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.sqrt((a0 * a0).sum() + eps) * np.sqrt((b0 * b0).sum() + eps))
    return float((a0 * b0).sum() / denom)

def intensity_to_roi_energies(I_blhw: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
    """
    I_blhw: (B,L,H,W) float
    roi_masks: (M,H,W) float
    return: (B,L,M) energies
    """
    I_blhw = I_blhw.to(torch.float32)
    roi_masks = roi_masks.to(torch.float32)
    return (I_blhw.unsqueeze(2) * roi_masks.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2))


@torch.no_grad()
def evaluate_spot_metrics_like_legacy_for_multiwl(
    model: D2NNModelMultiWL,
    loader: DataLoader,
    *,
    device: torch.device,
    roi_masks: torch.Tensor,          # (M,H,W) float
    base_wavelength_idx: int,
    eval_amplitudes: np.ndarray,      # (N,M) numpy, aligned with loader order
    L: int,
) -> dict:
    """
    输出字段对齐旧版绘图所需：
      - avg_amplitudes_diff
      - avg_relative_amp_err
      - cc_recon_amp  (per-sample corr, shape (N,))
    说明：
      true 使用 amplitude -> energy -> energy_frac -> amp_frac=sqrt(energy_frac)
      pred 使用 ROI energy -> energy_frac -> amp_frac=sqrt(energy_frac)
    """
    model.eval()
    roi_masks = roi_masks.to(device=device, dtype=torch.float32)

    # True: from amplitudes
    true_amp = torch.from_numpy(eval_amplitudes.astype(np.float32)).to(device)  # (N,M)
    true_energy = true_amp ** 2
    true_energy_frac = _safe_norm(true_energy)                 # (N,M)
    true_amp_frac = torch.sqrt(true_energy_frac + 1e-12)        # (N,M)

    pred_amp_frac_list = []
    for images, _y in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)  # (B,1,H,W)

        x = images.repeat(1, L, 1, 1).contiguous()              # (B,L,H,W)
        I_blhw = model(x)                                       # (B,L,H,W)
        I_bhw = I_blhw[:, base_wavelength_idx]                  # (B,H,W)

        # ROI energy -> frac
        E_bm = (I_bhw.unsqueeze(1) * roi_masks.unsqueeze(0)).sum(dim=(-1, -2))  # (B,M)
        pred_energy_frac = _safe_norm(E_bm)                       # (B,M)
        pred_amp_frac = torch.sqrt(pred_energy_frac + 1e-12)      # (B,M)

        pred_amp_frac_list.append(pred_amp_frac.detach().cpu())

    pred_amp_frac_all = torch.cat(pred_amp_frac_list, dim=0).to(device)         # (N,M)

    # diffs
    diff = pred_amp_frac_all - true_amp_frac
    abs_diff = diff.abs()
    rel = abs_diff / (true_amp_frac.abs() + 1e-12)

    avg_amp_diff = float(abs_diff.mean().item())
    avg_rel = float(rel.mean().item())

    # per-sample corrcoef
    pa = pred_amp_frac_all.detach().cpu().numpy()
    ta = true_amp_frac.detach().cpu().numpy()
    cc = np.asarray([_per_sample_corrcoef(pa[i], ta[i]) for i in range(pa.shape[0])], dtype=np.float64)

    return {
        "avg_amplitudes_diff": avg_amp_diff,
        "avg_relative_amp_err": avg_rel,
        "cc_recon_amp": cc,
        # 兼容旧变量名（你旧脚本里也会用到这些 list）
        "amplitudes_diff": diff.detach().cpu().numpy(),
    }


# ============================================================
# Label propagation for y_vec (训练仍然用 ROI-ratio)
#   ——保持你当前 MultiWL 脚本逻辑：每个 num_layer 重算标签
# ============================================================
def _complex_pad_mainstyle(E: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    Er = torch.view_as_real(E)
    Er_pad = F.pad(Er, (0, 0, pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
    return torch.view_as_complex(Er_pad.contiguous())

def _complex_crop_mainstyle(E_pad: torch.Tensor, H: int, W: int, pad_h: int, pad_w: int) -> torch.Tensor:
    return E_pad[..., pad_h:pad_h + H, pad_w:pad_w + W].contiguous()

class _PropagationMultiWLMainstyle(torch.nn.Module):
    def __init__(self, units: int, dx: float, wavelengths: np.ndarray, z: float, device: torch.device, pad_px: int = 0):
        super().__init__()
        self.units = int(units)
        self.dx = float(dx)
        self.z = float(z)
        self.pad_px = int(pad_px)

        wl = torch.tensor(np.asarray(wavelengths, dtype=np.float32), dtype=torch.float32, device=device)
        self.register_buffer("wavelengths", wl)

        self.register_buffer("kz_base", self._make_kz_stack(self.units, self.dx, wl, device))
        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz_stack(units_pad, self.dx, wl, device))
        else:
            self.kz_pad = None

    @staticmethod
    def _make_kz_stack(N: int, dx: float, wavelengths_ts: torch.Tensor, device: torch.device) -> torch.Tensor:
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")

        inv_lam2 = (1.0 / wavelengths_ts)[:, None, None] ** 2
        argument = (2 * torch.pi) ** 2 * (inv_lam2 - fxx[None] ** 2 - fyy[None] ** 2)

        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
        return kz

    @staticmethod
    def _propagate(E: torch.Tensor, kz: torch.Tensor, z: float) -> torch.Tensor:
        C = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
        return torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz[None] * z), dim=(-2, -1)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.is_complex()
        B, L_in, H, W = inputs.shape
        if L_in != int(self.wavelengths.numel()):
            raise ValueError("Input L mismatch wavelengths.")
        if self.pad_px > 0:
            p = self.pad_px
            Ein = _complex_pad_mainstyle(inputs, p, p)
            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return _complex_crop_mainstyle(Eout, H, W, p, p)
        return self._propagate(inputs, self.kz_base, self.z)

@torch.no_grad()
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
    不相干叠加：energy_weights = amplitudes^2
    返回 y_vec: (N, L, M_roi)  (ROI 能量比例)
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    wls = np.asarray(wls, dtype=np.float32).reshape(-1)

    N, M = amplitudes.shape
    L_wl = int(wls.shape[0])
    M_roi = int(masks.shape[0])

    pad_px = int(round(ls * float(pad_ratio)))
    prop = _PropagationMultiWLMainstyle(
        units=ls, dx=float(px), wavelengths=wls, z=float(z_tot), device=dev, pad_px=pad_px
    ).to(dev)

    E_ref = torch.zeros((M, L_wl, M_roi), dtype=torch.float32, device=dev)

    print(f"  Computing per-wavelength labels (z_total={z_tot*1e6:.1f} μm) ...")
    for m in range(M):
        mode_field = mmf_modes[m].to(dev, dtype=torch.complex64)  # (Hf,Wf)
        Hf, Wf = mode_field.shape
        if (Hf != ls) or (Wf != ls):
            canvas = torch.zeros((ls, ls), dtype=torch.complex64, device=dev)
            y0 = (ls - Hf) // 2
            x0 = (ls - Wf) // 2
            canvas[y0:y0 + Hf, x0:x0 + Wf] = mode_field
            mode_field = canvas

        Ein = mode_field[None, None].repeat(1, L_wl, 1, 1).contiguous()
        Eout = prop(Ein)                  # (1,L,H,W)
        Iout = torch.abs(Eout) ** 2       # (1,L,H,W)

        Ek = (Iout[0].unsqueeze(1) * masks.unsqueeze(0)).sum(dim=(-1, -2))  # (L,K)
        E_ref[m] = Ek

    amp_sq = torch.from_numpy(amplitudes ** 2).to(dev, dtype=torch.float32)  # (N,M)
    y_energy = torch.einsum("nm, mlk -> nlk", amp_sq, E_ref)                 # (N,L,K)
    y_vec = y_energy / (y_energy.sum(dim=2, keepdim=True) + 1e-12)
    return y_vec.cpu()


# ============================================================
# Data / label helpers
# ============================================================
def build_mode_context(base_modes: np.ndarray, num_modes: int) -> dict:
    if base_modes.shape[2] < num_modes:
        raise ValueError("Requested modes exceed file modes.")
    mmf_data = base_modes[:, :, :num_modes].transpose(2, 0, 1)

    mmf_data_amp_norm = (np.abs(mmf_data) - np.min(np.abs(mmf_data))) / (np.max(np.abs(mmf_data)) - np.min(np.abs(mmf_data)) + 1e-12)
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
# Build detector layout / evaluation regions (for debug only)
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
MMF_Label_data = torch.from_numpy(np.stack(mode_label_maps, axis=2).astype(np.float32))

evaluation_regions = create_evaluation_regions(layer_size, layer_size, num_detector, focus_radius, detectsize)
print("Detection Regions:", evaluation_regions)

if show_detection_overlap_debug:
    detection_debug_dir = Path("results/1550/detection_region_debug")
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

    for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        color = plt.cm.tab20(idx_region % 20)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=color, facecolor="none")
        axes[0].add_patch(rect)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        axes[0].add_patch(Circle((cx, cy), radius=focus_radius, linewidth=1.0, edgecolor=color, linestyle="--", fill=False))

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
# ROI masks for regression
# ============================================================
roi_stack, _ = build_circular_roi_masks(
    height=layer_size,
    width=layer_size,
    num_spots=num_modes,
    focus_radius=int(focus_radius),
    radius_scale=1.0,
)
roi_masks = torch.tensor(roi_stack, dtype=torch.float32, device=device)
print("roi_masks:", tuple(roi_masks.shape))


# ============================================================
# Dataset builders (保持你现在逻辑：每个 num_layer 重建 y_vec)
# ============================================================
def build_eigenmode_dataset(num_layers_current: int) -> tuple[TensorDataset, dict]:
    if phase_option == 4:
        num_samples = num_modes
        amplitudes = base_amplitudes[:num_samples]
        phases = base_phases[:num_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_samples = amplitudes.shape[0]

    z_total = z_input_to_first + (num_layers_current - 1) * z_layers + z_prop
    print(f"  [eigenmode] z_total for {num_layers_current} layers: {z_total*1e6:.1f} μm")

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
    image_data = generate_fields_ts(complex_weights_ts, MMF_data_ts, num_samples, num_modes, field_size).to(torch.complex64)

    dummy_label = torch.zeros([1, layer_size, layer_size], dtype=torch.float32)
    images_prepared = []
    for i in range(num_samples):
        img_i, _ = prepare_sample(image_data[i], dummy_label, layer_size)
        images_prepared.append(img_i)
    image_tensor = torch.stack(images_prepared, dim=0)

    ds = TensorDataset(image_tensor, y_vec)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, meta


def build_superposition_dataset(num_samples: int, rng_seed: int, num_layers_current: int) -> tuple[TensorDataset, dict]:
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
    images = tensor_dataset.tensors[0]  # (N,1,H,W) complex
    amplitudes = ctx["amplitudes"]
    phases = ctx["phases"]

    z_total = z_input_to_first + (num_layers_current - 1) * z_layers + z_prop
    print(f"  [superposition] z_total for {num_layers_current} layers: {z_total*1e6:.1f} μm")

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
    )

    ds = TensorDataset(images, y_vec)
    meta = {"amplitudes": amplitudes, "phases": phases}
    return ds, meta


# ============================================================
# Train/Eval loop
# ============================================================
all_losses = []
all_average_amplitudes_diff = []
all_amplitudes_relative_diff = []
all_cc_recon_amp = []
model_metrics = []

for num_layer in num_layer_option:
    print(f"\n{'='*70}\nTraining D2NNModelMultiWL with {num_layer} layers\n{'='*70}")

    # datasets
    if training_dataset_mode == "eigenmode":
        train_ds, train_meta = build_eigenmode_dataset(num_layers_current=num_layer)
    elif training_dataset_mode == "superposition":
        train_ds, train_meta = build_superposition_dataset(num_superposition_train_samples, superposition_train_seed, num_layer)
    else:
        raise ValueError("Unknown training_dataset_mode")

    if evaluation_mode == "eigenmode":
        test_ds, test_meta = build_eigenmode_dataset(num_layers_current=num_layer)
    elif evaluation_mode == "superposition":
        test_ds, test_meta = build_superposition_dataset(num_superposition_eval_samples, superposition_eval_seed, num_layer)
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

    # train (loss仍然用 ROI ratio)
    losses = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, y in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            if images.ndim == 3:
                images = images.unsqueeze(1)

            x = images.repeat(1, L, 1, 1).contiguous()  # (B,L,H,W)
            optimizer.zero_grad(set_to_none=True)

            I_blhw = model(x)  # (B,L,H,W) intensity
            pred_energy = intensity_to_roi_energies(I_blhw, roi_masks)     # (B,L,M)
            pred_ratio = pred_energy / (pred_energy.sum(dim=2, keepdim=True) + 1e-12)

            loss = F.mse_loss(pred_ratio, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        losses.append(avg_loss)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] loss={avg_loss:.10f}")

    total_time = time.time() - t0
    all_losses.append(losses)
    print(f"Training done: {num_layer} layers, time={total_time:.2f}s")

    # ========================================================
    # EVAL: 改成旧版口径指标（avg_amp_error / avg_relative_amp_error / cc_amp）
    # ========================================================
    eval_amplitudes = test_meta["amplitudes"]  # numpy (N,M) aligned with test_ds order

    legacy_metrics = evaluate_spot_metrics_like_legacy_for_multiwl(
        model,
        test_loader,
        device=device,
        roi_masks=roi_masks,
        base_wavelength_idx=base_wavelength_idx,
        eval_amplitudes=eval_amplitudes,
        L=L,
    )

    cc_mean = float(np.nanmean(legacy_metrics["cc_recon_amp"]))
    cc_std = float(np.nanstd(legacy_metrics["cc_recon_amp"]))

    print(
        f"[Legacy-like metrics | {num_layer} layers | λ_idx={base_wavelength_idx}] "
        f"avg_amp_error={legacy_metrics['avg_amplitudes_diff']:.6f}, "
        f"avg_relative_amp_error={legacy_metrics['avg_relative_amp_err']:.6f}, "
        f"cc_amp_mean±std={cc_mean:.6f}±{cc_std:.6f}"
    )

    model_metrics.append({"num_layers": int(num_layer), **legacy_metrics})
    all_average_amplitudes_diff.append(float(legacy_metrics["avg_amplitudes_diff"]))
    all_amplitudes_relative_diff.append(float(legacy_metrics["avg_relative_amp_err"]))
    all_cc_recon_amp.append(legacy_metrics["cc_recon_amp"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("All done.")


# ============================================================
# Metrics vs. layer count (旧版风格)
# ============================================================
if model_metrics:
    metrics_dir = Path("results/1550/metrics_analysis_legacy_like")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    layer_counts = np.asarray([m["num_layers"] for m in model_metrics], dtype=np.int32)
    amp_err = np.asarray(all_average_amplitudes_diff, dtype=np.float64)
    amp_err_rel = np.asarray(all_amplitudes_relative_diff, dtype=np.float64)

    cc_amp_mean_list = []
    cc_amp_std_list = []
    for cc_arr in all_cc_recon_amp:
        cc_amp_mean_list.append(float(np.nanmean(cc_arr)))
        cc_amp_std_list.append(float(np.nanstd(cc_arr)))
    cc_amp_mean = np.asarray(cc_amp_mean_list, dtype=np.float64)
    cc_amp_std = np.asarray(cc_amp_std_list, dtype=np.float64)

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

    fig.suptitle(f"Legacy-like metrics vs num_layers (λ_idx={base_wavelength_idx})")
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    fig_path = metrics_dir / f"metrics_vs_layers_legacy_like_{tag}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Metrics plot saved -> {fig_path}")

    # ----------------------------
    # Save metrics to MAT/NPZ
    # ----------------------------
    # 1) MAT: 方便你延续旧 Matlab/脚本读取习惯
    mat_path = metrics_dir / f"metrics_legacy_like_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "layer_counts": layer_counts,
            "avg_amp_error": amp_err,
            "avg_relative_amp_error": amp_err_rel,
            "cc_amp_mean": cc_amp_mean,
            "cc_amp_std": cc_amp_std,
            # 变长列表（每个 num_layer 一组 N 样本的 cc）：用 object array 存
            "cc_amp_all": np.array(all_cc_recon_amp, dtype=object),
            "all_losses": np.array(all_losses, dtype=object),
            "base_wavelength_idx": np.array([base_wavelength_idx], dtype=np.int32),
            "wavelengths": wavelengths.astype(np.float32),
            "evaluation_mode": np.array([evaluation_mode], dtype=object),
            "training_dataset_mode": np.array([training_dataset_mode], dtype=object),
        },
    )
    print(f"✔ Metrics MAT saved -> {mat_path}")

    # 2) NPZ: Python 里更好读
    npz_path = metrics_dir / f"metrics_legacy_like_{tag}.npz"
    np.savez(
        str(npz_path),
        layer_counts=layer_counts,
        avg_amp_error=amp_err,
        avg_relative_amp_error=amp_err_rel,
        cc_amp_mean=cc_amp_mean,
        cc_amp_std=cc_amp_std,
        cc_amp_all=np.array(all_cc_recon_amp, dtype=object),
        all_losses=np.array(all_losses, dtype=object),
        base_wavelength_idx=np.array([base_wavelength_idx], dtype=np.int32),
        wavelengths=wavelengths.astype(np.float32),
        evaluation_mode=np.array([evaluation_mode], dtype=object),
        training_dataset_mode=np.array([training_dataset_mode], dtype=object),
        allow_pickle=True,
    )
    print(f"✔ Metrics NPZ saved -> {npz_path}")

else:
    print("No model_metrics collected; skip plotting/saving.")
