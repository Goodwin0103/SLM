#%%
import math
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import time
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from SLM.SLM_MULTIWL.ODNN_functions import (
    create_evaluation_regions,
    generate_complex_weights,
    generate_fields_ts,
)
from SLM.SLM_MULTIWL.odnn_generate_label import (
    compute_label_centers,
    compose_labels_from_patterns,
    generate_detector_patterns,
)
from SLM.SLM_MULTIWL.odnn_io import load_complex_modes_from_mat
from odnn_processing import prepare_sample

# ✅ MultiWL model
from SLM.SLM_MULTIWL.odnn_multiwl_model import D2NNModelMultiWL

# ✅ Use mainfor6 evaluation utilities (same metrics!)
from SLM.SLM_MULTIWL.odnn_training_eval import (
    build_superposition_eval_context,
    format_metric_report,
)

# ✅ MultiWL per-wavelength utilities (you must have added these)
from SLM.SLM_MULTIWL.odnn_training_eval import (
    evaluate_spot_metrics_multiwl_each,
    save_prediction_diagnostics_multiwl_each,
)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ----------------------------
# Parameters (match mainfor6)
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

evaluation_mode = "superposition"  # "eigenmode" or "superposition"
num_superposition_eval_samples = 1000
num_superposition_visual_samples = 2
label_pattern_mode = "circle"
superposition_eval_seed = 20240116
show_detection_overlap_debug = True
detection_overlap_label_index = 0

training_dataset_mode = "eigenmode"  # "eigenmode" or "superposition"
num_superposition_train_samples = 100
superposition_train_seed = 20240115

num_layer_option = [2, 3, 4, 5, 6]

# propagation geometry (match mainfor6)
z_layers = 40e-6
pixel_size = 1e-6
z_prop = 120e-6
z_input_to_first = 40e-6

# ✅ MultiWL config
wavelengths = np.array([650e-9, 1568e-9, 1650e-9], dtype=np.float32)
base_wavelength_idx = 0
L = int(len(wavelengths))

# training hyperparams (match mainfor6)
epochs = 1000
lr = 1.99
padding_ratio = 0.5
phase_option = 4   # eigenmodes
pred_case = 1      # only amplitudes prediction (mainfor6)

# ----------------------------
# Output dirs
# ----------------------------
run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
root_dir = Path("results/650_1568_1650") / f"run_{run_tag}"
root_dir.mkdir(parents=True, exist_ok=True)

models_dir = root_dir / "models"
loss_dir = root_dir / "loss_curves"
metrics_dir = root_dir / "metrics"
viz_dir = root_dir / "prediction_viz"
debug_dir = root_dir / "detection_region_debug"
for d in [models_dir, loss_dir, metrics_dir, viz_dir, debug_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# JSON helper: robust serialization (supports complex/tensor/ndarray)
# ----------------------------
def to_jsonable(obj):
    """Recursively convert obj to JSON-serializable python types."""
    import numpy as _np
    import torch as _torch

    if obj is None:
        return None

    if isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.complexfloating,)):
        return {"real": float(obj.real), "imag": float(obj.imag)}

    if isinstance(obj, _torch.Tensor):
        return to_jsonable(obj.detach().cpu().numpy())

    if isinstance(obj, _np.ndarray):
        if _np.iscomplexobj(obj):
            return {
                "real": _np.asarray(obj.real, dtype=_np.float64).tolist(),
                "imag": _np.asarray(obj.imag, dtype=_np.float64).tolist(),
            }
        return obj.tolist()

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    return str(obj)


# save config snapshot
cfg_path = root_dir / "config.json"
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "SEED": SEED,
            "field_size": field_size,
            "layer_size": layer_size,
            "num_modes": num_modes,
            "batch_size": batch_size,
            "evaluation_mode": evaluation_mode,
            "training_dataset_mode": training_dataset_mode,
            "num_superposition_eval_samples": num_superposition_eval_samples,
            "num_superposition_train_samples": num_superposition_train_samples,
            "wavelengths_m": [float(x) for x in wavelengths.tolist()],
            "epochs": epochs,
            "lr": lr,
            "padding_ratio": padding_ratio,
            "phase_option": phase_option,
            "pred_case": pred_case,
            "z_layers": float(z_layers),
            "z_prop": float(z_prop),
            "z_input_to_first": float(z_input_to_first),
            "pixel_size": float(pixel_size),
            "num_layer_option": [int(x) for x in num_layer_option],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
print("✔ Saved config ->", cfg_path)

# ----------------------------
# Helpers
# ----------------------------
def build_mode_context(base_modes: np.ndarray, num_modes_: int) -> dict:
    if base_modes.shape[2] < num_modes_:
        raise ValueError(f"Requested {num_modes_} modes, but file has {base_modes.shape[2]}.")

    mmf_data = base_modes[:, :, :num_modes_].transpose(2, 0, 1)  # (M,H,W)
    mmf_data_amp_norm = (np.abs(mmf_data) - np.min(np.abs(mmf_data))) / (
        np.max(np.abs(mmf_data)) - np.min(np.abs(mmf_data))
    )
    mmf_data = mmf_data_amp_norm * np.exp(1j * np.angle(mmf_data))

    if phase_option in [1, 2, 3, 5]:
        base_amplitudes_local, base_phases_local = generate_complex_weights(
            1000, num_modes_, phase_option
        )
    elif phase_option == 4:
        base_amplitudes_local = np.eye(num_modes_, dtype=np.float32)
        base_phases_local = np.eye(num_modes_, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported phase_option: {phase_option}")

    return {
        "mmf_data_np": mmf_data,
        "mmf_data_ts": torch.from_numpy(mmf_data),
        "base_amplitudes": base_amplitudes_local,
        "base_phases": base_phases_local,
    }


def save_loss_curve(losses_list: list[float], out_png: Path, out_mat: Path):
    xs = np.arange(1, len(losses_list) + 1, dtype=np.int32)
    ys = np.asarray(losses_list, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(xs, ys, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    savemat(str(out_mat), {"epoch": xs.astype(np.float64), "loss": ys})


def scalarize_metrics_for_table(m: dict) -> dict:
    # keep only scalars + summarize arrays; handle complex by magnitude
    out = {}
    for k, v in m.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if np.iscomplexobj(arr):
                arr = np.abs(arr)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size:
                out[k + "_mean"] = float(np.nanmean(arr))
                out[k + "_std"] = float(np.nanstd(arr))
            else:
                out[k + "_mean"] = float("nan")
                out[k + "_std"] = float("nan")
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if np.iscomplexobj(arr):
                arr = np.abs(arr)
            arr = arr.astype(np.float64, copy=False)
            if arr.size:
                out[k + "_mean"] = float(np.nanmean(arr))
                out[k + "_std"] = float(np.nanstd(arr))
            else:
                out[k + "_mean"] = float("nan")
                out[k + "_std"] = float("nan")
        else:
            pass
    return out


# ----------------------------
# Load eigenmodes
# ----------------------------
eigenmodes_OM4 = load_complex_modes_from_mat(
    "mmf_103modes_25_PD_1.15.mat",
    key="modes_field",
)
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]

# ----------------------------
# Build detector label patterns (match mainfor6)
# ----------------------------
label_size = layer_size
num_detector = num_modes

if label_pattern_mode == "eigenmode":
    pattern_stack = np.transpose(np.abs(MMF_data), (1, 2, 0))
    pattern_h, pattern_w, _ = pattern_stack.shape
    if pattern_h > label_size or pattern_w > label_size:
        raise ValueError(f"Pattern {pattern_h}x{pattern_w} exceeds label canvas {label_size}.")
    layout_radius = math.ceil(max(pattern_h, pattern_w) / 2)
    focus_radius = eigenmode_focus_radius
    detectsize = eigenmode_detectsize
elif label_pattern_mode == "circle":
    circle_radius = circle_focus_radius
    pattern_size = circle_radius * 2
    if pattern_size % 2 == 0:
        pattern_size += 1
    pattern_stack = generate_detector_patterns(pattern_size, pattern_size, num_detector, shape="circle")
    layout_radius = circle_radius
    focus_radius = circle_focus_radius
    detectsize = circle_detectsize
else:
    raise ValueError(f"Unknown label_pattern_mode: {label_pattern_mode}")

centers, _, _ = compute_label_centers(label_size, label_size, num_detector, layout_radius)
mode_label_maps = [
    compose_labels_from_patterns(
        label_size, label_size, pattern_stack, centers, Index=i + 1, visualize=False
    )
    for i in range(num_detector)
]
MMF_Label_data = torch.from_numpy(np.stack(mode_label_maps, axis=2).astype(np.float32))  # (H,W,M)

# ----------------------------
# Detection regions (match mainfor6)
# ----------------------------
evaluation_regions = create_evaluation_regions(layer_size, layer_size, num_detector, focus_radius, detectsize)
print("Detection Regions:", evaluation_regions)

if show_detection_overlap_debug:
    overlap_map = np.zeros((layer_size, layer_size), dtype=np.float32)
    for (x0, x1, y0, y1) in evaluation_regions:
        overlap_map[y0:y1, x0:x1] += 1.0
    overlap_pixels = int(np.count_nonzero(overlap_map > 1.0 + 1e-6))

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
        circ = Circle((cx, cy), radius=focus_radius, linewidth=1.0, edgecolor=color, linestyle="--", fill=False)
        axes[0].add_patch(circ)

    im1 = axes[1].imshow(overlap_map, cmap="viridis")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Detector coverage count")
    axes[1].set_axis_off()

    overlap_plot_path = debug_dir / f"detection_overlap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(overlap_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if overlap_pixels > 0:
        print(f"⚠ Overlap detected: {overlap_pixels} pixels have >1 coverage.")
    else:
        print("✔ No overlap detected.")
    print("✔ Saved overlap debug ->", overlap_plot_path)

# ----------------------------
# Build training dataset (match mainfor6)
# ----------------------------
if training_dataset_mode == "eigenmode":
    if phase_option == 4:
        num_train_samples = num_modes
        amplitudes = base_amplitudes[:num_train_samples]
        phases = base_phases[:num_train_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_train_samples = amplitudes.shape[0]

    amplitudes_phases = np.hstack((amplitudes, phases[:, 1:] / (2 * np.pi)))
    label_data = torch.zeros([num_train_samples, 1, layer_size, layer_size], dtype=torch.float32)

    amplitude_weights = torch.from_numpy(amplitudes_phases[:, 0:num_modes]).float()
    energy_weights = amplitude_weights ** 2
    combined_labels = (energy_weights[:, None, None, :] * MMF_Label_data.unsqueeze(0)).sum(dim=3)
    label_data[:, 0, :, :] = combined_labels

    complex_weights = amplitudes * np.exp(1j * phases)
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_train_samples, num_modes, field_size
    ).to(torch.complex64)

    train_dataset = [prepare_sample(image_data[i], label_data[i], layer_size) for i in range(num_train_samples)]
    train_tensor_data = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_dataset)])

elif training_dataset_mode == "superposition":
    num_train_samples = num_superposition_train_samples
    super_train_ctx = build_superposition_eval_context(
        num_train_samples,
        num_modes=num_modes,
        field_size=field_size,
        layer_size=layer_size,
        mmf_modes=MMF_data_ts,
        mmf_label_data=MMF_Label_data,
        batch_size=batch_size,
        second_mode_half_range=True,
        rng_seed=superposition_train_seed,
    )
    train_tensor_data = super_train_ctx["tensor_dataset"]
    amplitudes = super_train_ctx["amplitudes"]
    phases = super_train_ctx["phases"]
    amplitudes_phases = super_train_ctx["amplitudes_phases"]
    image_data = super_train_ctx["image_data"]
    label_data = train_tensor_data.tensors[1]
else:
    raise ValueError(f"Unknown training_dataset_mode: {training_dataset_mode}")

# ----------------------------
# Build test dataset (match mainfor6)
# ----------------------------
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_tensor_data, batch_size=batch_size, shuffle=True, generator=g)

if evaluation_mode == "eigenmode":
    test_tensor_data = train_tensor_data
    test_loader = DataLoader(test_tensor_data, batch_size=batch_size, shuffle=False)
    test_dataset = train_dataset if "train_dataset" in locals() else None

    eval_amplitudes = amplitudes
    eval_amplitudes_phases = amplitudes_phases
    eval_phases = phases
    image_test_data = image_data

elif evaluation_mode == "superposition":
    super_ctx = build_superposition_eval_context(
        num_superposition_eval_samples,
        num_modes=num_modes,
        field_size=field_size,
        layer_size=layer_size,
        mmf_modes=MMF_data_ts,
        mmf_label_data=MMF_Label_data,
        batch_size=batch_size,
        second_mode_half_range=True,
        rng_seed=superposition_eval_seed,
    )
    test_dataset = super_ctx["dataset"]
    test_tensor_data = super_ctx["tensor_dataset"]
    test_loader = super_ctx["loader"]
    image_test_data = super_ctx["image_data"]

    eval_amplitudes = super_ctx["amplitudes"]
    eval_amplitudes_phases = super_ctx["amplitudes_phases"]
    eval_phases = super_ctx["phases"]
else:
    raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

# ----------------------------
# Train & evaluate for each layer count
# ----------------------------
model_metrics_rows: list[dict] = []

for num_layer in num_layer_option:
    print(f"\n{'='*60}\nTraining MultiWL model with {num_layer} layers\n{'='*60}")

    model_multi = D2NNModelMultiWL(
        num_layers=num_layer,
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelengths=wavelengths,
        device=device,
        padding_ratio=padding_ratio,
        z_input_to_first=float(z_input_to_first),
        base_wavelength_idx=int(base_wavelength_idx),
    ).to(device)

    optimizer = optim.Adam(model_multi.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.MSELoss()

    losses: list[float] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model_multi.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)  # (B,1,H,W) complex
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)   # (B,1,H,W) float

            optimizer.zero_grad(set_to_none=True)

            # MultiWL forward needs (B,L,H,W): repeat input across wavelengths
            x = images.repeat(1, L, 1, 1).contiguous()
            out_I = model_multi(x)  # (B,L,H,W) float intensity

            # label repeat to (B,L,H,W)
            y = labels.repeat(1, L, 1, 1).contiguous()

            loss = criterion(out_I, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        losses.append(avg_loss)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] Loss={avg_loss:.12f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = time.time() - t0
    print(f"Total training time: {total_time:.2f}s (~{total_time/60:.2f}min)")

    # ---- Save model checkpoint ----
    ckpt_path = models_dir / f"model_layers{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(
        {
            "state_dict": model_multi.state_dict(),
            "num_layers": int(num_layer),
            "wavelengths_m": wavelengths.astype(np.float64),
            "layer_size": int(layer_size),
            "padding_ratio": float(padding_ratio),
            "z_layers": float(z_layers),
            "z_prop": float(z_prop),
            "z_input_to_first": float(z_input_to_first),
            "pixel_size": float(pixel_size),
            "seed": int(SEED),
        },
        ckpt_path,
    )
    print("✔ Saved model ->", ckpt_path)

    # ---- Save loss curve (png+mat) ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_png = loss_dir / f"loss_layers{num_layer}_{ts}.png"
    loss_mat = loss_dir / f"loss_layers{num_layer}_{ts}.mat"
    save_loss_curve(losses, loss_png, loss_mat)
    print("✔ Saved loss curve ->", loss_png)

    # ---- Evaluate: metrics per wavelength ----
    metrics_by_wl = evaluate_spot_metrics_multiwl_each(
        model_multi,
        test_loader,
        evaluation_regions,
        detect_radius=detectsize,
        device=device,
        pred_case=pred_case,
        num_modes=num_modes,
        phase_option=phase_option,
        amplitudes=eval_amplitudes,
        amplitudes_phases=eval_amplitudes_phases,
        phases=eval_phases,
        mmf_modes=MMF_data_ts,
        field_size=field_size,
        image_test_data=image_test_data,
    )

    # print + collect rows + save per-layer metrics (mat + json)
    per_layer_dump = {
        "num_layers": int(num_layer),
        "wavelengths_m": wavelengths.astype(np.float64),
        "metrics_by_wl": {},
    }

    for wl_idx, m in metrics_by_wl.items():
        wl_nm = float(m["wavelength_m"] * 1e9)
        print(
            format_metric_report(
                num_modes=num_modes,
                phase_option=phase_option,
                pred_case=pred_case,
                label=f"MultiWL | {num_layer} layers | wl_idx={wl_idx} ({wl_nm:.1f} nm)",
                metrics=m,
            )
        )

        row = {
            "num_layers": int(num_layer),
            "wl_idx": int(wl_idx),
            "wavelength_m": float(m.get("wavelength_m", np.nan)),
            "wavelength_nm": float(wl_nm),
        }
        row.update(scalarize_metrics_for_table(m))
        model_metrics_rows.append(row)

        per_layer_dump["metrics_by_wl"][int(wl_idx)] = m

    # save per-layer metrics json (robust: supports complex)
    per_layer_json = metrics_dir / f"metrics_layers{num_layer}_{ts}.json"
    with open(per_layer_json, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(per_layer_dump), f, ensure_ascii=False, indent=2)
    print("✔ Saved per-layer metrics ->", per_layer_json)

    # also save mat (best for arrays)
    per_layer_mat = metrics_dir / f"metrics_layers{num_layer}_{ts}.mat"
    mat_dict = {"num_layers": float(num_layer), "wavelengths_m": wavelengths.astype(np.float64)}
    for wl_idx, m in metrics_by_wl.items():
        prefix = f"wl{int(wl_idx)}_"
        for k, v in m.items():
            if k == "wavelength_m":
                mat_dict[prefix + k] = float(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                mat_dict[prefix + k] = float(v)
            else:
                # handle complex explicitly: save real/imag
                try:
                    arr = np.asarray(v)
                    if np.iscomplexobj(arr):
                        mat_dict[prefix + k + "_real"] = np.asarray(arr.real, dtype=np.float64)
                        mat_dict[prefix + k + "_imag"] = np.asarray(arr.imag, dtype=np.float64)
                    else:
                        mat_dict[prefix + k] = np.asarray(arr, dtype=np.float64)
                except Exception:
                    pass
    savemat(str(per_layer_mat), mat_dict)
    print("✔ Saved per-layer metrics mat ->", per_layer_mat)

    # ---- Save qualitative diagnostics: per wavelength ----
    diag_dir = viz_dir / f"L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_by_wl = save_prediction_diagnostics_multiwl_each(
        model_multi,
        test_dataset,
        evaluation_regions=evaluation_regions,
        layer_size=layer_size,
        detect_radius=detectsize,
        num_samples=num_superposition_visual_samples if evaluation_mode == "superposition" else min(5, num_modes),
        output_dir=diag_dir,
        device=device,
        tag=f"aligned_multiwl_L{num_layer}",
    )
    for wl_idx, paths in saved_by_wl.items():
        if paths:
            print(f"✔ Saved diagnostics wl_idx={wl_idx} -> {paths[0].parent}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nDone training & per-wavelength evaluation.")

# ----------------------------
# Save aggregated metrics table + Plot metrics vs layer per wavelength
# ----------------------------
if model_metrics_rows:
    # save table json
    table_json = metrics_dir / f"metrics_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(model_metrics_rows), f, ensure_ascii=False, indent=2)
    print("✔ Saved aggregated metrics table ->", table_json)

    # save table mat
    wl_idx_arr = np.asarray([r["wl_idx"] for r in model_metrics_rows], dtype=np.int32)
    layers_arr = np.asarray([r["num_layers"] for r in model_metrics_rows], dtype=np.int32)
    wl_nm_arr = np.asarray([r.get("wavelength_nm", np.nan) for r in model_metrics_rows], dtype=np.float64)
    avg_rel_arr = np.asarray([r.get("avg_relative_amp_err", np.nan) for r in model_metrics_rows], dtype=np.float64)
    avg_abs_arr = np.asarray([r.get("avg_amplitudes_diff", np.nan) for r in model_metrics_rows], dtype=np.float64)
    cc_amp_mean_arr = np.asarray([r.get("cc_recon_amp_mean", np.nan) for r in model_metrics_rows], dtype=np.float64)
    cc_amp_std_arr = np.asarray([r.get("cc_recon_amp_std", np.nan) for r in model_metrics_rows], dtype=np.float64)

    table_mat = metrics_dir / f"metrics_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mat"
    savemat(
        str(table_mat),
        {
            "wl_idx": wl_idx_arr.astype(np.float64),
            "num_layers": layers_arr.astype(np.float64),
            "wavelength_nm": wl_nm_arr,
            "avg_relative_amp_err": avg_rel_arr,
            "avg_amplitudes_diff": avg_abs_arr,
            "cc_recon_amp_mean": cc_amp_mean_arr,
            "cc_recon_amp_std": cc_amp_std_arr,
        },
    )
    print("✔ Saved aggregated metrics mat ->", table_mat)

    # plot: avg_relative_amp_err vs layers per wavelength
    wl_indices = sorted({int(r["wl_idx"]) for r in model_metrics_rows})
    layer_counts = sorted({int(r["num_layers"]) for r in model_metrics_rows})

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for wl_idx in wl_indices:
        xs, ys = [], []
        wl_nm = None
        for Lc in layer_counts:
            recs = [r for r in model_metrics_rows if int(r["wl_idx"]) == wl_idx and int(r["num_layers"]) == Lc]
            if not recs:
                continue
            r0 = recs[0]
            wl_nm = r0.get("wavelength_nm", None)
            xs.append(Lc)
            ys.append(r0.get("avg_relative_amp_err", np.nan))
        label = f"wl_idx={wl_idx}" + (f" ({wl_nm:.1f} nm)" if wl_nm is not None else "")
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)

    ax.set_xlabel("Number of layers")
    ax.set_ylabel("avg_relative_amp_error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = metrics_dir / f"metrics_vs_layers_by_wavelength_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✔ Saved metrics plot ->", out_png)
else:
    print("No model_metrics_rows collected; skipping final plots.")
#%%
