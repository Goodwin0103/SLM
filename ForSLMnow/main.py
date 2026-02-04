#%%
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
from odnn_model import D2NNModel
from odnn_processing import prepare_sample
from odnn_training_eval import (
    build_superposition_eval_context,

    compute_amp_relative_error_with_shift,
    compute_model_prediction_metrics,
    evaluate_spot_metrics,
    format_metric_report,
    generate_superposition_sample,
    infer_superposition_output,
    save_prediction_diagnostics,
    spot_energy_ratios_circle,
)
from odnn_training_io import save_masks_one_file_per_layer, save_to_mat_light_plus
from odnn_training_visualization import (
    capture_eigenmode_propagation,
    export_superposition_slices,
    plot_amplitude_comparison_grid,
    plot_reconstruction_vs_input,
    plot_sys_vs_label_strict,
    save_superposition_triptych,
    save_mode_triptych,
    visualize_model_slices,
)
from odnn_wavelength_analysis import (
    ModelGeometry,
    compute_relative_amp_error_wavelength_sweep,
    compute_mode_isolation_wavelength_sweep,
    scale_phase_masks_for_wavelength,
)

SEED = 424242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 让 cuDNN/算子走确定性分支
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    device = torch.device('cuda:5')           # 或者 'cuda:0'
    print('Using Device:', device)
else:
    device = torch.device('cpu')
    print('Using Device: CPU')


#%% data generation (lightfield)
field_size = 123 # the field size in eigenmodes_OM4 is 25 pixels
layer_size_options = [300]  # layer canvas size sweep
run_layer_size_sweep = False  # toggle to run a sweep before the legacy single run
layer_size = layer_size_options[0]
num_data = 1000 # options: 1. random datas 2.eigenmodes
num_modes = 3 #default: take first N modes from the 103-mode file
num_modes_sweep_options = [0]
circle_focus_radius = 20 # radius when using uniform circular detectors
circle_detectsize = 40  # durchmeter (/2)
eigenmode_focus_radius = 12.5  # radius when using eigenmode patterns
eigenmode_detectsize = 15    # square window size for eigenmode patterns
focus_radius = circle_focus_radius
detectsize = circle_detectsize
batch_size = 16

# Evaluation selection: "eigenmode" uses the base modes, "superposition" samples random mixtures
evaluation_mode = "superposition"  # options: "eigenmode", "superposition"
num_superposition_eval_samples = 1000 #评估是看1000个样本
num_superposition_visual_samples = 2  #选两个看看那个对比标签啥的样本
run_superposition_debug = True
save_superposition_plots = True
save_superposition_slices = True
run_misalignment_robustness = True
label_pattern_mode = "mixed"  # options: "eigenmode", "circle", "mixed"
# Use when label_pattern_mode == "mixed" to assign different shapes per detector.
detector_shapes = ["circle", "square", "diamond"]
superposition_eval_seed = 20240116   # 控制 superposition 测试集的随机性
show_detection_overlap_debug = True
detection_overlap_label_index = 0

# Debug/visualization for wavelength sweep
enable_wavelength_debug_visuals = False
debug_wavelengths_nm = [1500.0, 1568.0, 1650.0]
debug_modes_to_plot = [0, 1]  # 0-based indices

#看过程切片的参数设置
prop_slices_per_segment = 10   # 每段传播取样张数（层间/输出）
prop_output_slices = 10        # 输出面前的采样张数
prop_scan_kmax = 10            # visualize_model_slices 每段最多展示的帧数
prop_slice_sample_mode = "random"  # "fixed" 使用 FIXED_E_INDEX，下方可选随机样本
prop_slice_seed = 20251121         # 控制随机选样的种子



# Training data selection: 默认用 eigenmode，也可以改成 superposition 并设定样本数量
training_dataset_mode = "eigenmode"  # options: "eigenmode", "superposition"
num_superposition_train_samples = 100  # superposition 训练样本数
superposition_train_seed = 20240115  # 控制 superposition 训练集的随机性

# Define multiple D2NN models 
num_layer_option = [2]   # Define the different layer-number ODNN
all_losses = [] #the loss for each epoch of each ODNN model
all_phase_masks = [] #the phase masks field of each ODNN model
all_predictions = [] #the output light field of each ODNN model
model_metrics: list[dict] = []
all_amplitudes_diff: list[np.ndarray] = []
all_average_amplitudes_diff: list[float] = []
all_amplitudes_relative_diff: list[float] = []
all_complex_weights_pred: list[np.ndarray] = []
all_image_data_pred: list[np.ndarray] = []
all_cc_real: list[np.ndarray] = []
all_cc_imag: list[np.ndarray] = []
all_cc_recon_amp: list[np.ndarray] = []
all_cc_recon_phase: list[np.ndarray] = []
all_training_summaries: list[dict] = []

# SLM
z_layers   = 49.465e-3        # 原 47.571e-3  -> 40 μm
pixel_size = 12.5e-6
z_prop     = 20e-2        # 原 16.74e-2   -> 60 μm plus 40（最后一层到相机）
wavelength = 654e-9      # 原 1568     -> 1550 nm
z_input_to_first = 0 # 40 μm # 新增：输入面到第一层的传播距离

phase_option = 4
#phase_option 1: (0,0,...,0)
#phase_option 2: (0,2pi,...,2pi)
#phase_option 3: (0,pi,...,2pi)
#phase_option 4: eigenmodes
#phase_option 5: (0,pi,...,pi)

def build_mode_context(base_modes: np.ndarray, num_modes: int) -> dict:
    """
    Prepare mode-dependent tensors and weights for a given num_modes value.
    """
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
            num_data, num_modes, phase_option
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


def build_uniform_fractions(count: int) -> tuple[float, ...]:
    """
    Evenly spaced fractions in (0, 1) used to sample propagation paths.
    """
    if count <= 0:
        return ()
    fractions = np.linspace(1.0 / (count + 1), count / (count + 1), count, dtype=float)
    return tuple(float(f) for f in fractions)


#%% 从 .mat 载入 (H, W, M) 的复数模场
max_modes_needed = max([num_modes] + num_modes_sweep_options)
eigenmodes_OM4 = load_complex_modes_from_mat(
    'mmf_3modes_123_PD_1.2.mat',
    key='modes_field'
)
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)
if eigenmodes_OM4.shape[2] < max_modes_needed:
    raise ValueError(
        f"Source modes only provide {eigenmodes_OM4.shape[2]} modes < required {max_modes_needed}."
    )

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]
loaded_field_size = int(MMF_data_ts.shape[-1])
if field_size != loaded_field_size:
    print(f"field_size={field_size} does not match mode size {loaded_field_size}; using mode size.")
    field_size = loaded_field_size

#%% labels generation upto the prediction case
'''
pred_case = 1: only amplitudes prediction
pred_case = 2: only phases prediction
pred_case = 3: amplitudes and phases prediction
pred_case = 4: amplitudes and phases prediction (extra energy phase area)
'''
#
pred_case = 1
label_size = layer_size
detection_masks = None

if pred_case == 1: # 3
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
    elif label_pattern_mode == "mixed":
        circle_radius = circle_focus_radius
        pattern_size = circle_radius * 2
        if pattern_size % 2 == 0:
            pattern_size += 1
        if len(detector_shapes) < num_detector:
            raise ValueError(
                f"detector_shapes 长度需 >= num_detector，但得到 {len(detector_shapes)} < {num_detector}"
            )
        pattern_stack = generate_detector_patterns(
            pattern_size,
            pattern_size,
            num_detector,
            shapes=detector_shapes,
            equal_area=True,
        )
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
    MMF_Label_data = torch.from_numpy(
        np.stack(mode_label_maps, axis=2).astype(np.float32)
    )
    if label_pattern_mode == "mixed":
        detection_masks = np.stack(
            [(label > 0.5).astype(np.float32) for label in mode_label_maps], axis=0
        )
    focus_radius = detector_focus_radius
    detectsize = detector_detectsize

#%% Build training dataset
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
    label_data = torch.zeros([num_train_samples, 1, layer_size, layer_size])
    amplitude_weights = torch.from_numpy(amplitudes_phases[:, 0:num_modes]).float()
    energy_weights = amplitude_weights**2  # 标签改为能量/强度权重
    combined_labels = (
        energy_weights[:, None, None, :] * MMF_Label_data.unsqueeze(0)
    ).sum(dim=3)    #重点用的是：能量去乘基本的模
    label_data[:, 0, :, :] = combined_labels

    complex_weights = amplitudes * np.exp(1j * phases) #生成输出的逻辑不变还是用amp哈
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_train_samples, num_modes, field_size
    ).to(torch.complex64)

    train_dataset = [
        prepare_sample(image_data[i], label_data[i], layer_size) for i in range(num_train_samples)
    ]
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
    train_dataset = super_train_ctx["dataset"]
    train_tensor_data = super_train_ctx["tensor_dataset"]
    image_data = super_train_ctx["image_data"]
    label_data = train_tensor_data.tensors[1]
    amplitudes = super_train_ctx["amplitudes"]
    phases = super_train_ctx["phases"]
    amplitudes_phases = super_train_ctx["amplitudes_phases"]
else:
    raise ValueError(f"Unknown training_dataset_mode: {training_dataset_mode}")

label_test_data = label_data
image_test_data = image_data


#%% Create test dataset
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_tensor_data,
    batch_size=batch_size,
    shuffle=True,               # 顺序会被 g 固定
    generator=g,                # 固定打乱
   
)
superposition_eval_ctx: dict | None = None
if evaluation_mode == "eigenmode":
    test_dataset = train_dataset
    test_tensor_data = train_tensor_data
    test_loader = DataLoader(test_tensor_data, batch_size=batch_size, shuffle=False)
    eval_amplitudes = amplitudes
    eval_amplitudes_phases = amplitudes_phases
    eval_phases = phases
    image_test_data = image_data
elif evaluation_mode == "superposition":
    if pred_case != 1:
        raise ValueError("Superposition evaluation mode currently supports pred_case == 1 only.")
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
    superposition_eval_ctx = super_ctx
else:
    raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

#%% Generate detection regions using existing function
if pred_case ==1:
    evaluation_regions = create_evaluation_regions(layer_size, layer_size, num_detector, focus_radius, detectsize)
    print("Detection Regions:", evaluation_regions)
    if show_detection_overlap_debug:
        detection_debug_dir = Path("results/detection_region_debug")
        detection_debug_dir.mkdir(parents=True, exist_ok=True)
        if detection_masks is not None:
            overlap_map = np.sum(detection_masks > 0.5, axis=0).astype(np.float32)
        else:
            overlap_map = np.zeros((layer_size, layer_size), dtype=np.float32)
            for (x0, x1, y0, y1) in evaluation_regions:
                overlap_map[y0:y1, x0:x1] += 1.0
        overlap_pixels = int(np.count_nonzero(overlap_map > 1.0 + 1e-6))
        max_overlap = float(overlap_map.max()) if overlap_map.size else 0.0

        label_sample_np = None
        if "label_data" in locals() and isinstance(label_data, torch.Tensor) and label_data.shape[0] > 0:
            sample_idx = min(max(0, detection_overlap_label_index), label_data.shape[0] - 1)
            label_sample_np = label_data[sample_idx, 0].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if label_sample_np is not None:
            im0 = axes[0].imshow(label_sample_np, cmap="inferno")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            axes[0].set_title(f"Label sample #{sample_idx + 1} with detectors")
        else:
            axes[0].imshow(np.zeros((layer_size, layer_size), dtype=np.float32), cmap="Greys")
            axes[0].set_title("Detector layout (no label sample)")
        axes[0].set_axis_off()

        if detection_masks is not None:
            for idx_region, mask in enumerate(detection_masks):
                color = plt.cm.tab20(idx_region % 20)
                axes[0].contour(mask, levels=[0.5], colors=[color], linewidths=1.0)
                ys, xs = np.where(mask > 0.5)
                if xs.size > 0:
                    center_x = float(xs.mean())
                    center_y = float(ys.mean())
                    axes[0].text(
                        center_x,
                        center_y,
                        f"M{idx_region + 1}",
                        color=color,
                        fontsize=8,
                        weight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4, edgecolor="none"),
                    )
        else:
            circle_radius = focus_radius
            for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
                color = plt.cm.tab20(idx_region % 20)
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=color, facecolor='none')
                axes[0].add_patch(rect)
                center_x = (x0 + x1) / 2.0
                center_y = (y0 + y1) / 2.0
                circle = Circle(
                    (center_x, center_y),
                    radius=circle_radius,
                    linewidth=1.0,
                    edgecolor=color,
                    linestyle="--",
                    fill=False,
                )
                axes[0].add_patch(circle)
                axes[0].text(
                    x0 + 1,
                    y0 + 4,
                    f"M{idx_region + 1}",
                    color=color,
                    fontsize=8,
                    weight="bold",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.4, edgecolor="none"),
                )

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


def run_experiment_for_layer_size(
    layer_size: int,
    *,
    num_modes: int,
    mmf_data_np: np.ndarray,
    mmf_data_ts: torch.Tensor,
    base_amplitudes: np.ndarray,
    base_phases: np.ndarray,
) -> dict:
    """
    Train and evaluate the ODNN for a given (layer_size, num_modes) pair and return key metrics.
    The flow mirrors the main script but is trimmed to keep the sweep concise.
    """
    print(f"\n===== Running experiment for layer_size={layer_size}, num_modes={num_modes} =====")
    viz_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_root = Path("results/prediction_viz") / f"m{num_modes}_ls{layer_size}_{viz_tag}"
    label_size = layer_size
    focus_radius = circle_focus_radius
    detectsize = circle_detectsize
    detection_masks = None

    if pred_case != 1:
        raise ValueError("Layer-size sweep currently supports pred_case == 1 only.")

    # Label generation (pred_case == 1 branch)
    num_detector = num_modes
    detector_focus_radius = focus_radius
    detector_detectsize = detectsize
    if label_pattern_mode == "eigenmode":
        pattern_stack = np.transpose(np.abs(mmf_data_np), (1, 2, 0))
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
    elif label_pattern_mode == "mixed":
        circle_radius = circle_focus_radius
        pattern_size = circle_radius * 2
        if pattern_size % 2 == 0:
            pattern_size += 1
        if len(detector_shapes) < num_detector:
            raise ValueError(
                f"detector_shapes 长度需 >= num_detector，但得到 {len(detector_shapes)} < {num_detector}"
            )
        pattern_stack = generate_detector_patterns(
            pattern_size,
            pattern_size,
            num_detector,
            shapes=detector_shapes,
            equal_area=True,
        )
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
    if label_pattern_mode == "mixed":
        detection_masks = np.stack(
            [(label > 0.5).astype(np.float32) for label in mode_label_maps], axis=0
        )
    focus_radius = detector_focus_radius
    detectsize = detector_detectsize

    # Build training dataset
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
        label_data = torch.zeros([num_train_samples, 1, label_size, label_size])
        amplitude_weights = torch.from_numpy(amplitudes_phases[:, 0:num_modes]).float()
        energy_weights = amplitude_weights**2  # 标签改为能量/强度权重
        combined_labels = (
            energy_weights[:, None, None, :] * MMF_Label_data.unsqueeze(0)
        ).sum(dim=3)
        label_data[:, 0, :, :] = combined_labels

        complex_weights = amplitudes * np.exp(1j * phases)
        complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
        image_data = generate_fields_ts(
            complex_weights_ts, mmf_data_ts, num_train_samples, num_modes, field_size
        ).to(torch.complex64)

        train_dataset = [
            prepare_sample(image_data[i], label_data[i], label_size) for i in range(num_train_samples)
        ]
        train_tensor_data = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_dataset)])
    elif training_dataset_mode == "superposition":
        num_train_samples = num_superposition_train_samples
        super_train_ctx = build_superposition_eval_context(
            num_train_samples,
            num_modes=num_modes,
            field_size=field_size,
            layer_size=label_size,
            mmf_modes=mmf_data_ts,
            mmf_label_data=MMF_Label_data,
            batch_size=batch_size,
            second_mode_half_range=True,
            rng_seed=superposition_train_seed,
        )
        train_dataset = super_train_ctx["dataset"]
        train_tensor_data = super_train_ctx["tensor_dataset"]
        image_data = super_train_ctx["image_data"]
        label_data = train_tensor_data.tensors[1]
        amplitudes = super_train_ctx["amplitudes"]
        phases = super_train_ctx["phases"]
        amplitudes_phases = super_train_ctx["amplitudes_phases"]
    else:
        raise ValueError(f"Unknown training_dataset_mode: {training_dataset_mode}")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_tensor_data,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )

    superposition_eval_ctx: dict | None = None
    if evaluation_mode == "eigenmode":
        test_dataset = train_dataset
        test_tensor_data = train_tensor_data
        test_loader = DataLoader(test_tensor_data, batch_size=batch_size, shuffle=False)
        eval_amplitudes = amplitudes
        eval_amplitudes_phases = amplitudes_phases
        eval_phases = phases
        image_test_data = image_data
    elif evaluation_mode == "superposition":
        super_ctx = build_superposition_eval_context(
            num_superposition_eval_samples,
            num_modes=num_modes,
            field_size=field_size,
            layer_size=label_size,
            mmf_modes=mmf_data_ts,
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
        superposition_eval_ctx = super_ctx
    else:
        raise ValueError(f"Unknown evaluation_mode: {evaluation_mode}")

    evaluation_regions = create_evaluation_regions(label_size, label_size, num_detector, focus_radius, detectsize)

    layer_results: list[dict] = []
    for num_layer in num_layer_option:
        print(f"\nTraining D2NN with {num_layer} layers (layer_size={layer_size}, num_modes={num_modes})...\n")

        D2NN = D2NNModel(
            num_layers=num_layer,
            layer_size=label_size,
            z_layers=z_layers,
            z_prop=z_prop,
            pixel_size=pixel_size,
            wavelength=wavelength,
            device=device,
            padding_ratio=0.5,
            z_input_to_first=z_input_to_first,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(D2NN.parameters(), lr=1.99)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        epochs = 1000
        losses = []

        for epoch in range(1, epochs + 1):
            D2NN.train()
            epoch_loss = 0
            for images, labels in train_loader:
                images = images.to(device, dtype=torch.complex64, non_blocking=True)
                labels = labels.to(device, dtype=torch.float32,   non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = D2NN(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
                print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.18f}')

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        metrics = evaluate_spot_metrics(
            D2NN,
            test_loader,
            evaluation_regions,
            detection_masks=detection_masks,
            detect_radius=detectsize,
            device=device,
            pred_case=pred_case,
            num_modes=num_modes,
            phase_option=phase_option,
            amplitudes=eval_amplitudes,
            amplitudes_phases=eval_amplitudes_phases,
            phases=eval_phases,
            mmf_modes=mmf_data_ts,
            field_size=field_size,
            image_test_data=image_test_data,
        )

        print(
            format_metric_report(
                num_modes=num_modes,
                phase_option=phase_option,
                pred_case=pred_case,
                label=f"{num_layer} layers @ {layer_size}",
                metrics=metrics,
            )
        )
        # Save qualitative predictions for a few samples
        viz_dir = viz_root / f"L{num_layer}"
        saved_plots = save_prediction_diagnostics(
            D2NN,
            test_dataset,
            evaluation_regions=evaluation_regions,
            layer_size=label_size,
            detect_radius=detectsize,
            num_samples=3,
            output_dir=viz_dir,
            device=device,
            tag=f"m{num_modes}_ls{layer_size}_L{num_layer}",
        )
        if saved_plots:
            print(f"✔ Saved prediction diagnostics ({len(saved_plots)} samples) -> {saved_plots[0].parent}")
        else:
            print("⚠ No prediction diagnostics were saved (empty dataset?)")
        layer_results.append(
            {
                "num_layer": num_layer,
                "metrics": metrics,
                "losses": losses,
                "prediction_plots": [str(p) for p in saved_plots],
            }
        )

    avg_relative_amp_errors = [
        float(r["metrics"].get("avg_relative_amp_err", float("nan"))) for r in layer_results
    ]
    return {
        "layer_size": layer_size,
        "evaluation_regions": evaluation_regions,
        "results": layer_results,
        "avg_relative_amp_errors": avg_relative_amp_errors,
    }

#%% D2NN models and train them，可以考虑多种layersize的可能
if run_layer_size_sweep:
    sweep_dir = Path("results/layer_size_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_results = []
    rel_err_matrix = np.full(
        (len(num_modes_sweep_options), len(layer_size_options)),
        np.nan,
        dtype=np.float64,
    )

    for mode_idx, sweep_num_modes in enumerate(num_modes_sweep_options):
        mode_ctx = build_mode_context(eigenmodes_OM4, sweep_num_modes)
        mode_results = []
        # Skip the smallest canvas when mode count is high to avoid overcrowded layouts
        mode_layer_sizes = [
            ls for ls in layer_size_options
            if not (sweep_num_modes in (50, 100) and ls == 110)
        ]

        for ls in mode_layer_sizes:
            mode_results.append(
                run_experiment_for_layer_size(
                    ls,
                    num_modes=sweep_num_modes,
                    mmf_data_np=mode_ctx["mmf_data_np"],
                    mmf_data_ts=mode_ctx["mmf_data_ts"],
                    base_amplitudes=mode_ctx["base_amplitudes"],
                    base_phases=mode_ctx["base_phases"],
                )
            )

        sweep_results.append(
            {"num_modes": sweep_num_modes, "results": mode_results, "layer_sizes": mode_layer_sizes}
        )
        mode_rel_errs = [float(np.nanmean(r["avg_relative_amp_errors"])) for r in mode_results]
        for ls, err in zip(mode_layer_sizes, mode_rel_errs):
            col_idx = layer_size_options.index(ls)
            rel_err_matrix[mode_idx, col_idx] = err

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots()
    for mode_entry in sweep_results:
        xs = [res["layer_size"] for res in mode_entry["results"]]
        ys = [float(np.nanmean(res["avg_relative_amp_errors"])) for res in mode_entry["results"]]
        ax.plot(
            xs,
            ys,
            marker="o",
            label=f"{mode_entry['num_modes']} modes",
        )
    ax.set_xlabel("Layer size")
    ax.set_ylabel("Relative amplitude error")
    ax.set_title("Relative amp. error vs layer size (multi num_modes)")
    ax.legend(title="num_modes")
    ax.grid(True, alpha=0.3)
    sweep_plot_path = sweep_dir / f"relative_amp_error_vs_layer_size_multi_mode_{timestamp_tag}.png"
    fig.savefig(sweep_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    sweep_mat_path = sweep_dir / f"relative_amp_error_vs_layer_size_multi_mode_{timestamp_tag}.mat"
    savemat(
        str(sweep_mat_path),
        {
            "layer_size": np.asarray(layer_size_options, dtype=np.float64),
            "avg_relative_amp_error": rel_err_matrix,
            "num_modes_list": np.asarray(num_modes_sweep_options, dtype=np.int32),
        },
    )

    print("\nLayer size sweep summary:")
    for mode_entry in sweep_results:
        rel_errs = [float(np.nanmean(r["avg_relative_amp_errors"])) for r in mode_entry["results"]]
        err_pairs = ", ".join(
            f"ls{ls}:{err:.6f}" for ls, err in zip(mode_entry["layer_sizes"], rel_errs)
        )
        skipped = [ls for ls in layer_size_options if ls not in mode_entry["layer_sizes"]]
        skip_note = f" (skipped: {skipped})" if skipped else ""
        print(f" - num_modes={mode_entry['num_modes']}: {err_pairs}{skip_note}")
    print("Note: skipped layer sizes are recorded as NaN in the saved matrix.")
    print(f"✔ Saved layer-size sweep plot -> {sweep_plot_path}")
    print(f"✔ Saved layer-size sweep data (.mat) -> {sweep_mat_path}")
    raise SystemExit

for num_layer in num_layer_option:
    print(f"\nTraining D2NN with {num_layer} layers...\n")

    D2NN = D2NNModel(
        num_layers=num_layer,
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelength=wavelength,
        device=device,
        padding_ratio=0.5,
        z_input_to_first=z_input_to_first,   # NEW
    ).to(device)

    print(D2NN)

    # Training
    criterion = nn.MSELoss()  # Define loss function (对比的是loss)
    optimizer = optim.Adam(D2NN.parameters(), lr=1.99) 
    scheduler = ExponentialLR(optimizer, gamma=0.99)  
    epochs = 1000
    losses = []
    epoch_durations: list[float] = []
    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        D2NN.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32,   non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = D2NN(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
        losses.append(avg_loss)  # the loss for each model
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(
                f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.18f}, '
                f'Epoch Time: {epoch_duration:.2f} seconds'
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_training_time = time.time() - training_start_time
    print(
        f'Total training time for {num_layer}-layer model: {total_training_time:.2f} seconds '
        f'(~{total_training_time / 60:.2f} minutes)'
    )
    all_losses.append(losses)  # save the loss for each model
    training_output_dir = Path("results/training_analysis")
    training_output_dir.mkdir(parents=True, exist_ok=True)
    epochs_array = np.arange(1, epochs + 1, dtype=np.int32)
    cumulative_epoch_times = np.cumsum(epoch_durations)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots()
    ax.plot(epochs_array, losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"D2NN Training Loss ({num_layer} layers)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    loss_plot_path = training_output_dir / f"loss_curve_layers{num_layer}_m{num_modes}_ls{layer_size}_{timestamp_tag}.png"
    fig.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_time, ax_time = plt.subplots()
    ax_time.plot(epochs_array, cumulative_epoch_times, label="Cumulative Time")
    ax_time.set_xlabel("Epoch")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title(f"Cumulative Training Time ({num_layer} layers)")
    ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_time.legend()
    time_plot_path = training_output_dir / f"epoch_time_layers{num_layer}_m{num_modes}_ls{layer_size}_{timestamp_tag}.png"
    fig_time.savefig(time_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig_time)

    mat_path = training_output_dir / f"training_curves_layers{num_layer}_m{num_modes}_ls{layer_size}_{timestamp_tag}.mat"
    savemat(
        str(mat_path),
        {
            "epochs": epochs_array,
            "losses": np.array(losses, dtype=np.float64),
            "epoch_durations": np.array(epoch_durations, dtype=np.float64),
            "cumulative_epoch_times": np.array(cumulative_epoch_times, dtype=np.float64),
            "total_training_time": np.array([total_training_time], dtype=np.float64),
            "num_layers": np.array([num_layer], dtype=np.int32),
        },
    )

    print(f"✔ Saved training loss plot -> {loss_plot_path}")
    print(f"✔ Saved cumulative time plot -> {time_plot_path}")
    print(f"✔ Saved training log data (.mat) -> {mat_path}")

    propagation_dir = Path("results/propagation_slices")
    eigenmode_index = min(2, MMF_data_ts.shape[0] - 1)
    layer_fractions = [build_uniform_fractions(prop_slices_per_segment) for _ in range(num_layer)]
    output_fractions = build_uniform_fractions(prop_output_slices)
    propagation_summary = capture_eigenmode_propagation(
        model=D2NN,
        eigenmode_field=MMF_data_ts[eigenmode_index],
        mode_index=eigenmode_index,
        layer_size=layer_size,
        z_input_to_first=z_input_to_first,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelength=wavelength,
        output_dir=propagation_dir,
        tag=f"layers{num_layer}_{timestamp_tag}",
        fractions_between_layers=layer_fractions,
        output_fractions=output_fractions,
    )
    print(f"✔ Saved eigenmode-{eigenmode_index + 1} propagation plot -> {propagation_summary['fig_path']}")
    print(f"✔ Saved eigenmode-{eigenmode_index + 1} propagation data (.mat) -> {propagation_summary['mat_path']}")
    energies = np.asarray(propagation_summary.get("energies", []), dtype=np.float64)
    z_positions = np.asarray(propagation_summary.get("z_positions", []), dtype=np.float64)
    if energies.size > 0 and energies[0] != 0:
        energy_drop_pct = (energies[0] - energies[-1]) / energies[0] * 100.0
        print(
            f"   Energy trace: start={energies[0]:.4e}, end={energies[-1]:.4e}, "
            f"drop={energy_drop_pct:.2f}% over {energies.size} slices"
        )
        # 想看具体位置可以考虑更改这个代码去看看每个具体具体的能量
        # if z_positions.size == energies.size:
        #     preview = ", ".join(
        #         f"{z_positions[i]*1e6:.1f}µm:{energies[i]:.3e}"
        #         for i in range(min(5, energies.size))
        #     )
        #     print(f"   z/energy (first slices): {preview}")

    mode_triptych_records: list[dict[str, str | int]] = []
    if evaluation_mode == "eigenmode":
        triptych_dir = Path("results/mode_triptychs")
        mode_tag = f"layers{num_layer}_m{num_modes}_{timestamp_tag}"
        for mode_idx in range(min(num_modes, len(MMF_data_ts))):
            label_tensor = label_data[mode_idx, 0]
            record = save_mode_triptych(
                model=D2NN,
                mode_index=mode_idx,
                eigenmode_field=MMF_data_ts[mode_idx],
                label_field=label_tensor,
                layer_size=layer_size,
                output_dir=triptych_dir,
                tag=mode_tag,
                evaluation_regions=evaluation_regions,
                detect_radius=detectsize,
                show_mask_overlays=True,
            )
            mode_triptych_records.append(
                {
                    "mode": mode_idx + 1,
                    "fig": record["fig_path"],
                    "mat": record["mat_path"],
                }
            )
            print(
                f"✔ Saved mode {mode_idx + 1} triptych -> {record['fig_path']}\n"
                f"  MAT -> {record['mat_path']}"
            )

    all_training_summaries.append(
        {
            "num_layers": num_layer,
            "total_time": total_training_time,
            "loss_plot": str(loss_plot_path),
            "time_plot": str(time_plot_path),
            "mat_path": str(mat_path),
            "propagation_fig": propagation_summary["fig_path"],
            "propagation_mat": propagation_summary["mat_path"],
            "mode_triptychs": mode_triptych_records,
        }
    )
   
    # === after training ===
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {
        "state_dict": D2NN.state_dict(),
        "meta": {
            "num_layers":        len(D2NN.layers),
            "layer_size":        layer_size,
            "z_layers":          z_layers,
            "z_prop":            z_prop,
            "pixel_size":        pixel_size,
            "wavelength":        wavelength,
            "padding_ratio":     0.5,         
            "field_size":        field_size,  
            "num_modes":         num_modes, 
            "z_input_to_first":  z_input_to_first, 
        }
    }
    save_path = os.path.join(ckpt_dir, f"odnn_{len(D2NN.layers)}layers_m{num_modes}_ls{layer_size}.pth")
    torch.save(ckpt, save_path)
    print("✔ Saved model ->", save_path)
    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Cache phase masks for later visualization/export
    phase_masks = []
    for layer in D2NN.layers:
        phase_np = layer.phase.detach().cpu().numpy()
        phase_masks.append(np.remainder(phase_np, 2 * np.pi))
    all_phase_masks.append(phase_masks)

    # 存给matlab用SLM的mask.mat
    mask_dir = Path("results_MD")
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_mat_path = mask_dir / f"phase_masks_{len(phase_masks)}layers_{mask_tag}.mat"
    phase_stack = np.stack([np.asarray(mask, dtype=np.float32) for mask in phase_masks], axis=0)
    savemat(
        str(mask_mat_path),
        {
            "phase_masks": phase_stack,
            "num_layers": np.array([len(phase_masks)], dtype=np.int32),
        },
        do_compression=True,
    )
    print(f"✔ Saved phase masks (.mat) -> {mask_mat_path}")

    # Collect evaluation metrics for this model
    metrics = evaluate_spot_metrics(
        D2NN,
        test_loader,
        evaluation_regions,
        detection_masks=detection_masks,
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

    # Qualitative check: label vs prediction heatmaps + amplitude bars
    diag_dir = Path("results/prediction_viz") / f"main_L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    diag_paths = save_prediction_diagnostics(
        D2NN,
        test_dataset,
        evaluation_regions=evaluation_regions,
        layer_size=layer_size,
        detect_radius=detectsize,
        num_samples=3,
        output_dir=diag_dir,
        device=device,
        tag=f"main_L{num_layer}",
    )
    if diag_paths:
        print(f"✔ Saved prediction diagnostics ({len(diag_paths)} samples) -> {diag_paths[0].parent}")
    else:
        print("No prediction diagnostics were saved (empty dataset?)")

    model_metrics.append(metrics)
    all_amplitudes_diff.append(metrics.get("amplitudes_diff", np.array([])))
    all_average_amplitudes_diff.append(float(metrics.get("avg_amplitudes_diff", float("nan"))))
    all_amplitudes_relative_diff.append(float(metrics.get("avg_relative_amp_err", float("nan"))))
    all_complex_weights_pred.append(metrics.get("complex_weights_pred", np.array([])))
    all_image_data_pred.append(metrics.get("image_data_pred", np.array([])))
    all_cc_recon_amp.append(metrics.get("cc_recon_amp", np.array([])))
    all_cc_recon_phase.append(metrics.get("cc_recon_phase", np.array([])))
    all_cc_real.append(metrics.get("cc_real", np.array([])))
    all_cc_imag.append(metrics.get("cc_imag", np.array([])))
    #看看testset的参数值
    print(
        format_metric_report(
            num_modes=num_modes,
            phase_option=phase_option,
            pred_case=pred_case,
            label=f"{num_layer} layers",
            metrics=metrics,
        )
    )



#%% Metrics vs. layer count

if model_metrics:
    metrics_dir = Path("results/metrics_analysis")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    layer_counts = np.asarray(num_layer_option[: len(model_metrics)], dtype=np.int32)
    amp_err = np.asarray(all_average_amplitudes_diff[: len(layer_counts)], dtype=np.float64)
    amp_err_rel = np.asarray(all_amplitudes_relative_diff[: len(layer_counts)], dtype=np.float64)

    cc_amp_mean_list: list[float] = []
    cc_amp_std_list: list[float] = []
    for cc_arr in all_cc_recon_amp[: len(layer_counts)]:
        cc_np = np.asarray(cc_arr, dtype=np.float64)
        if cc_np.size:
            cc_amp_mean_list.append(float(np.nanmean(cc_np)))
            cc_amp_std_list.append(float(np.nanstd(cc_np)))
        else:
            cc_amp_mean_list.append(float("nan"))
            cc_amp_std_list.append(float("nan"))
    cc_amp_mean = np.asarray(cc_amp_mean_list, dtype=np.float64)
    cc_amp_std = np.asarray(cc_amp_std_list, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(layer_counts, amp_err, marker="o")
    axes[0].set_ylabel("avg_amp_error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layer_counts, amp_err_rel, marker="o", color="tab:orange")
    axes[1].set_ylabel("avg_relative_amp_error")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(
        layer_counts,
        cc_amp_mean,
        yerr=cc_amp_std,
        marker="o",
        color="tab:green",
        ecolor="tab:green",
        capsize=4,
    )
    axes[2].set_xlabel("Number of layers")
    axes[2].set_ylabel("cc_amp mean ± std")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Metrics vs. layer count", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    metrics_plot_path = metrics_dir / f"metrics_vs_layers_{metrics_tag}.png"
    fig.savefig(metrics_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics_mat_path = metrics_dir / f"metrics_vs_layers_{metrics_tag}.mat"
    savemat(
        str(metrics_mat_path),
        {
            "layers": layer_counts.astype(np.float64),
            "avg_amp_error": amp_err,
            "avg_relative_amp_error": amp_err_rel,
            "cc_amp_mean": cc_amp_mean,
            "cc_amp_std": cc_amp_std,
        },
    )

    print(f"✔ Metrics vs. layers plot saved -> {metrics_plot_path}")
    print(f"✔ Metrics vs. layers data (.mat) -> {metrics_mat_path}")



#%% Propagation slices & mask export，看一个固定的输入的切片输出存个备份mat
temp_dataset = test_dataset
FIXED_E_INDEX = 4

def get_fixed_input(dataset, idx, device):
    if isinstance(dataset, list):
        sample = dataset[idx][0]
    else:
        sample = dataset.tensors[0][idx]
    return sample.squeeze(0).to(device)


assert len(temp_dataset) > 0, "test_dataset 为空"
if prop_slice_sample_mode == "random":
    rng = np.random.default_rng(prop_slice_seed)
    sample_idx = int(rng.integers(low=0, high=len(temp_dataset)))
else:
    sample_idx = FIXED_E_INDEX % len(temp_dataset)
temp_E = get_fixed_input(temp_dataset, sample_idx, device)
print(f"[Slices] Using sample #{sample_idx} for propagation snapshots (mode={prop_slice_sample_mode})")

z_start = 0.0
z_step = 5e-6
z_prop_plus = z_prop

save_root = Path("results_MD")
save_root.mkdir(parents=True, exist_ok=True)
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix = f"ODNN_vis_{run_stamp}"

phase_mask_entries: list[tuple[int, list[np.ndarray]]] = []
if all_phase_masks:
    phase_mask_entries.append((len(all_phase_masks), all_phase_masks[-1]))

for i_model, phase_masks in phase_mask_entries:
    model_dir = save_root / f"m{i_model}"
    scans, camera_field = visualize_model_slices(
        D2NN,
        phase_masks,
        temp_E,
        output_dir=model_dir,
        sample_tag=f"m{i_model}",
        z_input_to_first=z_input_to_first,
        z_layers=z_layers,
        z_prop_plus=z_prop_plus,
        z_step=z_step,
        pixel_size=pixel_size,
        wavelength=wavelength,
        kmax=prop_scan_kmax,
    )

    phase_stack = np.stack([np.asarray(mask, dtype=np.float32) for mask in phase_masks], axis=0)
    meta = {
        "z_start": float(z_start),
        "z_step": float(z_step),
        "z_layers": float(z_layers),
        "z_prop": float(z_prop),
        "z_prop_plus": float(z_prop_plus),
        "pixel_size": float(pixel_size),
        "wavelength": float(wavelength),
        "layer_size": int(layer_size),
        "padding_ratio": 0.5,
    }

    mat_path = model_dir / f"{filename_prefix}_mask{i_model}.mat"
    save_to_mat_light_plus(
        mat_path,
        phase_stack=phase_stack,
        input_field=temp_E.detach().cpu().numpy(),
        scans=scans,
        camera_field=camera_field,
        sample_stacks_kmax=20,
        save_amplitude_only=False,
        meta=meta,
    )
    print("Saved ->", mat_path)

    save_masks_one_file_per_layer(
        phase_masks,
        out_dir=model_dir,
        base_name=f"{filename_prefix}_MASK",
        save_degree=False,
        use_xlsx=True,
    )




# #%% 第一层mask做一些位移

# if run_misalignment_robustness and pred_case == 1:
#     robustness_dir = Path("results/robustness_analysis")
#     robustness_dir.mkdir(parents=True, exist_ok=True)
#     robustness_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     dx_um_values = np.arange(-20.0, 20.0, 2.0, dtype=np.float32)
#     dy_um_values = np.arange(-20.0, 20.0, 2.0, dtype=np.float32)
#     amp_err_surface = np.zeros((len(dy_um_values), len(dx_um_values)), dtype=np.float64)

#     def um_to_pixels(shift_um: float) -> int:
#         return int(round((shift_um * 1e-6) / pixel_size))

#     print("\nRunning misalignment robustness sweep (±200 µm, 5 µm steps)...")
#     for iy, dy_um in enumerate(dy_um_values):
#         shift_y_px = um_to_pixels(float(dy_um))
#         for ix, dx_um in enumerate(dx_um_values):
#             shift_x_px = um_to_pixels(float(dx_um))
#             metrics = compute_amp_relative_error_with_shift(
#                 D2NN,
#                 test_loader,
#                 shift_y_px=shift_y_px,
#                 shift_x_px=shift_x_px,
#                 evaluation_regions=evaluation_regions,
#                 pred_case=pred_case,
#                 num_modes=num_modes,
#                 eval_amplitudes=eval_amplitudes,
#                 eval_amplitudes_phases=eval_amplitudes_phases,
#                 eval_phases=eval_phases,
#                 phase_option=phase_option,
#                 mmf_modes=MMF_data_ts,
#                 field_size=field_size,
#                 image_test_data=image_test_data,
#                 device=device,
#             )
#             amp_err_surface[iy, ix] = float(metrics.get("avg_relative_amp_err", float("nan")))
#         print(f"  Completed shift row {iy + 1}/{len(dy_um_values)} (Δy = {dy_um:.1f} µm)")

#     DX, DY = np.meshgrid(dx_um_values, dy_um_values)
#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     surf = ax.plot_surface(DX, DY, amp_err_surface, cmap="viridis")
#     ax.set_xlabel("Δx (µm)")
#     ax.set_ylabel("Δy (µm)")
#     ax.set_zlabel("Relative amplitude error")
#     ax.set_title("Amplitude error vs. input-mask misalignment")
#     fig.colorbar(surf, shrink=0.6, aspect=12)
#     fig.tight_layout()

#     robustness_fig_path = robustness_dir / f"misalignment_surface_{robustness_tag}.png"
#     fig.savefig(robustness_fig_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     dx_px_values = np.rint(dx_um_values * 1e-6 / pixel_size).astype(np.int32)
#     dy_px_values = np.rint(dy_um_values * 1e-6 / pixel_size).astype(np.int32)
#     robustness_mat_path = robustness_dir / f"misalignment_surface_{robustness_tag}.mat"
#     savemat(
#         str(robustness_mat_path),
#         {
#             "dx_um": dx_um_values.astype(np.float32),
#             "dy_um": dy_um_values.astype(np.float32),
#             "dx_pixels": dx_px_values,
#             "dy_pixels": dy_px_values,
#             "relative_amp_error": amp_err_surface.astype(np.float64),
#             "pixel_size_m": np.array([pixel_size], dtype=np.float64),
#             "step_um": np.array([5.0], dtype=np.float32),
#             "range_um": np.array([200.0], dtype=np.float32),
#         },
#     )

#     if all_training_summaries:
#         all_training_summaries[-1]["robustness_fig"] = str(robustness_fig_path)
#         all_training_summaries[-1]["robustness_mat"] = str(robustness_mat_path)

#     print(f"\n✔ Misalignment robustness surface saved -> {robustness_fig_path}")
#     print(f"✔ Misalignment robustness data (.mat) -> {robustness_mat_path}")


