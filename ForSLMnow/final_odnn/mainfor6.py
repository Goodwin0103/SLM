#%%
import json
import math
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

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
    device = torch.device('cuda:0')           # 或者 'cuda:0'
    print('Using Device:', device)
else:
    device = torch.device('cpu')
    print('Using Device: CPU')


#%% data generation (lightfield)
field_size = 25 #the field size in eigenmodes_OM4 is 50 pixels
layer_size_options = [110]  # layer canvas size sweep
run_layer_size_sweep = False  # toggle to run a sweep before the legacy single run
layer_size = layer_size_options[0]
num_data = 1000 # options: 1. random datas 2.eigenmodes
num_modes = 5 #default: take first N modes from the 103-mode file
num_modes_sweep_options = [5]
circle_focus_radius = 5 # radius when using uniform circular detectors
circle_detectsize = 10  # durchmeter (/2)
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
label_pattern_mode = "circle"  # options: "eigenmode", "circle"，用圆还是用本征模
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

# # Wavelength sweep selection: "mode_isolation", "relative_amp_error", or "auto" to match evaluation_mode
# wavelength_sweep_mode = "relative_amp_error"
# When focusing on relative amplitude error, control which dataset powers the sweep
# wavelength_relative_eval_mode = "superposition"  # options: "match", "superposition"
# wavelength_relative_eval_samples = 1000
# wavelength_relative_eval_seed = 20240116


# Training data selection: 默认用 eigenmode，也可以改成 superposition 并设定样本数量
training_dataset_mode = "eigenmode"  # options: "eigenmode", "superposition"
num_superposition_train_samples = 100  # superposition 训练样本数
superposition_train_seed = 20240115  # 控制 superposition 训练集的随机性

# Define multiple D2NN models 
num_layer_option = [2,3,4,5,6]   # Define the different layer-number ODNN
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
z_layers   = 40e-6        # 原 47.571e-3  -> 40 μm
pixel_size = 1e-6
z_prop     = 120e-6        # 原 16.74e-2   -> 60 μm plus 40（最后一层到相机）
wavelength = 1568e-9      # 原 1568     -> 1550 nm
z_input_to_first = 40e-6  # 40 μm # 新增：输入面到第一层的传播距离

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
    'mmf_103modes_25_PD_1.15.mat',
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
        detection_debug_dir = Path("detection_region_debug")
        detection_debug_dir.mkdir(parents=True, exist_ok=True)
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
    viz_root = Path("prediction_viz") / f"m{num_modes}_ls{layer_size}_{viz_tag}"
    label_size = layer_size
    focus_radius = circle_focus_radius
    detectsize = circle_detectsize

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

#%% D2NN models and train them，考虑多种layersize的可能
if run_layer_size_sweep:
    sweep_dir = Path("layer_size_sweep")
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
    training_output_dir = Path("training_analysis")
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

    propagation_dir = Path("propagation_slices")
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
        triptych_dir = Path("mode_triptychs")
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

    # Collect evaluation metrics for this model
    metrics = evaluate_spot_metrics(
        D2NN,
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

    # Qualitative check: label vs prediction heatmaps + amplitude bars
    diag_dir = Path("prediction_viz") / f"main_L{num_layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    metrics_dir = Path("metrics_analysis")
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

# #%% Wavelength sweep: relative amplitude error (1550–1700 nm, fixed eval set)
# if pred_case == 1 and all_phase_masks:
#     wavelength_sweep_dir = Path("wavelength_analysis")
#     wavelength_sweep_dir.mkdir(parents=True, exist_ok=True)
#     sweep_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     base_phase_masks = all_phase_masks[-1]  # use the latest trained model
#     geometry = ModelGeometry(
#         layer_size=layer_size,
#         z_layers=z_layers,
#         z_prop=z_prop,
#         pixel_size=pixel_size,
#         z_input_to_first=z_input_to_first,
#     )

#     wavelengths_nm_target = np.arange(1500.0, 1650.0 + 1e-6, 1.0, dtype=np.float64)
#     wavelengths_m_target = wavelengths_nm_target * 1e-9

#     sweep_loader = test_loader
#     sweep_image_data = image_test_data
#     sweep_amplitudes = eval_amplitudes
#     sweep_amplitudes_phases = eval_amplitudes_phases
#     sweep_phases = eval_phases

#     if evaluation_mode == "superposition":
#         # reuse the exact same evaluation set (same 1000 samples + seed) for the sweep
#         if superposition_eval_ctx is None:
#             sweep_ctx = build_superposition_eval_context(
#                 num_superposition_eval_samples,
#                 num_modes=num_modes,
#                 field_size=field_size,
#                 layer_size=layer_size,
#                 mmf_modes=MMF_data_ts,
#                 mmf_label_data=MMF_Label_data,
#                 batch_size=batch_size,
#                 second_mode_half_range=True,
#                 rng_seed=superposition_eval_seed,
#             )
#         else:
#             sweep_ctx = superposition_eval_ctx
#         sweep_loader = sweep_ctx["loader"]
#         sweep_image_data = sweep_ctx["image_data"]
#         sweep_amplitudes = sweep_ctx["amplitudes"]
#         sweep_amplitudes_phases = sweep_ctx["amplitudes_phases"]
#         sweep_phases = sweep_ctx["phases"]

#     wavelengths_m, rel_amp_errors, sweep_metrics = compute_relative_amp_error_wavelength_sweep(
#         base_phase_masks,
#         base_wavelength=wavelength,
#         wavelength_list=wavelengths_m_target,
#         geometry=geometry,
#         device=device,
#         test_loader=sweep_loader,
#         evaluation_regions=evaluation_regions,
#         detect_radius=detectsize,
#         pred_case=pred_case,
#         num_modes=num_modes,
#         phase_option=phase_option,
#         amplitudes=sweep_amplitudes,
#         amplitudes_phases=sweep_amplitudes_phases,
#         phases=sweep_phases,
#         mmf_modes=MMF_data_ts,
#         field_size=field_size,
#         image_test_data=sweep_image_data,
#     )

#     rel_amp_errors = np.asarray(rel_amp_errors, dtype=np.float64)
#     wavelengths_nm = np.asarray(wavelengths_m, dtype=np.float64) * 1e9
#     avg_amp_diff = np.array(
#         [float(metrics.get("avg_amplitudes_diff", float("nan"))) for metrics in sweep_metrics],
#         dtype=np.float64,
#     )

#     fig_linear, ax_linear = plt.subplots()
#     ax_linear.plot(wavelengths_nm, rel_amp_errors, marker="o")
#     ax_linear.set_xlabel("Wavelength (nm)")
#     ax_linear.set_ylabel("Relative amplitude error")
#     ax_linear.set_title("Relative amplitude error vs. wavelength (linear)")
#     ax_linear.grid(True, alpha=0.3)
#     sweep_plot_path_linear = wavelength_sweep_dir / f"wavelength_relative_error_linear_{sweep_tag}.png"
#     fig_linear.savefig(sweep_plot_path_linear, dpi=300, bbox_inches="tight")
#     plt.close(fig_linear)

#     sweep_mat_path = wavelength_sweep_dir / f"wavelength_relative_error_{sweep_tag}.mat"
#     savemat(
#         str(sweep_mat_path),
#         {
#             "wavelength_nm": wavelengths_nm.astype(np.float64),
#             "wavelength_m": np.asarray(wavelengths_m, dtype=np.float64),
#             "relative_amp_error": rel_amp_errors.astype(np.float64),
#             "avg_amplitudes_diff": avg_amp_diff,
#             "num_superposition_eval_samples": np.array([num_superposition_eval_samples], dtype=np.int32),
#             "superposition_eval_seed": np.array([superposition_eval_seed], dtype=np.int32),
#         },
#     )

#     print(f"✔ Wavelength sweep (relative amp error) plot saved -> {sweep_plot_path_linear}")
#     print(f"✔ Wavelength sweep data (.mat) -> {sweep_mat_path}")

# #%% Relative amp error vs. layer count 层和realtiv的曲线
# if evaluation_mode == "superposition" and all_amplitudes_relative_diff:
#     rel_err_dir = Path("metrics_analysis")
#     rel_err_dir.mkdir(parents=True, exist_ok=True)
#     rel_err_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     layer_counts_rel = np.asarray(num_layer_option[: len(all_amplitudes_relative_diff)], dtype=np.int32)
#     rel_amp_errors = np.asarray(all_amplitudes_relative_diff[: len(layer_counts_rel)], dtype=np.float64)

#     fig, ax = plt.subplots()
#     ax.plot(layer_counts_rel, rel_amp_errors, marker="o")
#     ax.set_xlabel("Number of layers")
#     ax.set_ylabel("Avg relative amplitude error")
#     ax.set_title(f"{num_superposition_eval_samples} superposition samples (seed={superposition_eval_seed})")
#     ax.grid(True, alpha=0.3)
#     rel_err_plot_path = rel_err_dir / f"relative_amp_error_superposition_layers_{rel_err_tag}.png"
#     fig.savefig(rel_err_plot_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     rel_err_mat_path = rel_err_dir / f"relative_amp_error_superposition_layers_{rel_err_tag}.mat"
#     savemat(
#         str(rel_err_mat_path),
#         {
#             "layers": layer_counts_rel.astype(np.float64),
#             "relative_amp_error": rel_amp_errors,
#             "num_superposition_eval_samples": np.array([num_superposition_eval_samples], dtype=np.int32),
#             "superposition_eval_seed": np.array([superposition_eval_seed], dtype=np.int32),
#         },
#     )

#     print(f"✔ Superposition relative amp error plot saved -> {rel_err_plot_path}")
#     print(f"✔ Superposition relative amp error data (.mat) -> {rel_err_mat_path}")


# #%% Mode isolation vs. layer count isolation的图
# if pred_case == 1 and evaluation_mode == "eigenmode" and all_phase_masks:
#     isolation_dir = Path("layer_isolation_analysis")
#     isolation_dir.mkdir(parents=True, exist_ok=True)
#     isolation_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     geometry = ModelGeometry(
#         layer_size=layer_size,
#         z_layers=z_layers,
#         z_prop=z_prop,
#         pixel_size=pixel_size,
#         z_input_to_first=z_input_to_first,
#     )

#     target_indices: list[int] = []
#     ideal_detection_percent: np.ndarray | None = None
#     ideal_target_percent: np.ndarray | None = None
#     if "label_data" in locals() and isinstance(label_data, torch.Tensor):
#         label_maps_np = label_data[:, 0].detach().cpu().numpy()
#         ideal_detection_percent_list: list[np.ndarray] = []
#         ideal_target_percent_list: list[float] = []
#         for sample_idx in range(label_maps_np.shape[0]):
#             energies, _ = spot_energy_ratios_circle(
#                 label_maps_np[sample_idx],
#                 evaluation_regions,
#                 radius=detectsize,
#                 offset=(0, 0),
#                 eps=1e-12,
#             )
#             det_sum = float(np.sum(energies)) + 1e-12
#             fractions = energies / det_sum
#             ideal_detection_percent_list.append(fractions * 100.0)
#             tgt_idx = int(np.argmax(energies))
#             target_indices.append(tgt_idx)
#             ideal_target_percent_list.append(float(fractions[tgt_idx] * 100.0))
#         ideal_detection_percent = np.stack(ideal_detection_percent_list, axis=0)
#         ideal_target_percent = np.asarray(ideal_target_percent_list, dtype=np.float64)
#     else:
#         target_indices = list(range(num_modes))
#         ideal_detection_percent = None
#         ideal_target_percent = np.full((num_modes,), 100.0, dtype=np.float64)

#     layer_counts: list[int] = []
#     isolation_db_mean_list: list[float] = []
#     isolation_ratio_mean_list: list[float] = []
#     isolation_db_samples: list[np.ndarray] = []
#     isolation_ratio_samples: list[np.ndarray] = []
#     detection_fraction_percent_list: list[np.ndarray] = []

#     for num_layer, phase_masks in zip(num_layer_option, all_phase_masks):
#         sweep_results = compute_mode_isolation_wavelength_sweep(
#             phase_masks,
#             base_wavelength=wavelength,
#             wavelength_list=[wavelength],
#             geometry=geometry,
#             device=device,
#             test_loader=test_loader,
#             evaluation_regions=evaluation_regions,
#             detect_radius=detectsize // 2,
#         )

#         layer_counts.append(num_layer)
#         isolation_db_mean_list.append(float(sweep_results["isolation_db_mean"][0]))
#         isolation_ratio_mean_list.append(float(sweep_results["isolation_ratio_mean"][0]))
#         isolation_db_samples.append(np.asarray(sweep_results["isolation_db"][0], dtype=np.float64))
#         isolation_ratio_samples.append(np.asarray(sweep_results["isolation_ratios"][0], dtype=np.float64))
#         detection_percent = np.asarray(sweep_results["detection_fractions"][0], dtype=np.float64) * 100.0
#         detection_fraction_percent_list.append(detection_percent)
#         isolation_percent_samples = isolation_ratio_samples[-1]
#         percentiles = np.percentile(isolation_percent_samples, [0, 25, 50, 75, 100])
#         print(
#             f"Layer {num_layer}: mean isolation = "
#             f"{isolation_db_mean_list[-1]:.2f} dB ({isolation_ratio_mean_list[-1]:.2f} %)"
#         )
#         print("  Target isolation per mode (expected -> predicted):")
#         for mode_idx in range(min(len(target_indices), isolation_percent_samples.size)):
#             expected_pct = float(ideal_target_percent[mode_idx]) if ideal_target_percent is not None else 100.0
#             predicted_pct = float(isolation_percent_samples[mode_idx])
#             predicted_db = float(isolation_db_samples[-1][mode_idx])
#             print(
#                 f"    Mode {mode_idx + 1}: {expected_pct:.2f}% -> "
#                 f"{predicted_pct:.2f}% ({predicted_db:.2f} dB)"
#             )
#         print(
#             "  Energy % stats -> "
#             f"min:{percentiles[0]:.2f}%, Q1:{percentiles[1]:.2f}%, "
#             f"median:{percentiles[2]:.2f}%, Q3:{percentiles[3]:.2f}%, max:{percentiles[4]:.2f}%"
#         )
#         total_samples = isolation_percent_samples.size
#         if total_samples <= 10:
#             preview_count = total_samples
#             prefix = f"Energy % (all {total_samples} samples)"
#         else:
#             preview_count = 10
#             prefix = f"Energy % (first {preview_count} of {total_samples} samples)"
#         preview_vals = ", ".join(f"{val:.2f}%" for val in isolation_percent_samples[:preview_count])
#         print(f"  {prefix}: {preview_vals}")
#         print("  Detection % by mask (normalized across masks):")
#         max_samples_to_show = detection_percent.shape[0] if detection_percent.shape[0] <= 12 else 12
#         for sample_idx in range(max_samples_to_show):
#             row = detection_percent[sample_idx]
#             row_str = ", ".join(f"M{mask_idx + 1}:{val:.2f}%" for mask_idx, val in enumerate(row))
#             print(f"    Sample {sample_idx + 1:02d}: {row_str}")
#         if detection_percent.shape[0] > max_samples_to_show:
#             print(f"    ... ({detection_percent.shape[0] - max_samples_to_show} more samples)")

#         fig_det, ax_det = plt.subplots(figsize=(6, max(3.0, 0.4 * detection_percent.shape[0])))
#         im = ax_det.imshow(detection_percent, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
#         ax_det.set_xlabel("Mask index")
#         ax_det.set_ylabel("Sample index")
#         ax_det.set_title(f"Layer {num_layer} mask energy distribution (%)")
#         cbar = fig_det.colorbar(im, ax=ax_det)
#         cbar.set_label("Energy (%)")
#         heatmap_path = isolation_dir / f"layer{num_layer}_mask_energy_{isolation_tag}.png"
#         fig_det.tight_layout()
#         fig_det.savefig(heatmap_path, dpi=300, bbox_inches="tight")
#         plt.close(fig_det)
#         print(f"  ✔ Mask energy heatmap saved -> {heatmap_path}")

#     isolation_percent_by_layer = np.stack(isolation_ratio_samples, axis=0)
#     isolation_db_by_layer = np.stack(isolation_db_samples, axis=0)
#     detection_percent_by_layer = np.stack(detection_fraction_percent_list, axis=0)
#     mean_isolation_percent_by_layer = isolation_percent_by_layer.mean(axis=1)
#     layer_counts_arr = np.asarray(layer_counts, dtype=np.int32)
#     isolation_db_mean_arr = np.asarray(isolation_db_mean_list, dtype=np.float64)

#     fig, ax = plt.subplots()
#     ax.plot(layer_counts_arr, isolation_db_mean_arr, marker="o")
#     ax.set_xlabel("Number of phase masks (layers)")
#     ax.set_ylabel("Mode isolation (dB)")
#     ax.set_title("Mode isolation vs. layer count")
#     ax.grid(True, alpha=0.3)

#     fig_pct, ax_pct = plt.subplots()
#     ax_pct.plot(layer_counts_arr, mean_isolation_percent_by_layer, marker="o", color="tab:purple")
#     ax_pct.set_xlabel("Number of phase masks (layers)")
#     ax_pct.set_ylabel("Target isolation (%)")
#     ax_pct.set_title("Target isolation vs. layer count")
#     ax_pct.set_ylim(0.0, 100.0)
#     ax_pct.grid(True, alpha=0.3)

#     isolation_plot_path = isolation_dir / f"layer_mode_isolation_{isolation_tag}.png"
#     isolation_plot_pct_path = isolation_dir / f"layer_mode_isolation_percent_{isolation_tag}.png"
#     fig.savefig(isolation_plot_path, dpi=300, bbox_inches="tight")
#     fig_pct.savefig(isolation_plot_pct_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     plt.close(fig_pct)

#     isolation_mat_path = isolation_dir / f"layer_mode_isolation_{isolation_tag}.mat"
#     mat_payload = {
#         "layers": layer_counts_arr.astype(np.float64),
#         "isolation_db_mean": isolation_db_mean_arr,
#         "isolation_ratio_mean": np.asarray(isolation_ratio_mean_list, dtype=np.float64),
#         "isolation_db_by_layer": isolation_db_by_layer,
#         "isolation_percent_by_layer": isolation_percent_by_layer,
#         "mean_isolation_percent": mean_isolation_percent_by_layer,
#         "detection_fraction_percent": detection_percent_by_layer,
#         "target_indices": np.asarray(target_indices, dtype=np.int32),
#     }
#     if ideal_detection_percent is not None:
#         mat_payload["ideal_detection_percent"] = ideal_detection_percent.astype(np.float64)
#         mat_payload["ideal_target_percent"] = ideal_target_percent.astype(np.float64)
#     else:
#         mat_payload["ideal_detection_percent"] = np.array([], dtype=np.float64)
#         mat_payload["ideal_target_percent"] = np.asarray(ideal_target_percent, dtype=np.float64)

#     savemat(str(isolation_mat_path), mat_payload)

#     print(f"✔ Layer-count isolation plot saved -> {isolation_plot_path}")
#     print(f"✔ Layer-count isolation plot saved -> {isolation_plot_pct_path}")
#     print(f"✔ Layer-count isolation data (.mat) -> {isolation_mat_path}")


# #%% Wavelength sweep: per-mode isolation (3-layer model)
# if (
#     pred_case == 1
#     and evaluation_mode == "eigenmode"
#     and all_phase_masks
#     and 3 in num_layer_option
# ):
#     wl_dir = Path("wavelength_analysis")
#     wl_dir.mkdir(parents=True, exist_ok=True)
#     wl_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     geometry = ModelGeometry(
#         layer_size=layer_size,
#         z_layers=z_layers,
#         z_prop=z_prop,
#         pixel_size=pixel_size,
#         z_input_to_first=z_input_to_first,
#     )

#     layer3_idx = num_layer_option.index(3)
#     phase_masks_L3 = all_phase_masks[layer3_idx]

#     wavelengths_nm_target = np.arange(1500.0, 1650.0 + 1e-6, 1.0, dtype=np.float64)
#     wavelengths_m_target = wavelengths_nm_target * 1e-9

#     sweep_results = compute_mode_isolation_wavelength_sweep(
#         phase_masks_L3,
#         base_wavelength=wavelength,
#         wavelength_list=wavelengths_m_target,
#         geometry=geometry,
#         device=device,
#         test_loader=test_loader,
#         evaluation_regions=evaluation_regions,
#         detect_radius=detectsize // 2,
#     )

#     det_frac = np.asarray(sweep_results["detection_fractions"], dtype=np.float64)
#     num_lambda = det_frac.shape[0]
#     num_samples = det_frac.shape[1]

#     target_indices_wl: list[int] = []
#     if "label_data" in locals() and isinstance(label_data, torch.Tensor):
#         label_maps_np = label_data[:, 0].detach().cpu().numpy()
#         for sample_idx in range(min(num_samples, label_maps_np.shape[0])):
#             energies, _ = spot_energy_ratios_circle(
#                 label_maps_np[sample_idx],
#                 evaluation_regions,
#                 radius=detectsize // 2,
#                 offset=(0, 0),
#                 eps=1e-12,
#             )
#             tgt_idx = int(np.argmax(energies))
#             target_indices_wl.append(tgt_idx)
#     else:
#         target_indices_wl = [min(i, num_detector - 1) for i in range(num_samples)]

#     isolation_percent = np.zeros((num_lambda, num_modes), dtype=np.float64)
#     isolation_db = np.zeros_like(isolation_percent)

#     for lam_idx in range(num_lambda):
#         for mode_idx in range(num_modes):
#             sample_idx = mode_idx if mode_idx < num_samples else mode_idx % num_samples
#             tgt_idx = target_indices_wl[sample_idx] if sample_idx < len(target_indices_wl) else mode_idx
#             frac = float(det_frac[lam_idx, sample_idx, tgt_idx])
#             frac = np.clip(frac, 1e-12, 1.0)
#             isolation_percent[lam_idx, mode_idx] = frac * 100.0
#             ratio_vs_rest = frac / max(1e-12, (1.0 - frac))
#             isolation_db[lam_idx, mode_idx] = 10.0 * np.log10(ratio_vs_rest)

#     fig_wl, ax_wl = plt.subplots()
#     for mode_idx in range(num_modes):
#         ax_wl.plot(
#             wavelengths_nm_target,
#             isolation_db[:, mode_idx],
#             label=f"Mode {mode_idx + 1}",
#         )
#     ax_wl.set_xlabel("Wavelength (nm)")
#     ax_wl.set_ylabel("Isolation (dB)")
#     ax_wl.set_title("Per-mode isolation vs. wavelength (3-layer)")
#     ax_wl.grid(True, alpha=0.3)
#     ax_wl.legend(title="Modes", ncol=2)

#     wl_plot_path = wl_dir / f"per_mode_isolation_L3_{wl_tag}.png"
#     fig_wl.savefig(wl_plot_path, dpi=300, bbox_inches="tight")
#     plt.close(fig_wl)

#     wl_mat_path = wl_dir / f"per_mode_isolation_L3_{wl_tag}.mat"
#     savemat(
#         str(wl_mat_path),
#         {
#             "wavelength_nm": wavelengths_nm_target,
#             "wavelength_m": wavelengths_m_target,
#             "isolation_percent": isolation_percent,
#             "isolation_db": isolation_db,
#             "target_indices": np.asarray(target_indices_wl, dtype=np.int32),
#             "detect_radius_px": np.array([detectsize // 2], dtype=np.int32),
#             "num_modes": np.array([num_modes], dtype=np.int32),
#         },
#     )

#     print(f"✔ Per-mode isolation (3-layer) plot saved -> {wl_plot_path}")
#     print(f"✔ Per-mode isolation (3-layer) data (.mat) -> {wl_mat_path}")

#     # Optional debug visuals across wavelengths
#     if enable_wavelength_debug_visuals:
#         wl_debug_dir = wl_dir / f"debug_{wl_tag}"
#         wl_debug_dir.mkdir(parents=True, exist_ok=True)

#         for lam_nm in debug_wavelengths_nm:
#             lam_m = lam_nm * 1e-9
#             scaled_masks = scale_phase_masks_for_wavelength(
#                 phase_masks_L3,
#                 base_wavelength=wavelength,
#                 target_wavelength=lam_m,
#             )
#             phase_min = min(float(mask.min()) for mask in scaled_masks)
#             phase_max = max(float(mask.max()) for mask in scaled_masks)
#             print(
#                 f"[WL debug] λ={lam_nm:.1f} nm -> phase range: "
#                 f"{phase_min:.3f} to {phase_max:.3f} (rad)"
#             )

#             dbg_model = D2NNModel(
#                 num_layers=3,
#                 layer_size=layer_size,
#                 z_layers=z_layers,
#                 z_prop=z_prop,
#                 pixel_size=pixel_size,
#                 wavelength=lam_m,
#                 device=device,
#                 padding_ratio=0.5,
#                 z_input_to_first=z_input_to_first,
#             ).to(device)
#             for layer, mask in zip(dbg_model.layers, scaled_masks):
#                 mask_t = torch.from_numpy(mask.astype(np.float32, copy=False)).to(device)
#                 layer.phase.data.copy_(mask_t)
#             dbg_model.eval()

#             for mode_idx in debug_modes_to_plot:
#                 if mode_idx >= len(MMF_data_ts):
#                     continue
#                 _ = save_mode_triptych(
#                     model=dbg_model,
#                     mode_index=mode_idx,
#                     eigenmode_field=MMF_data_ts[mode_idx],
#                     label_field=label_data[mode_idx, 0] if isinstance(label_data, torch.Tensor) else label_data[mode_idx][1][0],
#                     layer_size=layer_size,
#                     output_dir=wl_debug_dir,
#                     tag=f"L3_{lam_nm:.1f}nm",
#                     evaluation_regions=evaluation_regions,
#                     detect_radius=detectsize,
#                     show_mask_overlays=True,
#                 )

#             _ = save_prediction_diagnostics(
#                 dbg_model,
#                 test_dataset,
#                 evaluation_regions=evaluation_regions,
#                 layer_size=layer_size,
#                 detect_radius=detectsize,
#                 num_samples=min(3, len(test_dataset)),
#                 output_dir=wl_debug_dir,
#                 device=device,
#                 tag=f"L3_{lam_nm:.1f}nm",
#             )


# #%%存结果画图
# if all_training_summaries:
#     print("\nTraining duration summary:")
#     for summary in all_training_summaries:
#         minutes = summary["total_time"] / 60
#         print(
#             f" - {summary['num_layers']} layers: {summary['total_time']:.2f} s "
#             f"(~{minutes:.2f} min)"
#         )
#         print(f"   Loss curve: {summary['loss_plot']}")
#         print(f"   Time curve: {summary['time_plot']}")
#         print(f"   Data (.mat): {summary['mat_path']}")
#         print(f"   Propagation plot: {summary['propagation_fig']}")
#         print(f"   Propagation data (.mat): {summary['propagation_mat']}")
#         mode_triptychs = summary.get("mode_triptychs", [])
#         if mode_triptychs:
#             print("   Mode triptychs:")
#             for trip in mode_triptychs:
#                 print(
#                     f"     Mode {trip['mode']}: fig={trip['fig']}, mat={trip['mat']}"
#                 )

# save_dir = "plots"
# os.makedirs(save_dir, exist_ok=True)
# num_samples_to_display = 6
# for idx, num_layer in enumerate(num_layer_option):
#     plot_amplitude_comparison_grid(
#         image_test_data,
#         all_image_data_pred[idx],
#         all_cc_recon_amp[idx],
#         max_samples=num_samples_to_display,
#         save_path=os.path.join(save_dir, f"Amp_{num_layer}layers.png"),
#         title=f"Amp. distribution of Real and Predicted Images({num_layer}_layer_ODNN)",
#     )

# #%% 直观的看看输出和label的差异
# for s in [0, 1, 2, 5]:
#     plot_sys_vs_label_strict(
#         D2NN,
#         test_dataset,
#         sample_idx=s,
#         evaluation_regions=evaluation_regions,
#         detect_radius=detectsize,
#         save_path=f"plots/IO_Pred_Label_RAW_{s}.png",
#         device=device,
#         use_big_canvas=False,
#         sys_scale="bg_pct",
#         sys_pct=99.5,
#         clip_pct=99.5,
#         mask_roi_for_scale=True,
#         show_signed=True,
#     )
#     plot_reconstruction_vs_input(
#         image_test_data=image_test_data,
#         reconstructed_fields=all_image_data_pred,
#         sample_idx=s,
#         model_idx=0,
#         save_path=f"plots/Reconstruction_vs_Input_{s}.png",
#     )


#%% Propagation slices & mask export，看一个固定的输入的切片输出
# temp_dataset = test_dataset
# FIXED_E_INDEX = 4

# def get_fixed_input(dataset, idx, device):
#     if isinstance(dataset, list):
#         sample = dataset[idx][0]
#     else:
#         sample = dataset.tensors[0][idx]
#     return sample.squeeze(0).to(device)


# assert len(temp_dataset) > 0, "test_dataset 为空"
# if prop_slice_sample_mode == "random":
#     rng = np.random.default_rng(prop_slice_seed)
#     sample_idx = int(rng.integers(low=0, high=len(temp_dataset)))
# else:
#     sample_idx = FIXED_E_INDEX % len(temp_dataset)
# temp_E = get_fixed_input(temp_dataset, sample_idx, device)
# print(f"[Slices] Using sample #{sample_idx} for propagation snapshots (mode={prop_slice_sample_mode})")

# z_start = 0.0
# z_step = 5e-6
# z_prop_plus = z_prop

# save_root = Path("results_MD")
# save_root.mkdir(parents=True, exist_ok=True)
# run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename_prefix = f"ODNN_vis_{run_stamp}"

# phase_mask_entries: list[tuple[int, list[np.ndarray]]] = []
# if all_phase_masks:
#     phase_mask_entries.append((len(all_phase_masks), all_phase_masks[-1]))

# for i_model, phase_masks in phase_mask_entries:
#     model_dir = save_root / f"m{i_model}"
#     scans, camera_field = visualize_model_slices(
#         D2NN,
#         phase_masks,
#         temp_E,
#         output_dir=model_dir,
#         sample_tag=f"m{i_model}",
#         z_input_to_first=z_input_to_first,
#         z_layers=z_layers,
#         z_prop_plus=z_prop_plus,
#         z_step=z_step,
#         pixel_size=pixel_size,
#         wavelength=wavelength,
#         kmax=prop_scan_kmax,
#     )

#     phase_stack = np.stack([np.asarray(mask, dtype=np.float32) for mask in phase_masks], axis=0)
#     meta = {
#         "z_start": float(z_start),
#         "z_step": float(z_step),
#         "z_layers": float(z_layers),
#         "z_prop": float(z_prop),
#         "z_prop_plus": float(z_prop_plus),
#         "pixel_size": float(pixel_size),
#         "wavelength": float(wavelength),
#         "layer_size": int(layer_size),
#         "padding_ratio": 0.5,
#     }

#     mat_path = model_dir / f"{filename_prefix}_LIGHT_m{i_model}.mat"
    # save_to_mat_light_plus(
    #     mat_path,
    #     phase_stack=phase_stack,
    #     input_field=temp_E.detach().cpu().numpy(),
    #     scans=scans,
    #     camera_field=camera_field,
    #     sample_stacks_kmax=20,
    #     save_amplitude_only=False,
    #     meta=meta,
    # )
    # print("Saved ->", mat_path)

    # save_masks_one_file_per_layer(
    #     phase_masks,
    #     out_dir=model_dir,
    #     base_name=f"{filename_prefix}_MASK",
    #     save_degree=False,
    #     use_xlsx=True,
    # )



# #%% Superposition
# z_step = 5e-6
# if pred_case == 1 and run_superposition_debug:
#     super_dir = Path("results_superposition")
#     super_dir.mkdir(parents=True, exist_ok=True)
#     super_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
#     super_records: list[dict[str, str | int]] = []
#     slice_reference_input: torch.Tensor | None = None

#     for sample_idx in range(num_superposition_visual_samples):
#         super_sample = generate_superposition_sample(
#             num_modes=num_modes,
#             field_size=field_size,
#             layer_size=layer_size,
#             mmf_modes=MMF_data_ts,
#             mmf_label_data=MMF_Label_data,
#         )
#         super_output_map = infer_superposition_output(
#             D2NN,
#             super_sample["padded_image"],
#             device,
#         )

#         sample_tag = f"{super_tag}_s{sample_idx:02d}"
#         triptych_paths = save_superposition_triptych(
#             input_field=super_sample["padded_image"][0],
#             output_intensity_map=super_output_map,
#             amplitudes=super_sample["amplitudes"],
#             phases=super_sample["phases"],
#             complex_weights=super_sample["complex_weights"],
#             label_map=super_sample["padded_label"][0],
#             evaluation_regions=evaluation_regions,
#             detect_radius=detectsize,
#             output_dir=super_dir,
#             tag=sample_tag,
#             save_plot=save_superposition_plots,
#         )
#         if triptych_paths["fig_path"]:
#             print(
#                 f"Superposition sample {sample_idx + 1}/{num_superposition_visual_samples} -> "
#                 f"{triptych_paths['fig_path']}"
#             )
#         print(f"  MAT saved -> {triptych_paths['mat_path']}")

#         super_records.append(
#             {
#                 "index": sample_idx,
#                 "tag": sample_tag,
#                 "fig": triptych_paths["fig_path"] if triptych_paths else "",
#                 "mat": triptych_paths["mat_path"] if triptych_paths else "",
#             }
#         )

#         if slice_reference_input is None:
#             slice_reference_input = (
#                 super_sample["padded_image"].squeeze(0).to(device, dtype=torch.complex64)
#             )

#     if save_superposition_slices and all_phase_masks and slice_reference_input is not None:
#         slices_root = super_dir / f"slices_{super_tag}"
#         export_superposition_slices(
#             D2NN,
#             all_phase_masks,
#             slice_reference_input,
#             slices_root,
#             sample_tag="superposition",
#             z_input_to_first=z_input_to_first,
#             z_layers=z_layers,
#             z_prop=z_prop,
#             z_step=z_step,
#             pixel_size=pixel_size,
#             wavelength=wavelength,
#         )

#     if super_records:
#         print("\nSuperposition sample outputs:")
#         for record in super_records:
#             print(
#                 f" - Sample {record['index'] + 1:02d} ({record['tag']}): "
#                 f"fig={record['fig']}, mat={record['mat']}"
#             )



# #%% 第一层mask位移

# if run_misalignment_robustness and pred_case == 1:
#     robustness_dir = Path("robustness_analysis")
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


# #%% Wavelength sweep analysis
# if pred_case == 1 and all_phase_masks:
#     wavelength_sweep_dir = Path("wavelength_analysis")
#     wavelength_sweep_dir.mkdir(parents=True, exist_ok=True)
#     sweep_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

#     base_phase_masks = all_phase_masks[-1]
#     geometry = ModelGeometry(
#         layer_size=layer_size,
#         z_layers=z_layers,
#         z_prop=z_prop,
#         pixel_size=pixel_size,
#         z_input_to_first=z_input_to_first,
#     )

#     wavelengths_nm_target = np.arange(1500.0, 1700.0 + 1e-6, 1.0, dtype=np.float64)
#     wavelengths_m_target = wavelengths_nm_target * 1e-9

#     if wavelength_sweep_mode not in {"auto", "mode_isolation", "relative_amp_error"}:
#         raise ValueError(f"Unknown wavelength_sweep_mode: {wavelength_sweep_mode}")
#     if wavelength_relative_eval_mode not in {"match", "superposition"}:
#         raise ValueError(f"Unknown wavelength_relative_eval_mode: {wavelength_relative_eval_mode}")

#     sweep_mode_choice = wavelength_sweep_mode
#     if sweep_mode_choice == "auto":
#         sweep_mode_choice = "mode_isolation" if evaluation_mode == "eigenmode" else "relative_amp_error"

#     if sweep_mode_choice == "mode_isolation":
#         sweep_results = compute_mode_isolation_wavelength_sweep(
#             base_phase_masks,
#             base_wavelength=wavelength,
#             wavelength_list=wavelengths_m_target,
#             geometry=geometry,
#             device=device,
#             test_loader=test_loader,
#             evaluation_regions=evaluation_regions,
#             detect_radius=detectsize,
#         )

#         wavelengths_m = np.asarray(sweep_results["wavelengths"], dtype=np.float64)
#         wavelengths_nm = wavelengths_m * 1e9
#         isolation_db_mean = np.asarray(sweep_results["isolation_db_mean"], dtype=np.float64)
#         isolation_ratio_mean = np.asarray(sweep_results["isolation_ratio_mean"], dtype=np.float64)
#         isolation_db = np.asarray(sweep_results["isolation_db"], dtype=np.float64)
#         isolation_ratios = np.asarray(sweep_results["isolation_ratios"], dtype=np.float64)
#         detection_fractions = np.asarray(sweep_results["detection_fractions"], dtype=np.float64)
#         detection_percent = detection_fractions * 100.0

#         log_count = min(10, len(wavelengths_nm))
#         if log_count > 0:
#             log_indices = np.linspace(0, len(wavelengths_nm) - 1, log_count, dtype=int)
#             log_indices = np.unique(log_indices)
#         else:
#             log_indices = np.array([], dtype=int)

#         for idx in log_indices:
#             lam_nm = wavelengths_nm[idx]
#             isolation_percent_samples = isolation_ratios[idx]
#             percentiles = np.percentile(isolation_percent_samples, [0, 25, 50, 75, 100])
#             print(
#                 f"Wavelength {lam_nm:.1f} nm: mean isolation = "
#                 f"{isolation_db_mean[idx]:.2f} dB ({isolation_ratio_mean[idx]:.2f} %)"
#             )
#             print(
#                 "  Energy % stats -> "
#                 f"min:{percentiles[0]:.2f}%, Q1:{percentiles[1]:.2f}%, "
#                 f"median:{percentiles[2]:.2f}%, Q3:{percentiles[3]:.2f}%, max:{percentiles[4]:.2f}%"
#             )

#             total_samples = isolation_percent_samples.size
#             if total_samples <= 10:
#                 preview_count = total_samples
#                 prefix = f"Energy % (all {total_samples} samples)"
#             else:
#                 preview_count = 10
#                 prefix = f"Energy % (first {preview_count} of {total_samples} samples)"
#             preview_vals = ", ".join(
#                 f"{val:.2f}%" for val in isolation_percent_samples[:preview_count]
#             )
#             print(f"  {prefix}: {preview_vals}")

#             det_percent = detection_percent[idx]
#             print("  Detection % by mask (normalized across masks):")
#             max_samples_to_show = det_percent.shape[0] if det_percent.shape[0] <= 12 else 12
#             for sample_idx in range(max_samples_to_show):
#                 row = det_percent[sample_idx]
#                 row_str = ", ".join(f"M{mask_idx + 1}:{val:.2f}%" for mask_idx, val in enumerate(row))
#                 print(f"    Sample {sample_idx + 1:02d}: {row_str}")
#             if det_percent.shape[0] > max_samples_to_show:
#                 remaining = det_percent.shape[0] - max_samples_to_show
#                 print(f"    ... ({remaining} more samples)")

#             heatmap_fig, ax_det = plt.subplots(
#                 figsize=(6, max(3.0, 0.4 * det_percent.shape[0]))
#             )
#             im = ax_det.imshow(det_percent, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
#             ax_det.set_xlabel("Mask index")
#             ax_det.set_ylabel("Sample index")
#             ax_det.set_title(f"{lam_nm:.1f} nm mask energy distribution (%)")
#             cbar = heatmap_fig.colorbar(im, ax=ax_det)
#             cbar.set_label("Energy (%)")
#             heatmap_path = wavelength_sweep_dir / f"wavelength_{lam_nm:.1f}nm_mask_energy_{sweep_tag}.png"
#             heatmap_fig.tight_layout()
#             heatmap_fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
#             plt.close(heatmap_fig)
#             print(f"  ✔ Mask energy heatmap saved -> {heatmap_path}")

#         fig, ax = plt.subplots()
#         ax.plot(wavelengths_nm, isolation_db_mean, marker="o")
#         ax.set_xlabel("Wavelength (nm)")
#         ax.set_ylabel("Mode isolation (dB)")
#         ax.set_title("Mode isolation vs. wavelength")
#         ax.grid(True, alpha=0.3)

#         sweep_plot_path = wavelength_sweep_dir / f"wavelength_mode_isolation_{sweep_tag}.png"
#         fig.savefig(sweep_plot_path, dpi=300, bbox_inches="tight")
#         plt.close(fig)

#         sweep_mat_path = wavelength_sweep_dir / f"wavelength_mode_isolation_{sweep_tag}.mat"
#         savemat(
#             str(sweep_mat_path),
#             {
#                 "wavelength_nm": wavelengths_nm.astype(np.float64),
#                 "wavelength_m": wavelengths_m.astype(np.float64),
#                 "isolation_ratio": isolation_ratios.astype(np.float64),
#                 "isolation_db": isolation_db.astype(np.float64),
#                 "isolation_ratio_mean": isolation_ratio_mean.astype(np.float64),
#                 "isolation_db_mean": isolation_db_mean.astype(np.float64),
#                 "detection_fraction_percent": detection_percent.astype(np.float64),
#             },
#         )
#     elif sweep_mode_choice == "relative_amp_error":
#         sweep_loader = test_loader
#         sweep_image_data = image_test_data
#         sweep_amplitudes = eval_amplitudes
#         sweep_amplitudes_phases = eval_amplitudes_phases
#         sweep_phases = eval_phases

#         if wavelength_relative_eval_mode == "superposition":
#             reuse_existing_ctx = (
#                 superposition_eval_ctx is not None
#                 and wavelength_relative_eval_samples == num_superposition_eval_samples
#                 and wavelength_relative_eval_seed == superposition_eval_seed
#             )
#             if reuse_existing_ctx:
#                 sweep_ctx = superposition_eval_ctx
#             else:
#                 sweep_ctx = build_superposition_eval_context(
#                     wavelength_relative_eval_samples,
#                     num_modes=num_modes,
#                     field_size=field_size,
#                     layer_size=layer_size,
#                     mmf_modes=MMF_data_ts,
#                     mmf_label_data=MMF_Label_data,
#                     batch_size=batch_size,
#                     second_mode_half_range=True,
#                     rng_seed=wavelength_relative_eval_seed,
#                 )
#             sweep_loader = sweep_ctx["loader"]
#             sweep_image_data = sweep_ctx["image_data"]
#             sweep_amplitudes = sweep_ctx["amplitudes"]
#             sweep_amplitudes_phases = sweep_ctx["amplitudes_phases"]
#             sweep_phases = sweep_ctx["phases"]

#         wavelengths_m, rel_amp_errors, sweep_metrics = compute_relative_amp_error_wavelength_sweep(
#             base_phase_masks,
#             base_wavelength=wavelength,
#             wavelength_list=wavelengths_m_target,
#             geometry=geometry,
#             device=device,
#             test_loader=sweep_loader,
#             evaluation_regions=evaluation_regions,
#             detect_radius=detectsize,
#             pred_case=pred_case,
#             num_modes=num_modes,
#             phase_option=phase_option,
#             amplitudes=sweep_amplitudes,
#             amplitudes_phases=sweep_amplitudes_phases,
#             phases=sweep_phases,
#             mmf_modes=MMF_data_ts,
#             field_size=field_size,
#             image_test_data=sweep_image_data,
#         )

#         rel_amp_errors = np.asarray(rel_amp_errors, dtype=np.float64)
#         rel_amp_error_db = 20.0 * np.log10(np.clip(rel_amp_errors, 1e-12, None))
#         wavelengths_nm = np.asarray(wavelengths_m, dtype=np.float64) * 1e9
#         avg_amp_diff = np.array(
#             [float(metrics.get("avg_amplitudes_diff", float("nan"))) for metrics in sweep_metrics],
#             dtype=np.float64,
#         )

#         # Linear-scale relative amplitude error plot
#         fig_linear, ax_linear = plt.subplots()
#         ax_linear.plot(wavelengths_nm, rel_amp_errors, marker="o")
#         ax_linear.set_xlabel("Wavelength (nm)")
#         ax_linear.set_ylabel("Relative amplitude error")
#         ax_linear.set_title("Relative amplitude error vs. wavelength (linear)")
#         ax_linear.grid(True, alpha=0.3)

#         sweep_plot_path_linear = wavelength_sweep_dir / f"wavelength_relative_error_linear_{sweep_tag}.png"
#         fig_linear.savefig(sweep_plot_path_linear, dpi=300, bbox_inches="tight")
#         plt.close(fig_linear)

#         fig, ax = plt.subplots()
#         ax.plot(wavelengths_nm, rel_amp_error_db, marker="o")
#         ax.set_xlabel("Wavelength (nm)")
#         ax.set_ylabel("Relative amplitude error (dB)")
#         ax.set_title("Relative amplitude error vs. wavelength (dB)")
#         ax.grid(True, alpha=0.3)

#         sweep_plot_path = wavelength_sweep_dir / f"wavelength_relative_error_{sweep_tag}.png"
#         fig.savefig(sweep_plot_path, dpi=300, bbox_inches="tight")
#         plt.close(fig)

#         sweep_mat_path = wavelength_sweep_dir / f"wavelength_relative_error_{sweep_tag}.mat"
#         savemat(
#             str(sweep_mat_path),
#             {
#                 "wavelength_nm": wavelengths_nm.astype(np.float64),
#                 "wavelength_m": np.asarray(wavelengths_m, dtype=np.float64),
#                 "relative_amp_error": rel_amp_errors.astype(np.float64),
#                 "relative_amp_error_db": rel_amp_error_db.astype(np.float64),
#                 "avg_amplitudes_diff": avg_amp_diff,
#             },
#         )
#     else:
#         raise ValueError(f"Unhandled sweep mode: {sweep_mode_choice}")

#     if sweep_mode_choice == "relative_amp_error":
#         print(f"✔ Wavelength sweep plot saved -> {sweep_plot_path_linear}")
#         print(f"✔ Wavelength sweep plot saved -> {sweep_plot_path}")
#     else:
#         print(f"✔ Wavelength sweep plot saved -> {sweep_plot_path}")
#     print(f"✔ Wavelength sweep data (.mat) -> {sweep_mat_path}")

# %%

# %%
