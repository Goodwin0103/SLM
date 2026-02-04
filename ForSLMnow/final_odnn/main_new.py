#%%
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
import numpy as np
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
    evaluate_spot_metrics,
    format_metric_report,
    save_prediction_diagnostics,
)
from odnn_training_visualization import (
    capture_eigenmode_propagation,
    save_mode_triptych,
)

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
    device = torch.device('cuda:2')
    print('Using Device:', device)
else:
    device = torch.device('cpu')
    print('Using Device: CPU')


#%% Parameters
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
num_superposition_visual_samples = 2
label_pattern_mode = "circle"  # options: "eigenmode", "circle"
superposition_eval_seed = 20240116
show_detection_overlap_debug = True
detection_overlap_label_index = 0

prop_slices_per_segment = 10
prop_output_slices = 10

training_dataset_mode = "eigenmode"  # options: "eigenmode", "superposition"
num_superposition_train_samples = 100
superposition_train_seed = 20240115

num_layer_option = [2, 3, 4, 5, 6]
all_losses = []
all_phase_masks = []
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

# SLM parameters
z_layers = 40e-6
pixel_size = 1e-6
z_prop = 120e-6
wavelength = 1568e-9
z_input_to_first = 40e-6

phase_option = 4

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


def build_uniform_fractions(count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    fractions = np.linspace(1.0 / (count + 1), count / (count + 1), count, dtype=float)
    return tuple(float(f) for f in fractions)


#%% Load eigenmode data
eigenmodes_OM4 = load_complex_modes_from_mat(
    'mmf_103modes_25_PD_1.15.mat',
    key='modes_field'
)
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]

#%% Generate labels
pred_case = 1
label_size = layer_size

if pred_case == 1:
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
    energy_weights = amplitude_weights**2
    combined_labels = (
        energy_weights[:, None, None, :] * MMF_Label_data.unsqueeze(0)
    ).sum(dim=3)
    label_data[:, 0, :, :] = combined_labels

    complex_weights = amplitudes * np.exp(1j * phases)
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

#%% Generate detection regions
if pred_case == 1:
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

        label_sample_np = None
        if isinstance(label_data, torch.Tensor) and label_data.shape[0] > 0:
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

#%% Train models
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
        z_input_to_first=z_input_to_first,
    ).to(device)

    print(D2NN)

    criterion = nn.MSELoss()
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
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = D2NN(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
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
    all_losses.append(losses)
    
    # Save training curves
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

    # Capture propagation
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
    if energies.size > 0 and energies[0] != 0:
        energy_drop_pct = (energies[0] - energies[-1]) / energies[0] * 100.0
        print(
            f"   Energy trace: start={energies[0]:.4e}, end={energies[-1]:.4e}, "
            f"drop={energy_drop_pct:.2f}% over {energies.size} slices"
        )

    # Save mode triptychs (only for eigenmode evaluation)
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

    # Save checkpoint
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {
        "state_dict": D2NN.state_dict(),
        "meta": {
            "num_layers": len(D2NN.layers),
            "layer_size": layer_size,
            "z_layers": z_layers,
            "z_prop": z_prop,
            "pixel_size": pixel_size,
            "wavelength": wavelength,
            "padding_ratio": 0.5,
            "field_size": field_size,
            "num_modes": num_modes,
            "z_input_to_first": z_input_to_first,
        }
    }
    save_path = os.path.join(ckpt_dir, f"odnn_{len(D2NN.layers)}layers_m{num_modes}_ls{layer_size}.pth")
    torch.save(ckpt, save_path)
    print("✔ Saved model ->", save_path)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Cache phase masks
    phase_masks = []
    for layer in D2NN.layers:
        phase_np = layer.phase.detach().cpu().numpy()
        phase_masks.append(np.remainder(phase_np, 2 * np.pi))
    all_phase_masks.append(phase_masks)

    # Evaluate model
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

    # Save prediction diagnostics
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

    # Collect metrics
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

    layer_counts = np.asarray(num_layer_option[:len(model_metrics)], dtype=np.int32)
    amp_err = np.asarray(all_average_amplitudes_diff[:len(layer_counts)], dtype=np.float64)
    amp_err_rel = np.asarray(all_amplitudes_relative_diff[:len(layer_counts)], dtype=np.float64)

    cc_amp_mean_list: list[float] = []
    cc_amp_std_list: list[float] = []
    for cc_arr in all_cc_recon_amp[:len(layer_counts)]:
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

# %%
