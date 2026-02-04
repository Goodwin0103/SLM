from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_modes_amp_phase(modes: np.ndarray, output_path: Path) -> None:
    
    num_modes = modes.shape[0]
    mmf_amp = np.abs(modes)
    mmf_phase = np.angle(modes)

    fig, axes = plt.subplots(2, num_modes, figsize=(3.5 * num_modes, 6), squeeze=False)
    for idx in range(num_modes):
        im_amp = axes[0, idx].imshow(mmf_amp[idx], cmap="turbo")
        axes[0, idx].set_title(f"Mode {idx + 1} |E|")
        axes[0, idx].axis("off")
        fig.colorbar(im_amp, ax=axes[0, idx], fraction=0.046, pad=0.02)

        im_phase = axes[1, idx].imshow(mmf_phase[idx], cmap="turbo")
        axes[1, idx].set_title(f"Mode {idx + 1} ∠E")
        axes[1, idx].axis("off")
        fig.colorbar(im_phase, ax=axes[1, idx], fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_field_amp_phase(field: np.ndarray, output_path: Path, title_prefix: str = "Field") -> None:
   
    amplitude = np.abs(field)
    phase = np.angle(field)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im_amp = axes[0].imshow(amplitude, cmap="turbo")
    axes[0].set_title(f"{title_prefix} Amplitude")
    axes[0].axis("off")
    fig.colorbar(im_amp, ax=axes[0], fraction=0.046, pad=0.04)

    im_phase = axes[1].imshow(phase, cmap="turbo")
    axes[1].set_title(f"{title_prefix} Phase")
    axes[1].axis("off")
    fig.colorbar(im_phase, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_best_cv_comparison(
    field_input: np.ndarray,
    output_intensity: np.ndarray,
    amplitude_target: np.ndarray,
    weights_pred: np.ndarray,
    field_reconstructed: np.ndarray,
    output_path: Path,
) -> None:
   
    indices = np.arange(1, amplitude_target.size + 1)
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(np.abs(field_input), cmap="turbo")
    axes[0, 0].set_title("Network Input |E|")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(output_intensity, cmap="turbo")
    axes[0, 1].set_title("Network Output (Intensity)")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].bar(indices - width / 2, amplitude_target, width, label="Target |best_cv|", color="tab:green")
    axes[1, 0].bar(indices + width / 2, weights_pred, width, label="Predicted (L2-normalized sum)", color="tab:blue")
    axes[1, 0].set_xlabel("Mode index")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_xticks(indices)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].set_title("Amplitude Comparison")
    axes[1, 0].text(
        0.02,
        0.9,
        f"∑pred²={np.sum(weights_pred**2):.3f}",
        transform=axes[1, 0].transAxes,
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0),
    )

    im3 = axes[1, 1].imshow(np.abs(field_reconstructed), cmap="turbo")
    axes[1, 1].set_title("Reconstructed Field (|amp_pred| + ∠best_cv)")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

