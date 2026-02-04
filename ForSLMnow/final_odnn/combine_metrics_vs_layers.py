#!/usr/bin/env python3
"""
Combine multiple metrics_vs_layers_*.mat files into one plot.

Usage:
    python combine_metrics_vs_layers.py --mats results/metrics_vs_layers_*.mat --out combined_metrics_vs_layers.png
If --mats is omitted, defaults to glob("results/metrics_vs_layers_*.mat").
"""

from __future__ import annotations
import argparse
import glob
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_metric_file(path: Path) -> dict:
    data = loadmat(path)
    # Expected keys saved by mainfor6.py
    layers = np.squeeze(data.get("layers", np.array([]))).astype(float)
    avg_amp_error = np.squeeze(data.get("avg_amp_error", np.array([]))).astype(float)
    avg_rel_amp_error = np.squeeze(data.get("avg_relative_amp_error", np.array([]))).astype(float)
    cc_amp_mean = np.squeeze(data.get("cc_amp_mean", np.array([]))).astype(float)
    cc_amp_std = np.squeeze(data.get("cc_amp_std", np.array([]))).astype(float)
    return {
        "layers": layers,
        "avg_amp_error": avg_amp_error,
        "avg_relative_amp_error": avg_rel_amp_error,
        "cc_amp_mean": cc_amp_mean,
        "cc_amp_std": cc_amp_std,
    }


def plot_combined(metrics_list: list[dict], labels: Sequence[str], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    for metrics, label in zip(metrics_list, labels):
        layers = metrics["layers"]
        axes[0].plot(layers, metrics["avg_amp_error"], marker="o", label=label)
        axes[1].plot(layers, metrics["avg_relative_amp_error"], marker="o", label=label)
        axes[2].errorbar(
            layers,
            metrics["cc_amp_mean"],
            yerr=metrics["cc_amp_std"],
            marker="o",
            label=label,
            capsize=3,
        )

    axes[0].set_ylabel("avg_amp_error")
    axes[1].set_ylabel("avg_relative_amp_error")
    axes[2].set_ylabel("cc_amp mean ± std")
    axes[2].set_xlabel("Number of layers")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Metrics vs. layer count (combined)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine metrics_vs_layers MAT files into one plot.")
    parser.add_argument(
        "--mats",
        nargs="+",
        help="List of .mat files (glob allowed). Default: results/metrics_vs_layers_*.mat",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/metrics_vs_layers_combined.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    mat_paths: list[str] = []
    if args.mats:
        for pattern in args.mats:
            mat_paths.extend(glob.glob(pattern))
    else:
        mat_paths = [
    "results/metrics_analysis/metrics_vs_layers_20251203_164757.mat",
    "results/metrics_analysis/metrics_vs_layers_20251203_164108.mat",
    "results/metrics_analysis/metrics_vs_layers_20251203_163653.mat",
    "results/metrics_analysis/metrics_vs_layers_20251203_163132.mat",
    "results/metrics_analysis/metrics_vs_layers_20251202_185654.mat",

]


    if not mat_paths:
        raise SystemExit("No .mat files found. Specify with --mats")

    paths = [Path(p) for p in sorted(mat_paths)]
    metrics_list = [load_metric_file(p) for p in paths]

    # Use filename stem as label
    labels = [p.stem for p in paths]
    out_path = Path(args.out)
    plot_combined(metrics_list, labels, out_path)
    print(f"✔ Combined plot saved -> {out_path}")
    print("Files included:")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
