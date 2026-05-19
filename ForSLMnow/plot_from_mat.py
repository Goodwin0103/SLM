"""
plot_metrics_from_mat.py
========================
Standalone script to replot ODNN metrics from saved .mat files.
Each metric is saved as a separate PNG (no shared subplots).

Usage
-----
# 自动找最新的 metrics_vs_layers_*.mat 并画图
python plot_metrics_from_mat.py

# 指定 mat 文件
python plot_metrics_from_mat.py path/to/metrics_vs_layers_20260514_031200.mat

# 指定输出目录
python plot_metrics_from_mat.py path/to/file.mat --out-dir my_plots/

# 同时画 crosstalk 热力图（如果 .mat 里包含 crosstalk_matrices）
python plot_metrics_from_mat.py path/to/file.mat --crosstalk

# 画 training loss 曲线（自动识别 training_curves_*.mat）
python plot_metrics_from_mat.py path/to/training_curves_layers3_*.mat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.io import loadmat


# ============================================================
# Helpers
# ============================================================
def _atleast_1d_float(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x)).astype(np.float64).ravel()


def _atleast_2d_float(x) -> np.ndarray:
    """对多波长字段：(NL, L) 形状。"""
    arr = np.asarray(x).astype(np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _get(data: dict, key: str, fallback_shape) -> np.ndarray:
    """从 mat dict 取出字段，缺失则返回 NaN 数组（兼容旧版 .mat）。"""
    if key in data:
        return _atleast_1d_float(data[key])
    if isinstance(fallback_shape, np.ndarray):
        return np.full_like(fallback_shape, np.nan, dtype=np.float64)
    return np.full(fallback_shape, np.nan, dtype=np.float64)


def _get_2d(data: dict, key: str, NL: int, L: int) -> np.ndarray:
    """取一个 (NL, L) 形状的字段；缺失返回 NaN。"""
    if key not in data:
        return np.full((NL, L), np.nan, dtype=np.float64)
    arr = _atleast_2d_float(data[key])
    # 如果存进来是 (L, NL)，自动转置
    if arr.shape == (L, NL) and L != NL:
        arr = arr.T
    if arr.shape != (NL, L):
        # 尽力 reshape
        try:
            arr = arr.reshape(NL, L)
        except Exception:
            print(f"  ⚠ {key} shape {arr.shape} != ({NL}, {L})，按原样使用")
    return arr


def _save_single(
    out_dir: Path,
    fname_stem: str,
    tag: str,
    title: str,
    ylabel: str,
    plot_fn,
    layer_counts: np.ndarray,
) -> Path:
    """单个 metric 一张图。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_fn(ax)
    ax.set_xlabel("Number of layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layer_counts)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{fname_stem}_{tag}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {ylabel:<32s} -> {out_path}")
    return out_path


# ============================================================
# Main: metrics vs layers
# ============================================================
def plot_metrics_vs_layers(mat_path: Path, out_dir: Path, do_crosstalk: bool = False) -> None:
    print(f"\n📂 Loading: {mat_path}")
    data = loadmat(str(mat_path), squeeze_me=True)

    layer_counts = _atleast_1d_float(data["layers"]).astype(int)
    if layer_counts.size == 0:
        print("⚠ No 'layers' field found, abort.")
        return
    NL = len(layer_counts)

    # --- ensure out_dir exists FIRST ---
    tag = mat_path.stem.replace("metrics_vs_layers_", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output dir: {out_dir}")
    print(f"🔢 Layer counts: {layer_counts.tolist()}")

    # --- detect multi-wavelength mode ---
    is_multiwl = "wavelengths_nm" in data
    if is_multiwl:
        wl_nm = _atleast_1d_float(data["wavelengths_nm"])
        L = int(wl_nm.size)
        print(f"🌈 Multi-wavelength mode: L={L}, λ={wl_nm.tolist()} nm\n")
    else:
        wl_nm = np.array([np.nan])
        L = 1
        print("📡 Single-wavelength mode\n")

    wl_labels = ([f"{w:.0f} nm" for w in wl_nm] if is_multiwl else ["single λ"])
    cmap_wl = (plt.cm.viridis(np.linspace(0.15, 0.85, L)) if L > 1 else
               np.array([[0.85, 0.20, 0.20, 1.0]]))

    # ============================================================
    # Load all fields (兼容单波长 1D 和多波长 2D)
    # ============================================================
    if is_multiwl:
        amp_err     = _get_2d(data, "avg_amp_error",          NL, L)
        amp_err_rel = _get_2d(data, "avg_relative_amp_error", NL, L)
        cc_mean     = _get_2d(data, "cc_amp_mean",            NL, L)
        cc_std      = _get_2d(data, "cc_amp_std",             NL, L)
        snr_db      = _get_2d(data, "snr_db_full",            NL, L)
        iso_mean    = _get_2d(data, "isolation_db_mean",      NL, L)
        iso_wc      = _get_2d(data, "isolation_db_worst",     NL, L)
        # 兼容旧字段名
        if np.all(np.isnan(iso_wc)):
            iso_wc  = _get_2d(data, "isolation_db_wc_mean",   NL, L)
        # 🆕 多波长全 ROI isolation
        iso_mean_all = _get_2d(data, "isolation_db_mean_allroi",  NL, L)
        iso_wc_all   = _get_2d(data, "isolation_db_worst_allroi", NL, L)
        # 🆕 target wavelength ratio
        target_wl    = _get_2d(data, "target_wl_ratio",       NL, L)
    else:
        # 单波长：把 1D 数组 reshape 成 (NL, 1)
        amp_err     = _get(data, "avg_amp_error",          layer_counts).reshape(-1, 1)
        amp_err_rel = _get(data, "avg_relative_amp_error", layer_counts).reshape(-1, 1)
        cc_mean     = _get(data, "cc_amp_mean",            layer_counts).reshape(-1, 1)
        cc_std      = _get(data, "cc_amp_std",             layer_counts).reshape(-1, 1)
        snr_db      = _get(data, "snr_db_full",            layer_counts).reshape(-1, 1)
        iso_mean    = _get(data, "isolation_db_mean",      layer_counts).reshape(-1, 1)
        iso_wc_raw  = _get(data, "isolation_db_worst",     layer_counts)
        if np.all(np.isnan(iso_wc_raw)):
            iso_wc_raw = _get(data, "isolation_db_wc_mean", layer_counts)
        iso_wc      = iso_wc_raw.reshape(-1, 1)
        iso_mean_all = np.full((NL, 1), np.nan, dtype=np.float64)
        iso_wc_all   = np.full((NL, 1), np.nan, dtype=np.float64)
        target_wl    = np.full((NL, 1), np.nan, dtype=np.float64)

    # ============================================================
    # 🛠 方案 A: 删除 "containment from full fraction" 那一段。
    # 训练侧已经存好了 snr_db_full（+10 dB 范围那个 union SNR），
    # 这里直接画它，不要再用 snr_ratio_full 重新算 r/(1-r)。
    # ============================================================

    # 1) avg_amp_error
    _save_single(
        out_dir, "metric_avg_amp_error", tag,
        "Average amplitude error vs. layers", "avg_amp_error",
        lambda ax: [ax.plot(layer_counts, amp_err[:, li],
                            marker="o", color=cmap_wl[li],
                            label=wl_labels[li]) for li in range(L)] + (
            [ax.legend(fontsize=8)] if L > 1 else []),
        layer_counts,
    )

    # 2) avg_relative_amp_error
    _save_single(
        out_dir, "metric_avg_relative_amp_error", tag,
        "Average relative amplitude error vs. layers", "avg_relative_amp_error",
        lambda ax: [ax.plot(layer_counts, amp_err_rel[:, li],
                            marker="o", color=cmap_wl[li],
                            label=wl_labels[li]) for li in range(L)] + (
            [ax.legend(fontsize=8)] if L > 1 else []),
        layer_counts,
    )

    # 3) cc_amp mean ± std
    _save_single(
        out_dir, "metric_cc_amp", tag,
        "Reconstruction amplitude correlation vs. layers", "cc_amp mean ± std",
        lambda ax: [ax.errorbar(layer_counts, cc_mean[:, li], yerr=cc_std[:, li],
                                marker="o", capsize=4, color=cmap_wl[li],
                                ecolor=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + (
            [ax.legend(fontsize=8)] if L > 1 else []),
        layer_counts,
    )

    # 4) ✅ SNR_full (union, dB) — 直接画训练侧保存的字段
    if not np.all(np.isnan(snr_db)):
        _save_single(
            out_dir, "metric_snr_full_db", tag,
            "SNR_full (union ROI, dB) vs. layers",
            "SNR_full (dB)",
            lambda ax: [ax.plot(layer_counts, snr_db[:, li],
                                marker="o", color=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + (
                [ax.legend(fontsize=8)] if L > 1 else []),
            layer_counts,
        )
    else:
        print("  ⚠ snr_db_full 字段缺失，跳过 SNR 图。")

    # 5) Isolation mean (same-λ)
    if not np.all(np.isnan(iso_mean)):
        _save_single(
            out_dir, "metric_isolation_db_mean", tag,
            ("Mode isolation (mean, same-λ) vs. layers" if is_multiwl
             else "Mode isolation (mean) vs. layers"),
            "Isolation mean (dB)",
            lambda ax: [ax.plot(layer_counts, iso_mean[:, li],
                                marker="o", color=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + (
                [ax.legend(fontsize=8)] if L > 1 else []),
            layer_counts,
        )
    else:
        print("  ⚠ isolation_db_mean 字段缺失，跳过。")

    # 6) Isolation worst-case (same-λ)
    if not np.all(np.isnan(iso_wc)):
        _save_single(
            out_dir, "metric_isolation_db_worst", tag,
            ("Mode isolation (worst-case, same-λ) vs. layers" if is_multiwl
             else "Mode isolation (worst-case) vs. layers"),
            "Isolation worst-case (dB)",
            lambda ax: [ax.plot(layer_counts, iso_wc[:, li],
                                marker="s", linestyle="--", color=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + (
                [ax.legend(fontsize=8)] if L > 1 else []),
            layer_counts,
        )

    # ============================================================
    # 🆕 多波长专属：all-ROI isolation（真正惩罚跨波长串扰）
    # ============================================================
    if is_multiwl and not np.all(np.isnan(iso_mean_all)):
        _save_single(
            out_dir, "metric_isolation_db_mean_allroi", tag,
            "Mode isolation (mean, all-ROI) vs. layers",
            "Isolation mean — all ROI (dB)",
            lambda ax: [ax.plot(layer_counts, iso_mean_all[:, li],
                                marker="o", color=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + [
                ax.legend(fontsize=8)],
            layer_counts,
        )

    if is_multiwl and not np.all(np.isnan(iso_wc_all)):
        _save_single(
            out_dir, "metric_isolation_db_wc_allroi", tag,
            "Mode isolation (worst-case, all-ROI) vs. layers",
            "Isolation worst — all ROI (dB)",
            lambda ax: [ax.plot(layer_counts, iso_wc_all[:, li],
                                marker="s", linestyle="--", color=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + [
                ax.legend(fontsize=8)],
            layer_counts,
        )

    # 🆕 same-λ vs all-ROI 对比图（gap = 跨波长串扰）
    if is_multiwl and not np.all(np.isnan(iso_mean_all)) and not np.all(np.isnan(iso_mean)):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for li in range(L):
            ax.plot(layer_counts, iso_mean[:, li], marker="o",
                    color=cmap_wl[li], label=f"{wl_labels[li]} same-λ")
            ax.plot(layer_counts, iso_mean_all[:, li], marker="x", linestyle="--",
                    color=cmap_wl[li], label=f"{wl_labels[li]} all-ROI")
        ax.set_xlabel("Number of layers")
        ax.set_ylabel("Isolation mean (dB)")
        ax.set_xticks(layer_counts)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_title("Same-wavelength vs. all-ROI isolation\n(gap = cross-wavelength crosstalk)")
        fig.tight_layout()
        compare_path = out_dir / f"metric_isolation_compare_{tag}.png"
        fig.savefig(compare_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✔ Same-λ vs all-ROI 对比图           -> {compare_path}")

    # 🆕 target_wl_over_all_wl_roi
    if is_multiwl and not np.all(np.isnan(target_wl)):
        _save_single(
            out_dir, "metric_target_wl_ratio", tag,
            "Wavelength-demux ratio vs. layers",
            "TargetWL / AllWL (ROI)",
            lambda ax: [ax.plot(layer_counts, target_wl[:, li],
                                marker="o", color=cmap_wl[li],
                                label=wl_labels[li]) for li in range(L)] + [
                ax.legend(fontsize=8)],
            layer_counts,
        )

    # ============================================================
    # Crosstalk heatmaps (可选)
    # ============================================================
    if do_crosstalk and "crosstalk_matrices" in data:
        ct_dir = out_dir / "crosstalk_heatmaps"
        ct_dir.mkdir(parents=True, exist_ok=True)
        ct = np.asarray(data["crosstalk_matrices"], dtype=np.float64)
        # 形状: 单波长 (NL, M, M) 或多波长 (NL, L, M, M)
        if ct.ndim == 2:
            ct = ct[None, ...]   # (1, M, M)

        if is_multiwl and ct.ndim == 4:
            # 多波长 (NL, L, M, M)
            n_layers_axis, L_, M, _ = ct.shape
            print(f"\n🔥 Plotting {n_layers_axis * L_} same-λ crosstalk heatmaps -> {ct_dir}")
            for i in range(n_layers_axis):
                for li in range(L_):
                    n_layer = int(layer_counts[i]) if i < len(layer_counts) else i + 1
                    _plot_one_crosstalk(
                        ct[i, li], ct_dir,
                        title_lin=f"Crosstalk (linear) — {n_layer} layers, λ={wl_nm[li]:.0f}nm",
                        title_db =f"Crosstalk (dB) — {n_layer} layers, λ={wl_nm[li]:.0f}nm",
                        fname_lin=f"crosstalk_linear_L{n_layer}_wl{li:02d}_{tag}.png",
                        fname_db =f"crosstalk_db_L{n_layer}_wl{li:02d}_{tag}.png",
                        M=M,
                    )
        else:
            # 单波长 (NL, M, M)
            n_layers_axis, M, _ = ct.shape
            print(f"\n🔥 Plotting {n_layers_axis} crosstalk heatmaps -> {ct_dir}")
            for i in range(n_layers_axis):
                n_layer = int(layer_counts[i]) if i < len(layer_counts) else i + 1
                _plot_one_crosstalk(
                    ct[i], ct_dir,
                    title_lin=f"Crosstalk (linear) — {n_layer} layers",
                    title_db =f"Crosstalk (dB) — {n_layer} layers",
                    fname_lin=f"crosstalk_linear_layers{n_layer}_{tag}.png",
                    fname_db =f"crosstalk_db_layers{n_layer}_{tag}.png",
                    M=M,
                )

        # 🆕 多波长全 ROI 串扰矩阵（M × M·L）
        if is_multiwl and "crosstalk_matrices_full" in data:
            ct_full = np.asarray(data["crosstalk_matrices_full"], dtype=np.float64)
            # (NL, L, M, M*L)
            if ct_full.ndim == 4:
                ct_full_dir = out_dir / "crosstalk_heatmaps_full"
                ct_full_dir.mkdir(parents=True, exist_ok=True)
                NL_ct, L_ct, M_ct, ML_ct = ct_full.shape
                for i in range(NL_ct):
                    for li in range(L_ct):
                        mat = ct_full[i, li]
                        mat_db = 10.0 * np.log10(np.clip(mat, 1e-6, None))
                        n_layer = int(layer_counts[i])

                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        im = ax.imshow(mat_db, cmap="magma", vmin=-30, vmax=0, aspect="auto")
                        ax.set_title(f"Full Crosstalk dB — {n_layer} layers, source λ={wl_nm[li]:.0f}nm")
                        ax.set_ylabel("Input mode")
                        ax.set_xlabel("Detector (mode × λ)")
                        ax.set_yticks(range(M_ct))
                        ax.set_xticks(list(range(ML_ct)))
                        ax.set_xticklabels(
                            [f"M{k}\nλ{w}" for k in range(M_ct) for w in range(L_ct)],
                            fontsize=6,
                        )
                        for k in range(1, M_ct):
                            ax.axvline(k * L_ct - 0.5, color="cyan", linewidth=0.5, alpha=0.6)
                        for k in range(M_ct):
                            ax.add_patch(Rectangle(
                                (k * L_ct + li - 0.5, -0.5), 1, M_ct,
                                fill=False, edgecolor="lime", linewidth=1.0,
                            ))
                        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dB")
                        fig.tight_layout()
                        p = ct_full_dir / f"ct_full_db_L{n_layer}_wl{li:02d}_{tag}.png"
                        fig.savefig(p, dpi=300, bbox_inches="tight")
                        plt.close(fig)
                        print(f"  ✔ Full-ROI crosstalk dB (L={n_layer}, λ={wl_nm[li]:.0f}nm) -> {p}")

    print(f"\n✅ Done. All metric figures saved under: {out_dir}\n")


def _plot_one_crosstalk(mat: np.ndarray, ct_dir: Path,
                        *, title_lin: str, title_db: str,
                        fname_lin: str, fname_db: str, M: int) -> None:
    mat_db = 10.0 * np.log10(np.clip(mat, 1e-6, None))

    # linear
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title_lin)
    ax.set_xlabel("Detector index"); ax.set_ylabel("Input mode index")
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="energy fraction")
    for r in range(M):
        for c in range(M):
            v = mat[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v < 0.5 else "black"), fontsize=8)
    fig.tight_layout()
    fig.savefig(ct_dir / fname_lin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Crosstalk linear -> {ct_dir / fname_lin}")

    # dB
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat_db, cmap="magma", vmin=-30, vmax=0)
    ax.set_title(title_db)
    ax.set_xlabel("Detector index"); ax.set_ylabel("Input mode index")
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
    for r in range(M):
        for c in range(M):
            v = mat_db[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        color=("white" if v < -15 else "black"), fontsize=8)
    fig.tight_layout()
    fig.savefig(ct_dir / fname_db, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Crosstalk dB     -> {ct_dir / fname_db}")


# ============================================================
# Bonus: training loss curve plotter
# ============================================================
def plot_training_curve(mat_path: Path, out_dir: Path) -> None:
    print(f"\n📂 Loading training curve: {mat_path}")
    data = loadmat(str(mat_path), squeeze_me=True)
    epochs = _atleast_1d_float(data["epochs"])
    losses = _atleast_1d_float(data["losses"])
    n_layer = int(np.atleast_1d(data["num_layers"])[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Training loss — {n_layer} layers")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    out_path = out_dir / f"{mat_path.stem}_replot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Loss curve -> {out_path}\n")


# ============================================================
# Entry point
# ============================================================
def find_latest_metrics_mat(search_root: Path) -> Path | None:
    """自动找最新的 metrics_vs_layers_*.mat。"""
    candidates = list(search_root.rglob("metrics_vs_layers_*.mat"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Replot ODNN metrics from saved .mat files (one PNG per metric)."
    )
    parser.add_argument(
        "mat_path", nargs="?", type=str, default=None,
        help="Path to .mat file. If omitted, auto-find the latest metrics_vs_layers_*.mat.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory. Default: <mat_dir>/replot/",
    )
    parser.add_argument(
        "--crosstalk", action="store_true",
        help="Also plot crosstalk heatmaps if 'crosstalk_matrices' is present.",
    )
    parser.add_argument(
        "--search-root", type=str, default="results_6modes_eigenmode2",
        help="Root directory to search for the latest .mat (used when mat_path is omitted).",
    )
    args = parser.parse_args()

    # Resolve mat_path
    if args.mat_path:
        mat_path = Path(args.mat_path).expanduser().resolve()
    else:
        root = Path(args.search_root).expanduser().resolve()
        mat_path = find_latest_metrics_mat(root)
        if mat_path is None:
            print(f"❌ No metrics_vs_layers_*.mat found under {root}")
            sys.exit(1)
        print(f"🔎 Auto-selected latest mat: {mat_path}")

    if not mat_path.exists():
        print(f"❌ File not found: {mat_path}")
        sys.exit(1)

    # Resolve out_dir
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = mat_path.parent / "replot"

    # Dispatch by filename
    name = mat_path.name.lower()
    if "training_curves" in name:
        plot_training_curve(mat_path, out_dir)
    else:
        plot_metrics_vs_layers(mat_path, out_dir, do_crosstalk=args.crosstalk)


if __name__ == "__main__":
    main()
