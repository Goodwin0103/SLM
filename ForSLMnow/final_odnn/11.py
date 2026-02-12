# extract_phi_multiwl.py
import os
import numpy as np
import torch

# 可选：画图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 你的模型类（确保路径/文件名正确）
from odnn_multiwl_model import D2NNModelMultiWL


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    wavelengths = np.array(meta["wavelengths"], dtype=np.float32)

    model = D2NNModelMultiWL(
        num_layers=meta["num_layers"],
        layer_size=meta["layer_size"],
        z_layers=float(meta["z_layers"]),
        z_prop=float(meta["z_prop"]),
        pixel_size=float(meta["pixel_size"]),
        wavelengths=wavelengths,
        device=device,
        padding_ratio=float(meta["padding_ratio"]),
        z_input_to_first=float(meta.get("z_input_to_first", 0.0)),
        base_wavelength_idx=int(meta["base_wavelength_idx"]),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, meta, wavelengths


def compute_phi_per_wavelength(model, layer_idx: int, wavelengths_m: np.ndarray, base_wl_idx: int):
    """
    返回：
      phi0: (H,W) float32
      phi_scaled: (L,H,W) float32   # 不 wrap
      phi_wrapped: (L,H,W) float32  # wrap 到 [0,2pi)
      lam0: float (m)
    """
    wls = torch.tensor(np.asarray(wavelengths_m, dtype=np.float32))  # (L,)
    layer = model.layers[layer_idx]

    # 1) phi0（要训练的那张相位参数）
    phi0 = layer.phase.detach().cpu().to(torch.float32)  # (H,W)

    # 2) lam0：优先读 layer.lam0；否则用 base_wl
    if hasattr(layer, "lam0"):
        lam0 = float(layer.lam0.detach().cpu().item())
    else:
        lam0 = float(wavelengths_m[base_wl_idx])

    # 3) phi(lambda) = phi0 * lam0/lambda
    scale = (lam0 / wls).view(-1, 1, 1)                   # (L,1,1)
    phi_scaled = (phi0.unsqueeze(0) * scale).contiguous() # (L,H,W)

    # 4) wrap 到 [0,2pi)
    two_pi = float(2 * np.pi)
    phi_wrapped = torch.remainder(phi_scaled, two_pi)

    return phi0.numpy(), phi_scaled.numpy(), phi_wrapped.numpy(), lam0


def quick_ratio_check(phi0_hw: np.ndarray, phi_scaled_lhw: np.ndarray, lam0: float, wavelengths_m: np.ndarray):
    """
    用 std 比例验证 phi_scaled 是否按 lam0/lam 缩放（在“不 wrap”的 phi_scaled 上检验）
    """
    std0 = float(phi0_hw.std())
    stds = phi_scaled_lhw.reshape(phi_scaled_lhw.shape[0], -1).std(axis=1)
    measured = stds / (std0 + 1e-12)
    expected = lam0 / wavelengths_m
    return expected, measured


def save_npz(out_path: str, phi0_hw, phi_scaled_lhw, phi_wrapped_lhw, lam0, wavelengths_m):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        phi0=phi0_hw.astype(np.float32),
        phi_scaled=phi_scaled_lhw.astype(np.float32),
        phi_wrapped=phi_wrapped_lhw.astype(np.float32),
        lam0_m=np.array([lam0], dtype=np.float64),
        wavelengths_m=wavelengths_m.astype(np.float64),
    )
    print("✔ Saved:", out_path)


def save_debug_png(out_dir: str, phi0_hw, phi_scaled_lhw, phi_wrapped_lhw, wavelengths_m, lam0, layer_idx: int):
    os.makedirs(out_dir, exist_ok=True)

    # phi0
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
    im = ax.imshow(np.remainder(phi0_hw, 2*np.pi), cmap="twilight", vmin=0, vmax=2*np.pi)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("phase (rad)")
    ax.set_title(f"Layer {layer_idx} | phi0 (wrapped) | lam0={lam0*1e9:.1f} nm")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"layer{layer_idx:02d}_phi0_wrapped.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # per wavelength
    for li, wl in enumerate(wavelengths_m):
        wl_nm = wl * 1e9

        # scaled (not wrapped): 画出来会很“花”，用对称色图更直观
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
        v = np.percentile(np.abs(phi_scaled_lhw[li]), 99)
        im = ax.imshow(phi_scaled_lhw[li], cmap="seismic", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("phase (rad)")
        ax.set_title(f"Layer {layer_idx} | phi_scaled (NOT wrapped) | λ={wl_nm:.1f} nm")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"layer{layer_idx:02d}_phi_scaled_l{li}_lambda{wl_nm:.1f}nm.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # wrapped
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
        im = ax.imshow(phi_wrapped_lhw[li], cmap="twilight", vmin=0, vmax=2*np.pi)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("phase (rad)")
        ax.set_title(f"Layer {layer_idx} | phi_wrapped | λ={wl_nm:.1f} nm")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"layer{layer_idx:02d}_phi_wrapped_l{li}_lambda{wl_nm:.1f}nm.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("✔ Saved PNGs ->", out_dir)


def main():
    # ====== 你需要改的 3 个参数 ======
    ckpt_path = "checkpoints/odnn_multiwl_5layers_m5_ls110.pth"  # 改成你的实际 checkpoint
    layer_idx = 0                                                # 想看第几层：0-based
    out_root = "results/phi_extract"                              # 输出目录
    save_png = True                                               # 是否输出 png

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, meta, wavelengths = load_model_from_ckpt(ckpt_path, device)
    print("Loaded:", ckpt_path)
    print("wavelengths (nm):", [float(w*1e9) for w in wavelengths])
    print("num_layers:", meta["num_layers"], "| layer_size:", meta["layer_size"])

    phi0_hw, phi_scaled_lhw, phi_wrapped_lhw, lam0 = compute_phi_per_wavelength(
        model,
        layer_idx=layer_idx,
        wavelengths_m=wavelengths,
        base_wl_idx=int(meta["base_wavelength_idx"]),
    )

    expected, measured = quick_ratio_check(phi0_hw, phi_scaled_lhw, lam0, wavelengths)
    print("lam0 (nm):", lam0 * 1e9)
    print("expected lam0/lam:", expected)
    print("measured std ratio:", measured)

    tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_npz = os.path.join(out_root, f"{tag}_layer{layer_idx:02d}_phi_multiwl.npz")
    save_npz(out_npz, phi0_hw, phi_scaled_lhw, phi_wrapped_lhw, lam0, wavelengths)

    if save_png:
        out_png_dir = os.path.join(out_root, "png", tag)
        save_debug_png(out_png_dir, phi0_hw, phi_scaled_lhw, phi_wrapped_lhw, wavelengths, lam0, layer_idx)
    # 统一色条：用三张里最大的 99 分位幅度作为 vmax
    v = np.percentile(np.abs(phi_scaled_lhw), 99)
    for li, wl in enumerate(wavelengths):
        wl_nm = wl * 1e9
        plt.figure(figsize=(5,5))
        plt.imshow(phi_scaled_lhw[li], cmap="seismic", vmin=-v, vmax=v)
        plt.colorbar(label="phase (rad)")
        plt.title(f"phi_scaled (NOT wrapped) | λ={wl_nm:.1f} nm | same vmin/vmax")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, f"same_vmin_vmax_layer{layer_idx:02d}_lambda{wl_nm:.1f}nm.png"))
        plt.close()
        # ============================
    # NEW: 画差值 / 残差（推荐）
    # ============================
    diff_dir = os.path.join(out_root, "diff", tag, f"layer{layer_idx:02d}")
    os.makedirs(diff_dir, exist_ok=True)

    # (A) 残差：phi_scaled - phi0*(lam0/lam) —— 理论上应≈0（浮点误差级别）
    expected_scaled = phi0_hw[None, :, :] * (lam0 / wavelengths)[:, None, None]  # (L,H,W)
    residual = phi_scaled_lhw - expected_scaled                                   # (L,H,W)

    max_abs = float(np.max(np.abs(residual)))
    mean_abs = float(np.mean(np.abs(residual)))
    print("Residual check: max|Δ| =", max_abs, " mean|Δ| =", mean_abs)

    # 用对称色条（别让全黑看不见，用 max_abs 做范围）
    v_res = max_abs + 1e-12

    for li, wl in enumerate(wavelengths):
        wl_nm = wl * 1e9
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
        im = ax.imshow(residual[li], cmap="seismic", vmin=-v_res, vmax=v_res)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("residual Δphase (rad)")
        ax.set_title(f"Residual: phi_scaled - phi0*(lam0/lam) | λ={wl_nm:.1f} nm")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(diff_dir, f"residual_lambda{wl_nm:.1f}nm.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # (B) 波长两两差值：phi_scaled(λa) - phi_scaled(λb)（NOT wrapped）
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        wl_a_nm = wavelengths[a] * 1e9
        wl_b_nm = wavelengths[b] * 1e9
        diff = phi_scaled_lhw[a] - phi_scaled_lhw[b]  # (H,W)

        v = np.percentile(np.abs(diff), 99) + 1e-12
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.0))
        im = ax.imshow(diff, cmap="seismic", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("phase diff (rad)")
        ax.set_title(f"Diff (NOT wrapped): {wl_a_nm:.1f}nm - {wl_b_nm:.1f}nm")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(diff_dir, f"diff_{wl_a_nm:.1f}nm_minus_{wl_b_nm:.1f}nm.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("✔ Saved diff/residual plots ->", diff_dir)


if __name__ == "__main__":
    main()
