import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from odnn_model import D2NNModel
from odnn_io import load_complex_modes_from_mat
from odnn_processing import (
    build_spot_masks,
    detector_weights_from_intensity,
    pad_field_to_layer,
)
from odnn_visualization import plot_best_cv_comparison, plot_field_amp_phase, plot_modes_amp_phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # === 配置 ===
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "verify_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path("/scratch/QZ_backup/best_result.npz").expanduser()
    mat_path = base_dir / "mmf_6modes_25_PD_1.15.mat"
    layer_size = 100
    num_modes = 6
    focus_radius = 5
    energy_radius = 8
    pixel_size = 1e-6
    wavelength = 1568e-9
    z_layers = 40e-6
    z_prop = 120e-6
    z_input_to_first = 40e-6

    # === 加载 complex weights ===
    data = np.load(npz_path)
    best_cv = np.squeeze(data["best_cv"])
    complex_weights_ref = best_cv[:num_modes].astype(np.complex64, copy=False)
    amplitude_ref = np.abs(complex_weights_ref)
    phase_ref = np.angle(complex_weights_ref)
    amplitude_ref_norm = amplitude_ref

    # === 加载模场并生成输入光场 ===
    eigenmodes = load_complex_modes_from_mat(mat_path, key="modes_field")
    mmf_modes = eigenmodes[:, :, :num_modes].transpose(2, 0, 1).astype(np.complex64)
    #plot_modes_amp_phase(mmf_modes, output_dir / "mmf_modes_amp_phase.png")
    field_complex = np.tensordot(complex_weights_ref, mmf_modes, axes=(0, 0))
    plot_field_amp_phase(field_complex, output_dir / "input_field_pre_padding.png", title_prefix="Input Field (pre-padding)")
    field_tensor = torch.from_numpy(field_complex).to(device=device, dtype=torch.complex64)

    # === 填充到网络输入尺寸 ===
    field_padded = pad_field_to_layer(field_tensor, layer_size)

    # === 构建并加载 ODNN 模型 ===
    D2NN = D2NNModel(
        num_layers=3,
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelength=wavelength,
        device=device,
        padding_ratio=0.5,
        z_input_to_first=z_input_to_first,
    ).to(device)

    mask_dir = base_dir
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".xlsx"))
    masks_np = [pd.read_excel(mask_dir / f, header=None).to_numpy(dtype=np.float32) for f in mask_files]
    for layer, mask_np in zip(D2NN.layers, masks_np):
        with torch.no_grad():
            layer.phase.copy_(torch.tensor(mask_np, dtype=torch.float32, device=device))

    # === 前向传播 ===
    input_tensor = field_padded.unsqueeze(0).unsqueeze(0).to(next(D2NN.parameters()).device)
    with torch.no_grad():
        output_intensity = D2NN(input_tensor).squeeze().cpu().numpy()

    # === 统计能量并转换为权重 ===
    spot_masks = build_spot_masks(layer_size, num_modes, focus_radius, energy_radius)
    energies_pred, weights_pred = detector_weights_from_intensity(output_intensity, spot_masks)

    # === 使用预测幅度 + 原始相位重建光场 ===
    complex_weights_pred = weights_pred * np.exp(1j * phase_ref)
    field_reconstructed = np.tensordot(complex_weights_pred, mmf_modes, axes=(0, 0))

    # === 结果可视化 ===
    field_input_np = field_padded.cpu().numpy()
    plot_best_cv_comparison(
        field_input_np,
        output_intensity,
        amplitude_ref_norm,
        weights_pred,
        field_reconstructed,
        output_dir / "best_cv_comparison.png",
    )

    # === 结果打印 ===
    print("✔ 输入幅度 L2 归一化:", np.round(amplitude_ref_norm, 4))
    print("✔ 预测幅度 (L2 归一):", np.round(weights_pred, 4))
    print(f"✔ 输出强度保存: {output_dir / 'best_cv_comparison.png'}")


if __name__ == "__main__":
    main()
