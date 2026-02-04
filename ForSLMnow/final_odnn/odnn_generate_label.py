 
import numpy as np
import matplotlib.pyplot as plt

def compute_label_centers(H, W, N, radius):
    """
    计算N个图案的中心位置（与圆形布局相同）。
    """
    num_rows = int(np.floor(np.sqrt(N)))
    num_cols = int(np.ceil(N / num_rows))

    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)

    if row_spacing < 0 or col_spacing < 0:
        raise ValueError("The patterns cannot fit into the image with the given parameters.")

    centers = []
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if len(centers) < N:
                cy = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                cx = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                centers.append((cy, cx))
            else:
                break
    
    center_row_spacing = 2 * radius + row_spacing
    center_col_spacing = 2 * radius + col_spacing
    print("相邻图案边缘间距：", f"行={row_spacing:.2f}, 列={col_spacing:.2f}")
    print("相邻图案中心间距：", f"行={center_row_spacing:.2f}, 列={center_col_spacing:.2f}")
    print("中心坐标：", centers)

    return centers, row_spacing, col_spacing


def compose_labels_from_patterns(H, W, patterns, centers, Index=None, visualize=False, save_path=None):
    """
    将给定的N个图案（patterns[..., i]）按照centers摆放到一张大图上。
    支持：
        - Index=None: 绘制所有图案
        - Index=k: 仅绘制第k个图案（1-based）

    Parameters
    ----------
    H, W : int
        输出图像的大小。
    patterns : np.ndarray
        形状为 (h, w, N) 的3D数组，每个图案一个通道。
    centers : list[tuple[int, int]]
        每个图案的中心坐标。
    Index : int | None
        若为 None，绘制所有；若为整数，仅绘制对应编号（1-based）。
    visualize : bool
        是否显示生成结果。
    save_path : str | None
        若提供路径，则将图像保存到该路径。

    Returns
    -------
    output_image : np.ndarray
        合成后的 H×W 图像。
    """
    h, w, N = patterns.shape
    output_image = np.zeros((H, W))

    # 决定要绘制哪些图案
    if Index is None:
        indices_to_draw = range(N)
    else:
        if not (1 <= Index <= N):
            raise ValueError(f"Index 应在 1~{N} 范围内，但得到 {Index}")
        indices_to_draw = [Index - 1]

    # 绘制图案
    for i in indices_to_draw:
        cy, cx = centers[i]
        pattern = patterns[:, :, i]

        # pattern 的位置范围（确保完整放置图案）
        y0 = cy - h // 2
        y1 = y0 + h
        x0 = cx - w // 2
        x1 = x0 + w

        # 边界检查
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            print(f"⚠️  图案 {i+1} 超出边界，已跳过。")
            continue

        # 叠加图案（取最大值相当于“并集”）
        output_image[y0:y1, x0:x1] = np.maximum(
            output_image[y0:y1, x0:x1],
            pattern[:y1 - y0, :x1 - x0]
        )

    # 可视化
    if visualize or save_path:
        plt.figure(figsize=(6, 6))
        plt.imshow(output_image, cmap='gray')
        title = "All Labels" if Index is None else f"Label #{Index}"
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()
    return output_image

def generate_detector_patterns(h, w, N, shape="circle", visualize=False, save_path=None):
    """
    生成 N 个检测区域图案（圆形或方形），组成一个三维数组。

    Parameters
    ----------
    h, w : int
        单个检测器图案的高度和宽度。
    N : int
        检测器数量（输出通道数）。
    shape : str
        检测区域形状:
            - "circle": 圆形区域（半径 = min(h, w)/2）
            - "square": 方形区域（全为1）
    visualize : bool
        是否显示生成的单个图案。
    save_path : str | None
        若提供路径，将单个检测器图案保存到该路径。

    Returns
    -------
    patterns : np.ndarray
        形状为 (h, w, N) 的三维数组，每个切片为一个检测区域（值为1）。
    """
    pattern = np.zeros((h, w))

    if shape == "circle":
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 2
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
        pattern[mask] = 1

    elif shape == "square":
        pattern[:, :] = 1

    else:
        raise ValueError(f"未知形状 '{shape}'，可选值为 'circle' 或 'square'。")

    # 可视化单个检测图案
    if visualize or save_path:
        plt.figure(figsize=(4, 4))
        plt.imshow(pattern, cmap='gray')
        plt.title(f"Detector pattern ({shape})")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()

    # 复制成 N 个检测图案（第三维）
    patterns = np.repeat(pattern[:, :, np.newaxis], N, axis=2)

    return patterns


def main():
    import os
    from odnn_io import load_complex_modes_from_mat

    current_path = os.getcwd()
    print("Current Working Directory:", current_path)
    mat_path = os.path.join(current_path, "mmf_6modes_25_PD_1.15.mat")
    figure_output_dir = os.path.join(current_path, "generated_figures")
    os.makedirs(figure_output_dir, exist_ok=True)

    eigenmodes = load_complex_modes_from_mat(mat_path, key="modes_field")
        

    # 假设每个图案是 41x41，N=9
    h, w, N = 110, 110, 6
    patterns = abs(eigenmodes)

    # # 创建一些测试图案：不同方向的十字
    # for i in range(N):
    #     p = np.zeros((h, w))
    #     p[h//2, :] = 1
    #     p[:, w//2] = 1
    #     if i % 2 == 0:
    #         p = np.rot90(p, k=i % 4)
    #     patterns[..., i] = p

    # 计算圆心位置
    H, W = 110, 110
    radius = 10
    centers, row_spacing, col_spacing = compute_label_centers(H, W, N, radius)

    # 组合成一张图
    # output = compose_labels_from_patterns(H, W, patterns, centers, visualize=True)
    output_all_path = os.path.join(figure_output_dir, "labels_all.png")
    output_all = compose_labels_from_patterns(
        H, W, patterns, centers, Index=None, visualize=False, save_path=output_all_path
    )
    print(f"All labels visualization saved to: {output_all_path}")

    output_single_path = os.path.join(figure_output_dir, "label_6.png")
    output_single = compose_labels_from_patterns(
        H, W, patterns, centers, Index=6, visualize=False, save_path=output_single_path
    )
    print(f"Label #6 visualization saved to: {output_single_path}")

    detector_pattern_path = os.path.join(figure_output_dir, "detector_pattern_square.png")
    patterns_circle = generate_detector_patterns(
        h=27, w=27, N=6, shape="circle", visualize=False, save_path=detector_pattern_path
    ) #circle or square
    print(f"Detector pattern visualization saved to: {detector_pattern_path}")
    # print(patterns_circle.shape)
    detector_layout_path = os.path.join(figure_output_dir, "detector_layout.png")
    detector = compose_labels_from_patterns(
        H, W, patterns=patterns_circle, centers=centers, Index=None, visualize=False, save_path=detector_layout_path
    )
    print(f"Detector layout visualization saved to: {detector_layout_path}")


if __name__ == "__main__":
    main()
