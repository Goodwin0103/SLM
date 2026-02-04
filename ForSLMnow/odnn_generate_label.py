 
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

        y0 = cy - h // 2
        y1 = y0 + h
        x0 = cx - w // 2
        x1 = x0 + w

        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            print(f"⚠️  图案 {i+1} 超出边界，已跳过。")
            continue
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

def _shape_score(h, w, shape):
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    Y, X = np.ogrid[:h, :w]
    dx = X - cx
    dy = Y - cy

    if shape == "circle":
        return dx**2 + dy**2
    if shape == "square":
        return np.maximum(np.abs(dx), np.abs(dy))
    if shape == "diamond":
        return np.abs(dx) + np.abs(dy)
    raise ValueError(f"未知形状 '{shape}'，可选值为 'circle'、'square' 或 'diamond'。")


def _build_equal_area_mask(h, w, shape, target_area):
    score = _shape_score(h, w, shape)
    flat = score.ravel()
    total = flat.size
    area = int(round(target_area))
    area = max(1, min(area, total))
    idx = np.argpartition(flat, area - 1)[:area]
    mask = np.zeros(total, dtype=np.float32)
    mask[idx] = 1.0
    return mask.reshape(h, w)


def _default_circle_area(h, w):
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) / 2.0
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2
    return int(mask.sum())

#这个是可以生成很多不同形状的输出
def generate_detector_patterns(
    h,
    w,
    N,
    shape="circle",
    shapes=None,
    equal_area=False,
    target_area=None,
    visualize=False,
    save_path=None,
):
   
    if shapes is None:
        shape_list = [shape] * N
    else:
        if len(shapes) < N:
            raise ValueError(f"shapes 长度需 >= N，但得到 {len(shapes)} < {N}")
        shape_list = list(shapes[:N])

    if equal_area and target_area is None:
        target_area = _default_circle_area(h, w)

    patterns = np.zeros((h, w, N), dtype=np.float32)
    for i, shape_i in enumerate(shape_list):
        if equal_area:
            pattern = _build_equal_area_mask(h, w, shape_i, target_area)
        else:
            pattern = np.zeros((h, w), dtype=np.float32)
            if shape_i == "circle":
                cy, cx = h // 2, w // 2
                radius = min(h, w) // 2
                Y, X = np.ogrid[:h, :w]
                mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2
                pattern[mask] = 1.0
            elif shape_i == "square":
                pattern[:, :] = 1.0
            else:
                raise ValueError(
                    f"未知形状 '{shape_i}'，可选值为 'circle' 或 'square'。"
                )
        patterns[:, :, i] = pattern

    # 可视化单个检测图案
    if visualize or save_path:
        plt.figure(figsize=(4, 4))
        plt.imshow(patterns[:, :, 0], cmap='gray')
        title_shape = shape_list[0] if shape_list else shape
        plt.title(f"Detector pattern ({title_shape})")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()

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
