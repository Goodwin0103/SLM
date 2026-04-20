 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

def compose_labels_from_patterns(H, W, patterns, centers, Index=None, 
                                 visualize=False, save_path=None, 
                                 auto_crop=True):  # 🔧 新增参数
    h, w, N = patterns.shape
    output_image = np.zeros((H, W))

    if Index is None:
        indices_to_draw = range(N)
    else:
        if not (1 <= Index <= N):
            raise ValueError(f"Index 应在 1~{N} 范围内,但得到 {Index}")
        indices_to_draw = [Index - 1]

    for i in indices_to_draw:
        cy, cx = centers[i]
        pattern = patterns[:, :, i]
        
        if auto_crop:
            # 🔧 自动裁剪到有效区域
            rows, cols = np.where(pattern > 0.5)
            if rows.size == 0:
                continue
            
            pattern_y0 = rows.min()
            pattern_y1 = rows.max() + 1
            pattern_x0 = cols.min()
            pattern_x1 = cols.max() + 1
            
            cropped = pattern[pattern_y0:pattern_y1, pattern_x0:pattern_x1]
            crop_h, crop_w = cropped.shape
            
            y0 = cy - crop_h // 2
            y1 = y0 + crop_h
            x0 = cx - crop_w // 2
            x1 = x0 + crop_w
            
            if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
                print(f"⚠️  图案 {i+1} 超出边界")
                continue
            
            output_image[y0:y1, x0:x1] = np.maximum(
                output_image[y0:y1, x0:x1], cropped
            )
        else:
            # 原始逻辑
            y0 = cy - h // 2
            y1 = y0 + h
            x0 = cx - w // 2
            x1 = x0 + w
            
            if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
                print(f"⚠️  图案 {i+1} 超出边界")
                continue
            
            output_image[y0:y1, x0:x1] = np.maximum(
                output_image[y0:y1, x0:x1], pattern[:y1-y0, :x1-x0]
            )

    # 可视化部分不变...
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
    """
    为每种形状返回一个 (h, w) 的距离/优先级评分图。
    分数越低的像素越优先被选中（用于 equal_area 模式）。

    支持形状：
        - "circle"  : 实心圆（距离中心越近分数越低）
        - "square"  : 正方形（Chebyshev 距离）
        - "diamond" : 菱形（Manhattan 距离）
        - "plus"    : 十字/加号（离最近轴线越近分数越低）
        - "ring"    : 圆环（离环中线越近分数越低）
    """
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    Y, X = np.ogrid[:h, :w]
    dx = (X - cx).astype(np.float64)
    dy = (Y - cy).astype(np.float64)

    if shape == "circle" or shape == "larger_circle" or shape == "small_circle":
        return dx ** 2 + dy ** 2

    if shape == "square":
        return np.maximum(np.abs(dx), np.abs(dy))

    if shape == "diamond":
        return np.abs(dx) + np.abs(dy)

    if shape == "plus":
        # 离水平轴或垂直轴越近 → 分数越低 → 优先选中
        return np.minimum(np.abs(dx), np.abs(dy))

    if shape == "ring":
        # 离「环中线」越近 → 分数越低 → 形成圆环
        dist = np.sqrt(dx ** 2 + dy ** 2)
        ring_mid_radius = min(h, w) * 0.35  # 环中线半径 ≈ 外径的 70%
        return np.abs(dist - ring_mid_radius)

    raise ValueError(
        f"未知形状 '{shape}'，可选值为 'circle'、'square'、'diamond'、'plus'、'ring'。"
    )

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
    ring_ratio=0.5,        # 圆环内径/外径比（仅 ring + 非 equal_area）
    plus_thickness=None,   # 十字臂宽像素（仅 plus + 非 equal_area）
    visualize=False,
    save_path=None,
):
    """
    生成 N 个检测区域图案，支持为每个标签指定不同形状。

    Parameters
    ----------
    h, w : int
        单个图案的高度和宽度。
    N : int
        检测器数量。
    shape : str
        统一形状（当 shapes=None 时使用）。
    shapes : list[str] | None
        每个检测器的形状列表，长度 >= N。
        可选值: "circle", "square", "diamond", "plus", "ring", "larger_circle"
    equal_area : bool
        若为 True，所有形状被强制缩放到相同面积（像素数）。
    target_area : int | None
        equal_area 模式的目标面积；默认使用内切圆面积。
    ring_ratio : float
        圆环内径 = 外径 × ring_ratio（仅 shape="ring" 且 equal_area=False）。
    plus_thickness : int | None
        十字臂宽（像素），默认 max(h,w)//5（仅 shape="plus" 且 equal_area=False）。
    visualize, save_path : 同前。

    Returns
    -------
    patterns : np.ndarray, shape (h, w, N)
    """
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
            # ---- 等面积模式：通过 _shape_score 排序截取 ----
            patterns[:, :, i] = _build_equal_area_mask(h, w, shape_i, target_area)
        else:
            # ---- 自由模式：按形状几何直接生成 ----
            pattern = np.zeros((h, w), dtype=np.float32)
            cy_p, cx_p = h // 2, w // 2

            if shape_i == "circle":
                radius = min(h, w) // 4
                Y, X = np.ogrid[:h, :w]
                mask = (X - cx_p) ** 2 + (Y - cy_p) ** 2 <= radius ** 2
                pattern[mask] = 1.0

            elif shape_i == "larger_circle":
                # Define a larger circle with a radius 90% of the smaller dimension
                radius = min(h, w) // 2  # 90% diameter = 45% radius
                Y, X = np.ogrid[:h, :w]
                mask = (X - cx_p) ** 2 + (Y - cy_p) ** 2 <= radius ** 2
                pattern[mask] = 1.0

            elif shape_i == "small_circle":
                radius = min(h, w) // 8  # 半径为原始圆的 1/8
                Y, X = np.ogrid[:h, :w]
                mask = (X - cx_p) ** 2 + (Y - cy_p) ** 2 <= radius ** 2
                pattern[mask] = 1.0

            elif shape_i == "square":
                pattern[:, :] = 1.0

            elif shape_i == "diamond":
                radius = min(h, w) // 2
                Y, X = np.ogrid[:h, :w]
                mask = np.abs(X - cx_p) + np.abs(Y - cy_p) <= radius
                pattern[mask] = 1.0

            elif shape_i == "plus":
                t = plus_thickness if plus_thickness is not None else max(h, w) // 5
                half_t = t // 2
                # 水平臂
                y0 = max(0, cy_p - half_t)
                y1 = min(h, cy_p + half_t + 1)
                pattern[y0:y1, :] = 1.0
                # 垂直臂
                x0 = max(0, cx_p - half_t)
                x1 = min(w, cx_p + half_t + 1)
                pattern[:, x0:x1] = 1.0

            elif shape_i == "ring":
                outer_radius = min(h, w) // 2
                inner_radius = int(outer_radius * ring_ratio)
                Y, X = np.ogrid[:h, :w]
                dist_sq = (X - cx_p) ** 2 + (Y - cy_p) ** 2
                mask = (dist_sq <= outer_radius ** 2) & (dist_sq >= inner_radius ** 2)
                pattern[mask] = 1.0

            else:
                raise ValueError(
                    f"未知形状 '{shape_i}'，可选值为 "
                    f"'circle'、'square'、'diamond'、'plus'、'ring'、'larger_circle'。"
                )
            patterns[:, :, i] = pattern

    # ---- 可视化（展示所有图案） ----
    if visualize or save_path:
        fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))
        if N == 1:
            axes = [axes]
        for i in range(N):
            axes[i].imshow(patterns[:, :, i], cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f"#{i+1}: {shape_list[i]}")
            axes[i].axis('off')
        plt.suptitle("Detector Patterns", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()

    return patterns

def generate_detector_patterns_multiwl(
    H: int,
    W: int,
    num_modes: int,
    num_wavelengths: int,
    radius: int,
    pattern_mode: str = "circle",
    show_debug: bool = False
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    生成多波长标签图案 - 按模式分行
    
    布局：num_modes 行 × num_wavelengths 列
    
    Returns:
        patterns: (H, W, num_modes * num_wavelengths)
        evaluation_regions: [(x0, x1, y0, y1), ...]
    """
    total_labels = num_modes * num_wavelengths
    
    # 🔧 强制按模式数量分行
    num_rows = num_modes
    num_cols = num_wavelengths
    
    # 计算间距（留出边距）
    margin = radius * 2
    available_height = H - 2 * margin
    available_width = W - 2 * margin
    
    row_spacing = available_height / num_rows
    col_spacing = available_width / num_cols
    
    # 计算每个标签的中心坐标
    centers = []
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            cy = margin + row_spacing * (mode_idx + 0.5)
            cx = margin + col_spacing * (wl_idx + 0.5)
            centers.append((int(cy), int(cx)))
    
    # 生成图案
    if pattern_mode == "circle":
        patterns = np.zeros((H, W, total_labels), dtype=np.float32)
        for idx, (cy, cx) in enumerate(centers):
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2
            patterns[:, :, idx] = mask.astype(np.float32)
    else:
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")
    
    # 生成评估区域
    evaluation_regions = []
    for cy, cx in centers:
        x0 = max(0, int(cx - radius))
        x1 = min(W, int(cx + radius))
        y0 = max(0, int(cy - radius))
        y1 = min(H, int(cy + radius))
        evaluation_regions.append((x0, x1, y0, y1))
    
    if show_debug:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(patterns.sum(axis=2), cmap='gray')
        ax.set_title(f"MultiWL Labels: {num_modes} modes (rows) × {num_wavelengths} wavelengths (cols)", 
                    fontsize=14, fontweight='bold')
        
        # 绘制每个标签
        for idx, (cy, cx) in enumerate(centers):
            mode_idx = idx // num_wavelengths
            wl_idx = idx % num_wavelengths
            
            # 每个模式用不同颜色
            color = plt.cm.Set3(mode_idx % 12)
            
            # 绘制圆形
            circle = Circle((cx, cy), radius=radius, 
                          linewidth=1.5, edgecolor=color, 
                          facecolor='none', alpha=0.8)
            ax.add_patch(circle)
            
            # 标注
            ax.text(cx, cy, f"M{mode_idx}W{wl_idx}", 
                   ha='center', va='center', 
                   color='white', fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=color, edgecolor='black', 
                           linewidth=0.8, alpha=0.85))
        
        # 添加网格线辅助查看
        for i in range(1, num_rows):
            y = margin + row_spacing * i
            ax.axhline(y=y, color='cyan', linestyle='--', linewidth=0.8, alpha=0.5)
        
        for j in range(1, num_cols):
            x = margin + col_spacing * j
            ax.axvline(x=x, color='cyan', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel(f"Wavelength Index (0-{num_wavelengths-1})", fontsize=11)
        ax.set_ylabel(f"Mode Index (0-{num_modes-1})", fontsize=11)
        ax.axis('on')
        
        plt.tight_layout()
        plt.savefig(RUN_ROOT / "debug_multiwl_labels.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✔ Debug label layout saved -> {RUN_ROOT / 'debug_multiwl_labels.png'}")
    
    return patterns, evaluation_regions

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
    # 快速验证三种形状
    h, w, N = 41, 41, 3
    shapes = ["larger_circle", "circle", "small_circle"]  # 可选 "circle"、"square"、"diamond"、"plus"、"ring"、"larger_circle"

    # --- 非等面积模式 ---
    p1 = generate_detector_patterns(
        h, w, N, shapes=shapes, equal_area=False,
        ring_ratio=0.5, plus_thickness=7,
        visualize=True, save_path="demo_free_area.png"
    )
    for i in range(N):
        print(f"  {shapes[i]:>8s}  面积 = {int(p1[:,:,i].sum())} px")

    # --- 等面积模式 ---
    p2 = generate_detector_patterns(
        h, w, N, shapes=shapes, equal_area=True,
        visualize=True, save_path="demo_equal_area.png"
    )
    for i in range(N):
        print(f"  {shapes[i]:>8s}  面积 = {int(p2[:,:,i].sum())} px")

    # --- 组合到大图 ---
    H, W = 150, 150
    radius = 20
    centers, _, _ = compute_label_centers(H, W, N, radius)
    output = compose_labels_from_patterns(
        H, W, p2, centers, visualize=True, save_path="demo_layout.png"
    )

def build_evaluation_regions_from_centers(centers, detectsize, H, W):
    """
    从已有的标签中心坐标构建 evaluation_regions，保证与标签布局完全一致。

    Parameters
    ----------
    centers : list of (cy, cx)
        由 compute_label_centers 返回的中心坐标列表。
    detectsize : int
        检测窗口边长（像素）。
    H, W : int
        画布高度和宽度。

    Returns
    -------
    evaluation_regions : list of (x0, x1, y0, y1)
        每个检测器的矩形边界。
    """
    half = detectsize // 2
    evaluation_regions = []
    for cy, cx in centers:
        x0 = max(0, int(cx - half))
        x1 = min(W, int(cx - half + detectsize))
        y0 = max(0, int(cy - half))
        y1 = min(H, int(cy - half + detectsize))
        evaluation_regions.append((x0, x1, y0, y1))
    return evaluation_regions
