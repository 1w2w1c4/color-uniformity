"""色彩均勻度分析核心：sRGB → CIELAB，切格後以 CIEDE2000 計算每格與參考色的色差。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SPOT_MAX_SIDE = 1000    # 主色估計與小點偵測的工作解析度上限（長邊）
MIN_DETECT_PX = 2.5     # 工作解析度下可穩定分辨的最小瑕疵直徑（像素），用來回報本張照片的偵測下限
HIST_BIN = 3.0          # 主色估計：Lab 直方圖每格寬度
CLUSTER_RADIUS = 6.0    # 主色估計：收斂時納入的 Lab 半徑
REGION_RATIO = 0.05     # 瑕疵面積超過畫面 5% 視為「大面積不均」，否則為「小點」
MAX_SPOTS_LISTED = 50

# sRGB (D65) → XYZ 轉換矩陣與 D65 白點
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3)，值域 0–255。回傳同形狀的 Lab。"""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE_D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 色差（向量化，kL=kC=kH=1）。"""
    lab1, lab2 = np.broadcast_arrays(np.asarray(lab1, float), np.asarray(lab2, float))
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbarp = np.where(np.abs(h1p - h2p) > 180,
                     np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
                     hsum / 2)
    hbarp = np.where(C1p * C2p == 0, hsum, hbarp)

    T = (1 - 0.17 * np.cos(np.radians(hbarp - 30)) + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6)) - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    d_theta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    Cbarp7 = Cbarp ** 7
    Rc = 2 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbarp - 50) ** 2 / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc

    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Lab → sRGB（0–255，超出色域者截斷）。"""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    eps, kappa = 216 / 24389, 24389 / 27
    xyz = np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kappa) * _WHITE_D65
    lin = np.clip(xyz @ np.linalg.inv(_M_RGB2XYZ).T, 0, 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(srgb * 255, 0, 255)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
        raise ValueError(f"無效的色碼：{hex_color!r}（需為 #RRGGBB）")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


@dataclass
class UniformityResult:
    rows: int
    cols: int
    reference_lab: list[float]
    reference_hex: str
    reference_source: str
    reference_coverage: float
    delta_e_grid: list[list[float]]
    cell_lab_grid: list[list[list[float]]]
    delta_e_max: float
    delta_e_mean: float
    delta_e_p95: float
    lightness_std: float
    luminance_uniformity: float
    worst_cell: dict
    cells_over_threshold: int
    threshold: float
    grid_passed: bool
    spots: dict
    passed: bool
    score: float
    pixel_delta_e: np.ndarray | None = field(default=None, repr=False)  # 逐像素 ΔE（工作解析度），供熱圖使用
    defect_map: np.ndarray | None = field(default=None, repr=False)     # 瑕疵編號圖（工作解析度），供熱圖使用

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("pixel_delta_e")
        d.pop("defect_map")
        return d


def classify_tone(lab: np.ndarray, ref_lab: np.ndarray) -> str:
    """描述瑕疵相對基準色的變化方向（ΔL 亮度、ΔC 彩度）。"""
    dL = float(lab[0] - ref_lab[0])
    dC = float(np.hypot(lab[1], lab[2]) - np.hypot(ref_lab[1], ref_lab[2]))
    dE = float(np.linalg.norm(np.asarray(lab, float) - ref_lab))
    dH = np.sqrt(max(dE ** 2 - dL ** 2 - dC ** 2, 0.0))   # 色相方向的變化量
    if dH > max(abs(dL), abs(dC)):
        return "偏色"            # 主要是色相改變（例如紅偏橘、偏紫）
    lightness_driven = abs(dL) > 1 and abs(dL) >= 0.4 * abs(dC)  # 亮度變化夠顯著才以亮度描述
    if lightness_driven and dL > 0:
        return "泛白" if dC < -1 else "偏亮"   # 變亮且變淡：壓力不足時的典型表現
    if lightness_driven:
        return "偏暗"
    if dC < -1:
        return "變灰"
    if dC > 1:
        return "偏濃"
    return "偏色"


def position_name(fx: float, fy: float) -> str:
    """相對座標 (0–1) → 九宮格方位文字。"""
    col = "左" if fx < 1 / 3 else ("右" if fx > 2 / 3 else "")
    row = "上" if fy < 1 / 3 else ("下" if fy > 2 / 3 else "")
    if not col and not row:
        return "中央"
    if col and row:
        return f"{col}{row}方"
    return f"{col or row}側" if col else f"{row}方"


def _score(de_max: float, threshold: float) -> float:
    """最大色差 0 → 100 分，達門檻 → 60 分，2 倍門檻以上 → 0 分。"""
    if de_max <= threshold:
        return float(np.clip(100 - 40 * de_max / threshold, 0, 100))
    return float(np.clip(60 * (2 - de_max / threshold), 0, 60))


def working_lab(rgb: np.ndarray) -> np.ndarray:
    """縮小後以 3×3 中值濾波壓掉顆粒雜訊（不會擴大瑕疵面積），轉 Lab。

    一律至少縮小 2 倍（長邊 ≤ 原圖一半，且 ≤ SPOT_MAX_SIDE）：BOX 縮小會平均相鄰像素，
    相機雜訊減半。照片尺寸不固定時，小照片若不縮小，雜訊會被誤判為小點。
    """
    H, W = rgb.shape[:2]
    scale = min(SPOT_MAX_SIDE, max(H, W) / 2) / max(H, W)
    img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
    img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.Resampling.BOX)
    img = img.filter(ImageFilter.MedianFilter(3))
    return srgb_to_lab(np.asarray(img, dtype=np.float64))


def estimate_reference(lab: np.ndarray) -> np.ndarray:
    """從影像本身找出「主色」：佔最多像素的那一群相近顏色。

    不用整張圖的平均或中位數——當泛白面積很大時，平均/中位數會落在紅與白之間的粉紅色，
    變成一個實際上不存在的顏色，導致紅色與白色區域同時被判為異常。
    作法：
    1. 在 Lab 空間建 3D 直方圖（每格 HIST_BIN），平滑後取最高峰 → 主色的大致位置。
    2. 取峰值附近 CLUSTER_RADIUS 內的像素做中位數，反覆收斂 3 次 → 主色。
    前提：正常顏色是畫面中佔比最大的單一色群。
    """
    px = lab.reshape(-1, 3)
    lo = px.min(axis=0)
    idx = np.floor((px - lo) / HIST_BIN).astype(np.int64)
    shape = idx.max(axis=0) + 1
    raw = np.bincount(np.ravel_multi_index(idx.T, shape),
                      minlength=int(np.prod(shape))).reshape(shape)
    smooth = ndimage.uniform_filter(raw.astype(float), size=3, mode="constant")
    peak = np.array(np.unravel_index(int(smooth.argmax()), shape))
    # 平滑後峰值附近可能多格同值（例如無雜訊影像全落在同一格），回到原始直方圖取該鄰域真正最多的格子
    lo_i, hi_i = np.maximum(peak - 1, 0), np.minimum(peak + 2, shape)
    nb = raw[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1], lo_i[2]:hi_i[2]]
    best = lo_i + np.array(np.unravel_index(int(nb.argmax()), nb.shape))
    center = lo + (best + 0.5) * HIST_BIN
    for _ in range(3):
        near = px[np.linalg.norm(px - center, axis=1) < CLUSTER_RADIUS]
        if len(near) == 0:
            break
        center = np.median(near, axis=0)
    return center


def detect_spots(lab: np.ndarray, size: tuple[int, int], ref_lab: np.ndarray,
                 threshold: float, min_area: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """第二階段：逐像素比對參考色，找出局部瑕疵（如單一小白點）。

    lab 為 working_lab() 的結果；size 為輸入影像 (H, W)。
    1. 每個像素計算與參考色的 ΔE00，超過 threshold 者標為瑕疵像素。
    2. 相鄰（8 連通）瑕疵像素合併成一塊；面積小於 min_area（輸入影像像素²）的視為灰塵忽略。
    回傳 (結果, 逐像素 ΔE, 瑕疵編號圖)。結果中的座標與面積皆換算回輸入影像的像素單位；
    瑕疵編號圖與 lab 同尺寸，像素值 = items 的排序編號（1 起算），0 = 非瑕疵。
    """
    H, W = size
    de = delta_e_2000(lab, ref_lab)

    # 開運算（先侵蝕再膨脹）：去除細碎、不連續的雜訊像素，保留實心的瑕疵
    mask = ndimage.binary_opening(de > threshold, structure=ndimage.generate_binary_structure(2, 1))
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sx, sy = W / lab.shape[1], H / lab.shape[0]
    px_area = sx * sy
    items = []
    if n:
        idx = np.arange(1, n + 1)
        areas = ndimage.sum_labels(np.ones_like(de), labels, idx) * px_area
        maxes = ndimage.maximum(de, labels, idx)
        means = ndimage.mean(de, labels, idx)
        centers = ndimage.center_of_mass(mask, labels, idx)
        mean_lab = np.stack([ndimage.mean(lab[..., k], labels, idx) for k in range(3)], axis=-1)
        for i, sl in enumerate(ndimage.find_objects(labels)):
            if areas[i] < min_area:
                continue
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            ratio = areas[i] / (H * W)
            cx, cy = centers[i][1] * sx + sx / 2, centers[i][0] * sy + sy / 2
            items.append({
                "_label": i + 1,
                # 等面積圓直徑佔長邊的百分比：與照片解析度無關，可與 min_spot_size 直接比較
                "size_pct": round(float(np.sqrt(4 * areas[i] / np.pi) / max(H, W) * 100), 3),
                "type": "region" if ratio >= REGION_RATIO else "spot",
                "tone": classify_tone(mean_lab[i], ref_lab),
                "position": position_name(cx / W, cy / H),
                "center": [round(cx), round(cy)],
                "bbox": [round(x0 * sx), round(y0 * sy), round((x1 - x0) * sx), round((y1 - y0) * sy)],
                "area_px": round(float(areas[i])),
                "area_ratio": round(float(ratio), 6),
                "delta_e_max": round(float(maxes[i]), 3),
                "delta_e_mean": round(float(means[i]), 3),
            })
    items.sort(key=lambda d: d["delta_e_max"] * d["area_px"], reverse=True)
    item_map = np.zeros(de.shape, np.int32)
    for rank, it in enumerate(items, 1):
        lbl = it.pop("_label")
        if rank <= MAX_SPOTS_LISTED:
            item_map[labels == lbl] = rank

    de_max = max((d["delta_e_max"] for d in items), default=0.0)
    return {
        "threshold": threshold,
        "min_area_px": round(min_area, 1),
        # 本張照片解析度下可穩定偵測的最小瑕疵（長邊百分比）；照片越小，此值越大
        "detectable_min_pct": round(MIN_DETECT_PX / max(lab.shape[:2]) * 100, 2),
        "count": len(items),
        "spot_count": sum(d["type"] == "spot" for d in items),
        "region_count": sum(d["type"] == "region" for d in items),
        "defect_area_ratio": round(sum(d["area_ratio"] for d in items), 6),
        "delta_e_max": round(de_max, 3),
        "passed": not items,
        "items": items[:MAX_SPOTS_LISTED],
    }, de, item_map


def analyze(rgb: np.ndarray, rows: int = 5, cols: int = 5, threshold: float = 3.0,
            reference_hex: str | None = None, margin: float = 0.1,
            spot_threshold: float | None = None, min_spot_size: float = 0.5) -> UniformityResult:
    """rgb: (H, W, 3) uint8 影像。分兩階段判定：

    第一階段（整體）：
    - 影像切成 rows×cols 格；每格只取中心 (1-2*margin) 區域，避免邊界過渡色。
    - 每格以中位數取代表色，看出大範圍的漸層、單側泛白等整體不均。
    - 參考色：未指定時以 estimate_reference() 從影像本身找出主色（每張圖各自的基準）。
    - 最大 ΔE00 ≤ threshold 為整體通過（grid_passed）。

    第二階段（小點）：見 detect_spots，以同一參考色逐像素找局部瑕疵；
    spot_threshold 未指定時沿用 threshold。

    兩階段皆通過才算 passed；score 取兩者較低者。
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("影像須為 RGB 三通道")
    if not (1 <= rows <= 50 and 1 <= cols <= 50):
        raise ValueError("rows 與 cols 須介於 1–50")
    if rows * cols < 2:
        raise ValueError("至少需要 2 格才能比較均勻度")
    if not 0 <= margin < 0.5:
        raise ValueError("margin 須介於 0 與 0.5 之間")
    if threshold <= 0 or (spot_threshold is not None and spot_threshold <= 0):
        raise ValueError("threshold 與 spot_threshold 須大於 0")
    if min_spot_size < 0:
        raise ValueError("min_spot_size 不可為負")
    H, W = rgb.shape[:2]
    if H < rows * 2 or W < cols * 2:
        raise ValueError(f"影像太小（{W}×{H}），無法切成 {cols}×{rows} 格")

    ys = np.linspace(0, H, rows + 1).astype(int)
    xs = np.linspace(0, W, cols + 1).astype(int)
    cell_rgb = np.empty((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            dy, dx = int((y1 - y0) * margin), int((x1 - x0) * margin)
            patch = rgb[y0 + dy:y1 - dy, x0 + dx:x1 - dx].reshape(-1, 3)
            cell_rgb[r, c] = np.median(patch, axis=0)

    cell_lab = srgb_to_lab(cell_rgb)
    work = working_lab(rgb)
    if reference_hex:
        ref_lab = srgb_to_lab(np.array(hex_to_rgb(reference_hex), float))
        ref_source = f"指定色 {reference_hex}"
    else:
        ref_lab = estimate_reference(work)
        ref_source = "影像主色（自動）"
    s_thr = spot_threshold if spot_threshold is not None else threshold
    # 主色覆蓋率：與參考色 ΔE00 ≤ 小點門檻的像素比例
    coverage = float(np.mean(delta_e_2000(work, ref_lab) <= s_thr))
    ref_hex = "#" + "".join(f"{round(v):02X}" for v in lab_to_srgb(ref_lab))

    de = delta_e_2000(cell_lab, ref_lab)
    worst = np.unravel_index(int(np.argmax(de)), de.shape)

    # 亮度均勻度（顯示器/照明常用）：最暗格 Y / 最亮格 Y
    lin = np.where(cell_rgb / 255 <= 0.04045, cell_rgb / 255 / 12.92,
                   ((cell_rgb / 255 + 0.055) / 1.055) ** 2.4)
    Y = lin @ _M_RGB2XYZ[1]
    lum_u = float(Y.min() / Y.max()) if Y.max() > 0 else 1.0

    de_max = float(de.max())
    grid_passed = de_max <= threshold

    # 第二階段：整體判定後，再逐像素找單一小點等局部瑕疵
    # 最小瑕疵尺寸以「長邊百分比」表示（等面積圓直徑），換算成本張影像的像素面積；照片大小不同仍是同一標準
    min_area = np.pi / 4 * (min_spot_size / 100 * max(H, W)) ** 2
    spots, pixel_de, item_map = detect_spots(work, (H, W), ref_lab, s_thr, min_area)
    spots["min_spot_size_pct"] = min_spot_size
    score = min(_score(de_max, threshold), _score(spots["delta_e_max"], s_thr))

    rnd = lambda x, n=3: round(float(x), n)
    return UniformityResult(
        rows=rows, cols=cols,
        reference_lab=[rnd(v) for v in ref_lab],
        reference_hex=ref_hex,
        reference_source=ref_source,
        reference_coverage=round(coverage, 4),
        delta_e_grid=[[rnd(v) for v in row] for row in de],
        cell_lab_grid=[[[rnd(v, 2) for v in cell] for cell in row] for row in cell_lab],
        delta_e_max=rnd(de_max),
        delta_e_mean=rnd(de.mean()),
        delta_e_p95=rnd(np.percentile(de, 95)),
        lightness_std=rnd(cell_lab[..., 0].std()),
        luminance_uniformity=rnd(lum_u, 4),
        worst_cell={"row": int(worst[0]), "col": int(worst[1]), "delta_e": rnd(de_max),
                    "tone": classify_tone(cell_lab[worst], ref_lab),
                    "position": position_name((worst[1] + 0.5) / cols, (worst[0] + 0.5) / rows)},
        cells_over_threshold=int((de > threshold).sum()),
        threshold=threshold,
        grid_passed=grid_passed,
        spots=spots,
        passed=grid_passed and spots["passed"],
        score=round(score, 1),
        pixel_delta_e=pixel_de,
        defect_map=item_map,
    )
