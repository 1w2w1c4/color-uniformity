"""產生「全紅但因壓力不均而局部泛白」的測試影像到 images/。執行：.venv/bin/python make_samples.py

模型：每個像素有壓力值 p∈[0,1]，p=1 呈飽和紅、p 越低越接近白色，並加上細微顆粒雜訊。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

W, H = 1200, 900
RED = np.array([200, 22, 34], float)
WHITE = np.array([246, 240, 238], float)
OUT = Path(__file__).resolve().parent / "images"
rng = np.random.default_rng(42)
yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]


def smooth_noise(cells: int) -> np.ndarray:
    """低解析度亂數放大成平滑雜訊，值域 0–1。"""
    small = Image.fromarray((rng.random((cells, cells)) * 255).astype(np.uint8))
    return np.asarray(small.resize((W, H), Image.Resampling.BICUBIC), float) / 255


def blob(cx, cy, r) -> np.ndarray:
    """以 (cx, cy) 為中心、半徑 r（相對尺寸）的高斯低壓區，值域 0–1。"""
    return np.exp(-(((xx - cx) * W / H) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))


def render(pressure: np.ndarray, name: str) -> None:
    p = np.clip(pressure, 0, 1)[..., None] ** 1.6           # 低壓時快速褪色
    rgb = WHITE + (RED - WHITE) * p
    rgb += rng.normal(0, 4, rgb.shape)                       # 顯色顆粒
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img.save(OUT / name)
    print(f"{name}: 紅色區 {np.mean(p > 0.8):.0%}，泛白區 {np.mean(p < 0.3):.0%}")


OUT.mkdir(exist_ok=True)

# 1. 單側壓力不足：壓板傾斜，右側越來越白，邊界帶點不規則
edge = 1 - np.clip((xx - 0.55) / 0.4 + (smooth_noise(6) - 0.5) * 0.5, 0, 1)
render(edge, "red_uneven_1_edge.png")

# 2. 中央壓力不足：壓板中央內凹，中心大片泛白，另有兩處小白點
center = 1 - np.clip(blob(0.5, 0.5, 0.18) * 1.1 + blob(0.15, 0.2, 0.05) * 0.8
                     + blob(0.85, 0.78, 0.04) * 0.7, 0, 1)
center -= (smooth_noise(10) - 0.5) * 0.15
render(center, "red_uneven_2_center.png")

# 3. 斑駁點狀：表面不平整，散布多處大小不一的泛白斑
mottle = smooth_noise(14) * 0.6 + smooth_noise(30) * 0.4
render(np.clip((mottle - 0.32) / 0.25, 0, 1), "red_uneven_3_mottled.png")

# 4. 單一小白點：整體均勻，只有一處局部壓力不足。
#    直徑約 24px、邊緣 6px 漸層，刻意落在 5×5 格線交點 (720, 540)——舊版格子判定會漏掉的位置。
dist = np.hypot(xx * W - 720, yy * H - 540)
render(np.clip((dist - 9) / 6, 0, 1), "red_uneven_4_single_spot.png")
