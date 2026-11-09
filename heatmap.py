"""把判定結果輸出成一張熱圖報告 PNG。

版面：
  標題列（檔名、site、判定結果）
  [原圖＋瑕疵框] [第一階段：格子 ΔE 熱圖] [第二階段：逐像素 ΔE 熱圖]
  色階（固定 0–SCALE_MAX，不同圖片可直接比較；標出兩階段門檻）
  綜合說明 summary
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# 單一色相（藍）由淺到深的連續色階：淺 = 色差小（均勻），深 = 色差大（不均勻）
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SCALE_MAX = 20.0        # 色階上限 ΔE00；≥20 皆以最深色呈現。須高於小點門檻（目前 10），才分得出門檻上下的輕重
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_LINE = "#e3e2de"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

PANEL_W = 460
PAD = 28
GAP = 24

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msjh.ttc",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], float)


_RAMP_RGB = np.stack([_hex(c) for c in RAMP])


def colorize(de: np.ndarray) -> np.ndarray:
    """ΔE 陣列 → RGB（線性插值於藍色色階）。"""
    t = np.clip(np.asarray(de, float) / SCALE_MAX, 0, 1) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    f = (t - i)[..., None]
    return (_RAMP_RGB[i] * (1 - f) + _RAMP_RGB[i + 1] * f).astype(np.uint8)


def _ink_on(rgb) -> str:
    """色塊上的文字：淺底用深字、深底用白字。"""
    r, g, b = (np.asarray(rgb, float) / 255)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.42 else "#ffffff"


def _wrap(text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        # 標點不放行首（避頭點），允許略為超出寬度
        if font.getlength(cur + ch) > width and ch not in "，。、；：！？）」』%":
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


def _tag(draw, x, y, n, font, bounds):
    """編號標籤：白底黑框，限制在面板範圍內。"""
    tag = str(n)
    w = font.getlength(tag) + 10
    bx0, by0, bx1, by1 = bounds
    x = min(max(x, bx0), bx1 - w)
    y = min(max(y, by0), by1 - 19)
    draw.rounded_rectangle([x, y, x + w, y + 19], radius=4, fill="#ffffff", outline=INK)
    draw.text((x + 5, y + 1), tag, fill=INK, font=font)


def _contours(defect_map: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """大面積瑕疵的輪廓線（白色內線＋黑色外線，任何底色都看得見），RGBA 疊圖。"""
    m = np.asarray(Image.fromarray((defect_map > 0).astype(np.uint8) * 255)
                   .resize(size, Image.Resampling.NEAREST)) > 0
    inner = m & ~ndimage.binary_erosion(m, iterations=2)
    outer = ndimage.binary_dilation(m, iterations=1) & ~m
    rgba = np.zeros((size[1], size[0], 4), np.uint8)
    rgba[outer] = (11, 11, 11, 255)
    rgba[inner] = (255, 255, 255, 255)
    return Image.fromarray(rgba, "RGBA")


SMALL_PX = 24   # 面板上小於此尺寸的瑕疵，另加方框以免看不到


def _mark_defects(canvas, draw, items, defect_map, sx, sy, ox, oy, pw, ph, font):
    """所有瑕疵沿輪廓描邊；太小的另加方框。皆標上與 JSON items 相同的編號。"""
    if (defect_map > 0).any():
        overlay = _contours(defect_map, (pw, ph))
        canvas.paste(overlay, (ox, oy), overlay)
    bounds = (ox, oy, ox + pw, oy + ph)
    for n, it in enumerate(items, 1):
        x, y, w, h = it["bbox"]
        if max(w * sx, h * sy) < SMALL_PX:
            x0, y0 = ox + x * sx, oy + y * sy
            x1, y1 = x0 + max(w * sx, 6), y0 + max(h * sy, 6)
            draw.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline="#ffffff", width=3)
            draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=INK, width=1)
            _tag(draw, x0 - 3, y0 - 24, n, font, bounds)
        else:
            cx, cy = it["_center"]
            _tag(draw, ox + cx * sx - 8, oy + cy * sy - 10, n, font, bounds)


def _badge_icon(draw, x, y, ok):
    """判定圖示（自行繪製，不依賴字型）：✓ 或 ✕。"""
    if ok:
        draw.line([(x, y + 10), (x + 6, y + 17), (x + 18, y + 2)], fill="#ffffff", width=4, joint="curve")
    else:
        draw.line([(x + 2, y + 2), (x + 17, y + 17)], fill="#ffffff", width=4)
        draw.line([(x + 17, y + 2), (x + 2, y + 17)], fill="#ffffff", width=4)


def render_report(img: Image.Image, result: dict, de_map: np.ndarray, defect_map: np.ndarray,
                  path: Path, to_analyzed=lambda b: b) -> Path:
    """img：實際分析的影像；result：API 回傳內容；de_map：逐像素 ΔE；defect_map：瑕疵編號圖
    （兩者皆為工作解析度）。to_analyzed：把回傳中的原圖 bbox 換回 img 座標的函式。"""
    f_title, f_head, f_body, f_small = _font(26), _font(18), _font(16), _font(14)
    W0, H0 = img.size
    ph = round(PANEL_W * H0 / W0)
    sx = sy = PANEL_W / W0
    width = PAD * 2 + PANEL_W * 3 + GAP * 2

    summary_lines = _wrap(result["summary"], f_body, width - PAD * 2)
    top = PAD + 44 + 34                       # 標題 + 欄位標題
    legend_y = top + ph + 30
    summary_y = legend_y + 70
    height = summary_y + len(summary_lines) * 26 + PAD

    canvas = Image.new("RGB", (width, height), SURFACE)
    d = ImageDraw.Draw(canvas)

    # 標題列：判定以「圖示＋文字」呈現，不只靠顏色
    ok = result["passed"]
    badge = "Pass" if ok else "Fail"
    bw = f_title.getlength(badge) + 58
    d.rounded_rectangle([PAD, PAD, PAD + bw, PAD + 38], radius=6, fill=GOOD if ok else CRITICAL)
    _badge_icon(d, PAD + 13, PAD + 10, ok)
    _, t, _, b = d.textbbox((0, 0), badge, font=f_title)
    d.text((PAD + 42, PAD + 19 - (t + b) / 2), badge, fill="#ffffff", font=f_title)  # 依實際字高垂直置中
    d.text((PAD + bw + 16, PAD + 8),
           f"{result['filename']}　·　site：{result['site']}　·　基準色 {result['reference_hex']}",
           fill=INK, font=f_head)

    xs = [PAD + i * (PANEL_W + GAP) for i in range(3)]
    items = [{**it, "bbox": to_analyzed(it["bbox"]),
              "_center": to_analyzed([*it["center"], 0, 0])[:2]} for it in result["spots"]["items"]]
    grid_ok = "通過" if result["grid_passed"] else "不通過"
    spot_ok = "通過" if result["spots"]["passed"] else "不通過"
    detrended = result.get("grid_mode") == "detrended"
    local = result["spots"].get("mode") == "local"
    heads = ["原圖（描邊 = 局部瑕疵）",
             f"第一階段 {'扣除漸層' if detrended else '整體'} {result['rows']}×{result['cols']} 格 ΔE｜{grid_ok}",
             f"第二階段 逐像素 ΔE{'（對周圍）' if local else ''}｜{spot_ok}（{result['spots']['count']} 處）"]
    for x, h in zip(xs, heads):
        d.text((x, top - 28), h, fill=INK, font=f_head)

    # 1. 原圖
    canvas.paste(img.resize((PANEL_W, ph), Image.Resampling.LANCZOS), (xs[0], top))
    _mark_defects(canvas, d, items, defect_map, sx, sy, xs[0], top, PANEL_W, ph, f_small)

    # 2. 格子熱圖：每格填色＋數值；超過門檻的格子加粗框
    grid = np.array(result["delta_e_grid"])
    R, C = grid.shape
    thr = result["threshold"]
    ys_ = np.linspace(0, ph, R + 1)
    xs_ = np.linspace(0, PANEL_W, C + 1)
    colors = colorize(grid)
    cell_font = _font(max(11, min(18, int(min(PANEL_W / C, ph / R) / 3.2))))
    for r in range(R):
        for c in range(C):
            x0, y0 = xs[1] + xs_[c], top + ys_[r]
            x1, y1 = xs[1] + xs_[c + 1], top + ys_[r + 1]
            d.rectangle([x0, y0, x1, y1], fill=tuple(colors[r, c]), outline=SURFACE, width=2)
            v = grid[r, c]
            label = f"{v:.1f}" if v < 100 else f"{v:.0f}"
            tw = cell_font.getlength(label)
            d.text(((x0 + x1 - tw) / 2, (y0 + y1) / 2 - cell_font.size * 0.6), label,
                   fill=_ink_on(colors[r, c]), font=cell_font)
            if v > thr:
                d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=INK, width=3)

    # 3. 逐像素熱圖
    pix = Image.fromarray(colorize(de_map)).resize((PANEL_W, ph), Image.Resampling.BILINEAR)
    canvas.paste(pix, (xs[2], top))
    _mark_defects(canvas, d, items, defect_map, sx, sy, xs[2], top, PANEL_W, ph, f_small)

    for x in xs:
        d.rectangle([x - 1, top - 1, x + PANEL_W, top + ph], outline=GRID_LINE)

    # 色階：固定 0–SCALE_MAX，標出兩階段門檻
    lx, lw, lh = PAD, 520, 14
    bar = colorize(np.tile(np.linspace(0, SCALE_MAX, lw), (lh, 1)))
    canvas.paste(Image.fromarray(bar), (lx, legend_y))
    d.text((lx, legend_y - 24), "色差 ΔE00（淺 = 均勻，深 = 不均勻）", fill=INK_2, font=f_small)
    for v in range(0, int(SCALE_MAX) + 1, max(1, int(SCALE_MAX) // 5)):
        tx = lx + v / SCALE_MAX * (lw - 1)
        d.line([tx, legend_y + lh, tx, legend_y + lh + 4], fill=INK_2)
        lab = f"≥{v}" if v == SCALE_MAX else str(v)
        d.text((tx - f_small.getlength(lab) / 2, legend_y + lh + 6), lab, fill=INK_2, font=f_small)
    marks = [("整體門檻", thr), ("小點門檻", result["spots"]["threshold"])]
    if marks[0][1] == marks[1][1]:
        marks = [("門檻", thr)]
    for k, (name, v) in enumerate(marks):
        tx = lx + min(v, SCALE_MAX) / SCALE_MAX * (lw - 1)
        d.line([tx, legend_y - 3, tx, legend_y + lh + 3], fill=INK, width=2)
        ky = legend_y - 2 + k * 20
        d.line([lx + lw + 20, ky, lx + lw + 20, ky + 16], fill=INK, width=2)
        d.text((lx + lw + 28, ky - 2), f"{name} ΔE {v:g}", fill=INK, font=f_small)
    d.text((lx + lw + 180, legend_y - 4), "粗黑框格 = 超過整體門檻", fill=INK_2, font=f_small)
    d.text((lx + lw + 180, legend_y + 16), "描邊 = 局部瑕疵；方框 = 過小的瑕疵；編號同 JSON items 順序",
           fill=INK_2, font=f_small)

    # 綜合說明
    for i, line in enumerate(summary_lines):
        d.text((PAD, summary_y + i * 26), line, fill=INK, font=f_body)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def safe_stem(filename: str | None) -> str:
    """上傳檔名 → 安全的輸出檔名主體（去除路徑與特殊字元，防止寫到 result 資料夾外）。"""
    stem = Path(filename or "image").stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("._") or "image"
    return stem[:80]
