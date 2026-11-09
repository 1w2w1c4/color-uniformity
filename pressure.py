"""感壓紙壓力均勻度分析（掃描影像專用），與 uniformity.py 的色差判定互為另一種方式。

原理：感壓紙顏色越深代表壓力越大。不比較「顏色差多少」，而是把顏色換算成一個濃度值，
再分析濃度（≈ 壓力）分布：

1. 濃度：取互補色通道（紅／洋紅色感壓紙 → 綠色通道）算光學濃度 OD = -log10(I / I_白紙)。
   白紙基準自動取畫面中未受壓的白紙；整張都是受壓區時改用預設白紙色，也可用 white 指定。
2. 相對壓力：各像素濃度 ÷ 受壓區濃度中位數。型號未知、沒有校正曲線時，只判斷「相對」均勻度；
   濃度與壓力為單調但非線性關係，相對值是壓力的代理指標。
3. 判定指標：CV（std ÷ mean）、p5 ÷ p95、接觸面積比。
4. 形狀拆解：對濃度圖做二次曲面擬合 →
   平面部分 = 傾斜（平行度不良）；二次部分 = 中凸／中凹（壓板變形）；殘差 = 局部低壓點／高壓點（異物、凸點）。
5. 掃描器 DPI 已知 → 瑕疵大小以 mm 表示。DPI 讀自影像檔，讀不到時用 dpi 參數（預設 300）。

執行：.venv/bin/python pressure.py images/scan.png [--dpi 600] [--roi x,y,w,h] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from heatmap import GOOD, CRITICAL, INK, INK_2, PAD, SURFACE, _font, _ink_on, _wrap, safe_stem
from uniformity import hex_to_rgb, position_name, srgb_to_lab

WORK_MAX_SIDE = 1000          # 工作解析度上限（長邊）
DEFAULT_DPI = 300.0
DEFAULT_WHITE = (246, 240, 238)  # 找不到未受壓白紙時的預設白紙色
GRAIN_SIGMA_MM = 0.15         # 抑制感壓紙顯色顆粒的高斯模糊（mm）
BACKGROUND_MM = 5.0           # 局部瑕疵比較的局部背景範圍（高斯 σ，mm）
MIN_DETECT_PX = 2.5          # 工作解析度下可穩定分辨的最小瑕疵直徑（像素）
SCANNER_CLIP = 2              # 通道值 ≤ 此值視為掃描器截斷
MAX_DEFECTS_LISTED = 50
CHANNELS = {"R": 0, "G": 1, "B": 2}


@dataclass
class PressureSettings:
    max_cv: float = 0.15            # 相對壓力變異係數上限（15%）
    min_uniformity: float = 0.70    # p5 ÷ p95 下限
    min_contact: float = 0.95       # 接觸面積比下限
    max_tilt: float = 0.20          # 傾斜：擬合平面最高與最低處差 ÷ 平均，上限 20%
    max_bow: float = 0.20           # 中凸／中凹：中央與邊緣差 ÷ 平均，上限 20%
    local_tol: float = 0.30         # 局部瑕疵：殘差超過平均濃度的 30%
    min_defect_mm: float = 1.0      # 小於此等面積圓直徑（mm）的局部瑕疵視為雜訊忽略
    contact_od: float = 0.08        # 濃度高於此值視為有接觸（受壓）
    saturation_od: float = 1.2      # 濃度高於此值視為可能接近飽和
    rows: int = 3                   # 分區統計（3×3 = 業界常用的 9 點量測）
    cols: int = 3


# ---------------------------------------------------------------- 濃度換算

def _work_image(img: Image.Image) -> tuple[np.ndarray, float]:
    """縮到工作解析度（BOX 平均，順便壓顆粒），回傳 (RGB float, 縮放倍率)。"""
    W, H = img.size
    scale = min(1.0, WORK_MAX_SIDE / max(W, H))
    if scale < 1:
        img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float64), scale


def find_white(rgb: np.ndarray, white: str | None) -> tuple[np.ndarray, str]:
    """白紙基準：指定值 > 畫面中未受壓白紙（高亮度、近中性、佔 ≥0.5%）> 預設白紙色。"""
    if white:
        return np.array(hex_to_rgb(white), float), "指定"
    lab = srgb_to_lab(rgb)
    paper = (lab[..., 0] > 88) & (np.hypot(lab[..., 1], lab[..., 2]) < 10)
    if paper.mean() >= 0.005:
        return np.median(rgb[paper], axis=0), "畫面中的未受壓白紙"
    return np.array(DEFAULT_WHITE, float), "預設白紙色（畫面中沒有未受壓白紙）"


def optical_density(rgb: np.ndarray, white_rgb: np.ndarray, channel: str = "auto") -> tuple[np.ndarray, str]:
    """回傳 (OD 圖, 採用的通道)。auto：選濃度動態範圍最大的通道（紅色感壓紙通常是 G）。"""
    ods = {}
    for name, i in CHANNELS.items():
        ratio = np.clip(rgb[..., i] / max(white_rgb[i], 1.0), 1e-3, 1.0)
        ods[name] = -np.log10(ratio)
    if channel == "auto":
        channel = max(ods, key=lambda k: np.percentile(ods[k], 95))
    if channel not in ods:
        raise ValueError(f"channel 需為 auto/R/G/B，收到 {channel!r}")
    return ods[channel], channel


# ---------------------------------------------------------------- 形狀拆解

def _design(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], axis=-1)


def fit_surface(od: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """二次曲面穩健擬合（排除離群點後再擬合一次）。回傳 (係數, 擬合曲面)。
    座標正規化到 [-1, 1]，係數 [c, x, y, x², xy, y²]。"""
    H, W = od.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xn = 2 * xx / max(W - 1, 1) - 1
    yn = 2 * yy / max(H - 1, 1) - 1
    A = _design(xn, yn)
    use = mask.copy()
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(A[use], od[use], rcond=None)
        res = od - A @ coef
        s = 1.4826 * np.median(np.abs(res[mask] - np.median(res[mask]))) + 1e-9
        use = mask & (np.abs(res) < 3 * s)
        if use.sum() < 10:
            break
    return coef, A @ coef


def _direction(vx: float, vy: float) -> str:
    """向量方向 → 方位文字（影像座標，y 向下）。"""
    ang = np.degrees(np.arctan2(vy, vx)) % 360
    names = ["右側", "右下方", "下方", "左下方", "左側", "左上方", "上方", "右上方"]
    return names[int(((ang + 22.5) % 360) // 45)]


def decompose(coef: np.ndarray, mean_od: float, mask: np.ndarray) -> dict:
    """由二次曲面係數算出傾斜與中凸／中凹。"""
    c, bx, by, qxx, qxy, qyy = coef
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xn = 2 * xx / max(W - 1, 1) - 1
    yn = 2 * yy / max(H - 1, 1) - 1
    plane = c + bx * xn + by * yn
    tilt = float((plane[mask].max() - plane[mask].min()) / mean_od) if mask.any() else 0.0
    quad = qxx * xn ** 2 + qxy * xn * yn + qyy * yn ** 2
    border = mask & ~ndimage.binary_erosion(mask, iterations=max(1, min(H, W) // 20))
    cy, cx = ndimage.center_of_mass(mask)
    center = quad[int(round(cy)), int(round(cx))]
    bow = float((center - quad[border].mean()) / mean_od) if border.any() else 0.0
    return {
        "tilt": round(tilt, 4),
        "tilt_low_side": _direction(-bx, -by) if tilt > 0.01 else None,
        "bow": round(bow, 4),
        "bow_shape": None if abs(bow) <= 0.01 else ("中凸（中央高、邊緣低）" if bow > 0 else "中凹（中央低、邊緣高）"),
    }


# ---------------------------------------------------------------- 局部瑕疵

def local_background(v: np.ndarray, mean_od: float, tol: float, sigma_px: float) -> np.ndarray:
    """局部背景：高斯加權的局部平面迴歸（在影像邊界也能跟上漸層，不會把邊角誤判為瑕疵），
    並迭代排除離群點，瑕疵本身與鄰近深坑不會把背景拉偏。"""
    f = max(1, int(sigma_px / 4))        # 背景是大尺度平滑量：縮小 f 倍計算再放大，結果幾乎相同、快很多
    if f > 1:
        H0, W0 = v.shape
        Hs, Ws = H0 // f, W0 // f
        small = v[:Hs * f, :Ws * f].reshape(Hs, f, Ws, f).mean(axis=(1, 3))
        bg = local_background(small, mean_od, tol, sigma_px / f)
        return ndimage.zoom(bg, (H0 / Hs, W0 / Ws), order=1, mode="nearest", grid_mode=True)
    H, W = v.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx - W / 2) / sigma_px
    y = (yy - H / 2) / sigma_px
    g = lambda a: ndimage.gaussian_filter(a, sigma_px, mode="constant", cval=0.0)
    w = np.ones_like(v)
    bg = v
    for _ in range(3):
        m = [g(w), g(w * x), g(w * y), g(w * x * x), g(w * x * y), g(w * y * y)]
        A = np.stack([np.stack([m[0], m[1], m[2]], -1),
                      np.stack([m[1], m[3], m[4]], -1),
                      np.stack([m[2], m[4], m[5]], -1)], -2) + np.eye(3) * 1e-6
        rhs = np.stack([g(w * v), g(w * x * v), g(w * y * v)], -1)[..., None]
        coef = np.linalg.solve(A, rhs)[..., 0]
        bg = coef[..., 0] + coef[..., 1] * x + coef[..., 2] * y
        w = (np.abs(v - bg) < tol * mean_od).astype(float)
    return bg


def find_defects(od: np.ndarray, fit: np.ndarray, contact: np.ndarray, mask: np.ndarray,
                 mean_od: float, s: PressureSettings, mm_per_px: float, scale: float,
                 offset: tuple[int, int]) -> tuple[list[dict], np.ndarray]:
    """局部瑕疵 = 與周圍 BACKGROUND_MM 範圍的局部背景相比偏低／偏高；大範圍形狀由 CV、傾斜、中凸凹負責。"""
    detrended = od - fit                 # 先扣掉大範圍曲面，影像邊角的漸層才不會被當成局部瑕疵
    bg = local_background(detrended, mean_od, s.local_tol, BACKGROUND_MM / mm_per_px)
    res = (detrended - bg) / mean_od
    no_contact = mask & ~contact
    kinds = {
        "低壓點": no_contact | (mask & (res < -s.local_tol)),   # 未接觸與周圍低壓合併為同一處
        "高壓點": mask & contact & (res > s.local_tol),
    }
    H, W = od.shape
    label_map = np.zeros(od.shape, int)
    items = []
    min_area = np.pi / 4 * (s.min_defect_mm / mm_per_px) ** 2
    for kind, m in kinds.items():
        m = ndimage.binary_opening(m, iterations=1)
        lab, n = ndimage.label(m)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            comp = lab[sl] == i
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(comp)
            cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
            vals = res[sl][comp]
            nc = float(no_contact[sl][comp].mean())
            items.append({
                "type": "未接觸" if nc >= 0.5 else kind,
                "no_contact_ratio": round(nc, 3),
                "position": position_name(cx / W, cy / H),
                "center": [round(cx / scale) + offset[0], round(cy / scale) + offset[1]],
                "bbox": [round(sl[1].start / scale) + offset[0], round(sl[0].start / scale) + offset[1],
                         round((sl[1].stop - sl[1].start) / scale), round((sl[0].stop - sl[0].start) / scale)],
                "size_mm": round(2 * np.sqrt(area / np.pi) * mm_per_px, 2),
                "area_mm2": round(area * mm_per_px ** 2, 2),
                "deviation": round(float(vals.min() if kind != "高壓點" else vals.max()), 3),
                "_mask": (sl, comp),
            })
    items.sort(key=lambda d: -d["area_mm2"])
    for n, it in enumerate(items, start=1):
        sl, comp = it.pop("_mask")
        if n <= MAX_DEFECTS_LISTED:
            label_map[sl][comp] = n
    return items[:MAX_DEFECTS_LISTED], label_map


# ---------------------------------------------------------------- 主流程

def analyze_pressure(img: Image.Image, dpi: float | None = None, roi: tuple[int, int, int, int] | None = None,
                     white: str | None = None, channel: str = "auto",
                     settings: PressureSettings | None = None) -> tuple[dict, dict]:
    """回傳 (結果 dict, 繪圖用的中間資料)。"""
    s = settings or PressureSettings()
    img = ImageOps.exif_transpose(img).convert("RGB") if img.mode != "RGB" else img
    file_dpi = img.info.get("dpi")
    if dpi:
        dpi_used, dpi_src = float(dpi), "指定"
    elif file_dpi and float(file_dpi[0]) > 1:
        dpi_used, dpi_src = round(float(file_dpi[0]), 1), "影像檔"
    else:
        dpi_used, dpi_src = DEFAULT_DPI, f"預設 {DEFAULT_DPI:g}（影像檔沒有 DPI 資訊）"

    full = img
    ox = oy = 0
    if roi:
        x, y, w, h = roi
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img.width or y + h > img.height:
            raise ValueError(f"roi {roi} 超出影像範圍 {img.width}×{img.height}")
        img = img.crop((x, y, x + w, y + h))
        ox, oy = x, y

    raw = np.asarray(img)
    rgb, scale = _work_image(img)
    mm_per_px = 25.4 / dpi_used / scale

    white_rgb, white_src = find_white(rgb, white)
    od, ch = optical_density(rgb, white_rgb, channel)
    od = ndimage.gaussian_filter(od, max(GRAIN_SIGMA_MM / mm_per_px, 0.5))

    mask = np.ones(od.shape, bool)
    contact = od > s.contact_od
    if contact.sum() < 10:
        raise ValueError("幾乎沒有受壓區（整張接近白紙），無法分析")

    mean_od = float(np.median(od[contact]))
    rel = od / mean_od
    vals = rel[contact]
    cv = float(vals.std() / vals.mean())
    p5, p95 = np.percentile(vals, [5, 95])
    uni = float(p5 / p95)
    contact_ratio = float(contact[mask].mean())

    coef, fit = fit_surface(od, mask)
    shape = decompose(coef, mean_od, mask)
    defects, defect_map = find_defects(od, fit, contact, mask, mean_od, s, mm_per_px, scale, (ox, oy))

    H, W = od.shape
    grid = [[round(float(rel[r * H // s.rows:(r + 1) * H // s.rows,
                             c * W // s.cols:(c + 1) * W // s.cols].mean()), 3)
             for c in range(s.cols)] for r in range(s.rows)]

    sat_ratio = float((od[contact] >= s.saturation_od).mean())
    clip_ratio = float((raw[..., CHANNELS[ch]] <= SCANNER_CLIP).mean())
    warnings = []
    if sat_ratio > 0.05:
        warnings.append(f"{sat_ratio:.0%} 受壓區濃度 ≥ {s.saturation_od:g}，可能接近感壓紙飽和，"
                        "實際壓力差異可能被低估；建議改用量測範圍較高的感壓紙。")
    if clip_ratio > 0.01:
        warnings.append(f"{clip_ratio:.1%} 像素已被掃描器截斷（通道值 ≤ {SCANNER_CLIP}），請確認已關閉自動曝光／自動色彩校正。")
    if "預設" in dpi_src:
        warnings.append("影像沒有 DPI 資訊，mm 尺寸以 300 DPI 估算，請用 --dpi 指定實際掃描解析度。")
    if "預設" in white_src:
        warnings.append("畫面中沒有未受壓白紙，濃度以預設白紙色計算；建議掃描時保留一小塊白紙，或用 --white 指定。")

    checks = {
        "cv": {"value": round(cv, 4), "limit": s.max_cv, "passed": cv <= s.max_cv},
        "uniformity_p5_p95": {"value": round(uni, 4), "limit": s.min_uniformity, "passed": uni >= s.min_uniformity},
        "contact_ratio": {"value": round(contact_ratio, 4), "limit": s.min_contact,
                          "passed": contact_ratio >= s.min_contact},
        "tilt": {"value": shape["tilt"], "limit": s.max_tilt, "passed": shape["tilt"] <= s.max_tilt},
        "bow": {"value": shape["bow"], "limit": s.max_bow, "passed": abs(shape["bow"]) <= s.max_bow},
        "local_defects": {"value": len(defects), "limit": 0, "passed": not defects},
    }
    result = {
        "mode": "pressure",
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "shape": shape,
        "relative_grid": grid,
        "relative_min": round(float(vals.min()), 3),
        "relative_max": round(float(vals.max()), 3),
        "density_median": round(mean_od, 4),
        "defects": defects,
        "defect_area_ratio": round(float((defect_map > 0).mean()), 5),
        "saturation_ratio": round(sat_ratio, 4),
        "channel": ch,
        "white_rgb": [round(float(v)) for v in white_rgb],
        "white_source": white_src,
        "dpi": dpi_used,
        "dpi_source": dpi_src,
        "mm_per_px": round(25.4 / dpi_used, 5),
        "detectable_min_mm": round(MIN_DETECT_PX * mm_per_px, 2),
        "settings": asdict(s),
        "warnings": warnings,
    }
    result["summary"] = summarize_pressure(result)
    return result, {"rgb": rgb.astype(np.uint8), "rel": rel, "defect_map": defect_map,
                    "coef": coef, "scale": scale, "offset": (ox, oy)}


# ---------------------------------------------------------------- 中文說明

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def summarize_pressure(r: dict) -> str:
    c, sh = r["checks"], r["shape"]
    parts = ["【Pass】壓力分布均勻。" if r["passed"] else "【Fail】壓力分布不均勻。"]
    parts.append(f"相對壓力 CV {_pct(c['cv']['value'])}（上限 {_pct(c['cv']['limit'])}），"
                 f"p5/p95 = {c['uniformity_p5_p95']['value']:.2f}（下限 {c['uniformity_p5_p95']['limit']:.2f}），"
                 f"接觸面積 {_pct(c['contact_ratio']['value'])}。")
    if sh["tilt_low_side"]:
        parts.append(f"傾斜 {_pct(sh['tilt'])}，{sh['tilt_low_side']}壓力偏低"
                     f"{'（超過上限，疑似平行度不良）' if not c['tilt']['passed'] else ''}。")
    if sh["bow_shape"]:
        parts.append(f"{sh['bow_shape']} {_pct(abs(sh['bow']))}"
                     f"{'（超過上限，疑似壓板變形）' if not c['bow']['passed'] else ''}。")
    d = r["defects"]
    if d:
        kinds = {}
        for it in d:
            kinds[it["type"]] = kinds.get(it["type"], 0) + 1
        w = d[0]
        parts.append("局部瑕疵 " + "、".join(f"{k} {n} 處" for k, n in kinds.items())
                     + f"；最大的是{w['position']}的{w['type']}，約 {w['size_mm']} mm，"
                       f"中心 ({w['center'][0]}, {w['center'][1]})。")
    else:
        parts.append(f"沒有 ≥ {r['settings']['min_defect_mm']:g} mm 的局部瑕疵。")
    parts.append("（相對濃度判定，未換算為 MPa。）")
    return "".join(parts)


# ---------------------------------------------------------------- 報告圖

REL_LOW, REL_MID, REL_HIGH = "#2a78d6", "#f2f1ee", "#d03b3b"   # 低壓藍、平均白、高壓紅


def colorize_rel(rel: np.ndarray, span: float = 0.5) -> np.ndarray:
    """相對壓力 → 發散色階：1 - span 最藍、1 白、1 + span 最紅。"""
    t = np.clip((rel - 1) / span, -1, 1)[..., None]
    lo, mid, hi = (np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], float) for c in (REL_LOW, REL_MID, REL_HIGH))
    out = np.where(t < 0, mid + (lo - mid) * -t, mid + (hi - mid) * t)
    return out.astype(np.uint8)


def render_pressure_report(result: dict, extra: dict, path: Path, title: str = "") -> Path:
    rgb, rel, dmap = extra["rgb"], extra["rel"], extra["defect_map"]
    H, W = rel.shape
    pw = 520
    ph = round(pw * H / W)
    f_title, f_head, f_body, f_small = _font(26), _font(18), _font(16), _font(14)
    summary_lines = _wrap(result["summary"] + " " + " ".join(result["warnings"]), f_body, pw * 2 + 24)
    top = PAD + 60 + 34
    total_w = PAD * 2 + pw * 2 + 24
    total_h = top + ph + 80 + len(summary_lines) * 26 + PAD
    canvas = Image.new("RGB", (total_w, total_h), SURFACE)
    d = ImageDraw.Draw(canvas)

    ok = result["passed"]
    d.rounded_rectangle([PAD, PAD, PAD + 150, PAD + 40], radius=8, fill=GOOD if ok else CRITICAL)
    badge = "Pass" if ok else "Fail"
    _, t, _, b = d.textbbox((0, 0), badge, font=f_title)
    d.text((PAD + 14, PAD + 20 - (t + b) / 2), badge, fill="#ffffff", font=f_title)  # 依實際字高垂直置中
    d.text((PAD + 170, PAD + 10), f"{title}　壓力模式　通道 {result['channel']}　{result['dpi']:g} DPI",
           fill=INK, font=f_head)

    # 左：原圖 + 瑕疵描邊
    x0 = PAD
    d.text((x0, top - 28), "掃描影像與局部瑕疵", fill=INK, font=f_head)
    canvas.paste(Image.fromarray(rgb).resize((pw, ph)), (x0, top))
    if dmap.any():
        edge = (dmap > 0) & ~ndimage.binary_erosion(dmap > 0, iterations=2)
        ov = Image.fromarray(np.where(edge, 255, 0).astype(np.uint8)).resize((pw, ph), Image.Resampling.NEAREST)
        canvas.paste(Image.new("RGB", (pw, ph), INK), (x0, top), ov)
        for n, sl in enumerate(ndimage.find_objects(dmap), start=1):
            if sl is None:
                continue
            tx = x0 + sl[1].start * pw / W
            ty = top + sl[0].start * ph / H
            d.rounded_rectangle([tx, ty - 20, tx + 22, ty - 1], radius=4, fill="#ffffff", outline=INK)
            d.text((tx + 6, ty - 19), str(n), fill=INK, font=f_small)

    # 右：相對壓力圖 + 分區數值 + 傾斜箭頭
    x1 = PAD + pw + 24
    d.text((x1, top - 28), "相對壓力分布（藍 = 偏低、紅 = 偏高）", fill=INK, font=f_head)
    canvas.paste(Image.fromarray(colorize_rel(rel)).resize((pw, ph)), (x1, top))
    sh = result["shape"]
    if sh["tilt_low_side"]:
        _, bx, by = extra["coef"][:3]
        n = np.hypot(bx, by) or 1
        cx, cy, L = x1 + pw / 2, top + ph / 2, min(pw, ph) * 0.3
        ex, ey = cx - bx / n * L, cy - by / n * L          # 指向低壓側
        d.line([(cx, cy), (ex, ey)], fill=INK, width=4)
        ang = np.arctan2(ey - cy, ex - cx)
        for da in (2.6, -2.6):
            d.line([(ex, ey), (ex + 16 * np.cos(ang + da), ey + 16 * np.sin(ang + da))], fill=INK, width=4)

    grid = result["relative_grid"]
    R, C = len(grid), len(grid[0])
    for r in range(R):
        for c in range(C):
            gx0, gy0 = x1 + c * pw / C, top + r * ph / R
            d.rectangle([gx0, gy0, gx0 + pw / C, gy0 + ph / R], outline="#ffffff", width=1)
            v = grid[r][c]
            label = f"{v * 100:.0f}%"
            col = colorize_rel(np.array([[v]]))[0, 0]
            lw = f_body.getlength(label)
            lx, ly0 = gx0 + pw / C / 2 - lw / 2, gy0 + ph / R / 2 - 10
            d.rounded_rectangle([lx - 6, ly0 - 2, lx + lw + 6, ly0 + 22], radius=5, fill=tuple(int(v) for v in col),
                                outline="#ffffff")
            d.text((lx, ly0), label, fill=_ink_on(col), font=f_body)
    # 色階
    ly = top + ph + 22
    bar = colorize_rel(np.linspace(0.5, 1.5, pw)[None, :].repeat(14, 0))
    canvas.paste(Image.fromarray(bar), (x1, ly))
    for v in (0.5, 0.75, 1.0, 1.25, 1.5):
        lab = f"{v * 100:.0f}%"
        tx = x1 + (v - 0.5) * pw
        d.text((tx - f_small.getlength(lab) / 2, ly + 18), lab, fill=INK_2, font=f_small)
    d.text((x0, ly), "箭頭指向壓力偏低的一側；格內數字 = 分區平均 ÷ 整體中位數", fill=INK_2, font=f_small)

    for i, line in enumerate(summary_lines):
        d.text((PAD, top + ph + 80 + i * 26), line, fill=INK, font=f_body)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="感壓紙掃描影像的壓力均勻度分析（相對濃度）")
    ap.add_argument("image", nargs="+", type=Path)
    ap.add_argument("--dpi", type=float, help="掃描解析度；未填則讀影像檔，再不行用 300")
    ap.add_argument("--roi", help="x,y,w,h（原圖像素），只分析此範圍")
    ap.add_argument("--white", help="白紙色 #RRGGBB；未填自動找畫面中的白紙")
    ap.add_argument("--channel", default="auto", choices=["auto", "R", "G", "B"])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "result")
    ap.add_argument("--no-report", action="store_true", help="不輸出報告圖")
    ap.add_argument("--json", action="store_true", help="輸出完整 JSON")
    for f, v in asdict(PressureSettings()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)
    s = PressureSettings(**{f: getattr(a, f) for f in asdict(PressureSettings())})
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None

    all_ok = True
    for p in a.image:
        result, extra = analyze_pressure(Image.open(p), dpi=a.dpi, roi=roi, white=a.white,
                                         channel=a.channel, settings=s)
        if not a.no_report:
            result["report_path"] = str(render_pressure_report(
                result, extra, a.out / f"{safe_stem(p.name)}_pressure.png", title=p.name))
        all_ok &= result["passed"]
        if a.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"{p.name}: {result['summary']}")
            for w in result["warnings"]:
                print(f"  ⚠ {w}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
