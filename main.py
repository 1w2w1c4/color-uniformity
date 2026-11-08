"""色彩均勻度辨識 API。啟動：.venv/bin/uvicorn main:app --reload  文件：http://127.0.0.1:8000/docs"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from heatmap import render_report, safe_stem
from sites import SiteConfigError, SiteRegistry, UnknownSiteError
from summary import summarize
from uniformity import analyze

MAX_BYTES = 20 * 1024 * 1024
MAX_SIDE = 2000  # 超過時等比縮小，加快運算；均勻度分析不需要全解析度
RESULT_DIR = Path(os.environ.get("RESULT_DIR", Path(__file__).resolve().parent / "result"))

app = FastAPI(title="Color Uniformity API", version="1.1.0",
              description="上傳影像，切格比較各區域與參考色的 CIEDE2000 色差，判定色彩是否均勻。"
                          "帶 site 參數即套用該 site 的判定標準（見 GET /v1/sites）。")
registry = SiteRegistry()


def _site_config_error(e: SiteConfigError) -> HTTPException:
    return HTTPException(500, f"site 設定檔錯誤：{e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/sites")
def list_sites():
    """列出各 site 的判定標準（已合併 default）。"""
    try:
        return registry.sites()
    except SiteConfigError as e:
        raise _site_config_error(e)


@app.post("/v1/uniformity")
async def uniformity(
    file: UploadFile = File(..., description="待測影像（JPG/PNG/BMP/TIFF）"),
    site: str | None = Form(None, description="site 名稱，套用該 site 的判定標準；未填用 default"),
    roi: str | None = Form(None, description="感興趣區域 x,y,w,h（像素），未填則用整張圖"),
    # 以下參數未填時依 site 標準；有填則覆蓋 site 標準
    rows: int | None = Form(None, description="縱向格數"),
    cols: int | None = Form(None, description="橫向格數"),
    threshold: float | None = Form(None, gt=0, description="整體最大允許 ΔE00；常見：≤1 幾乎無感、≤3 一般可接受"),
    reference: str | None = Form(None, description="參考色 #RRGGBB"),
    margin: float | None = Form(None, ge=0, lt=0.5, description="每格內縮比例，排除邊界"),
    spot_threshold: float | None = Form(None, gt=0, description="小點偵測的 ΔE00 門檻"),
    min_spot_size: float | None = Form(None, ge=0, description="小點最小尺寸，以照片長邊的百分比表示（例：0.5 = 長邊 0.5%），更小的視為灰塵忽略；與照片解析度無關"),
    heatmap: bool = Form(True, description="是否在 result 資料夾輸出熱圖 PNG"),
):
    overrides = dict(rows=rows, cols=cols, threshold=threshold, reference=reference or None,
                     margin=margin, spot_threshold=spot_threshold, min_spot_size=min_spot_size)
    try:
        cfg, sources = registry.resolve(site or None, overrides)
    except UnknownSiteError:
        raise HTTPException(422, f"未知的 site：{site!r}；可用：{registry.names()}")
    except SiteConfigError as e:
        raise _site_config_error(e)

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "檔案超過 20MB")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "無法辨識的影像檔")

    ox = oy = 0
    if roi:
        try:
            x, y, w, h = (int(v) for v in roi.split(","))
        except ValueError:
            raise HTTPException(422, "roi 格式須為 x,y,w,h")
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img.width or y + h > img.height:
            raise HTTPException(422, f"roi 超出影像範圍（{img.width}×{img.height}）")
        img = img.crop((x, y, x + w, y + h))
        ox, oy = x, y

    k = 1.0  # 分析影像 → 原圖的縮放倍率
    if max(img.size) > MAX_SIDE:
        orig_w = img.width
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        k = orig_w / img.width

    try:
        result = analyze(np.asarray(img), rows=cfg["rows"], cols=cfg["cols"],
                         threshold=cfg["threshold"], reference_hex=cfg["reference"],
                         margin=cfg["margin"], spot_threshold=cfg["spot_threshold"],
                         min_spot_size=cfg["min_spot_size"])
    except ValueError as e:
        raise HTTPException(422, str(e))

    # 小點座標與面積換算回上傳原圖
    spots = result.spots
    spots["min_area_px"] = round(spots["min_area_px"] * k ** 2, 1)  # 換算成原圖像素²
    for s in spots["items"]:
        s["center"] = [round(s["center"][0] * k + ox), round(s["center"][1] * k + oy)]
        bx, by, bw, bh = s["bbox"]
        s["bbox"] = [round(bx * k + ox), round(by * k + oy), round(bw * k), round(bh * k)]
        s["area_px"] = round(s["area_px"] * k ** 2)

    body = {"filename": file.filename, "site": site or "default",
            "applied_settings": {k: {"value": cfg[k], "source": sources[k]} for k in cfg},
            "analyzed_size": list(img.size), **result.to_dict()}
    out = {"summary": summarize(body), **body}

    if heatmap:
        # bbox 已換算為原圖座標，畫圖時換回分析影像座標
        def to_analyzed(b):
            return [(b[0] - ox) / k, (b[1] - oy) / k, b[2] / k, b[3] / k]
        path = RESULT_DIR / f"{safe_stem(file.filename)}_heatmap.png"
        try:
            out["heatmap_path"] = str(render_report(img, out, result.pixel_delta_e, result.defect_map, path, to_analyzed))
        except OSError as e:
            out["heatmap_error"] = f"熱圖輸出失敗：{e}"
    return out
