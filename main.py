"""色彩均勻度辨識 API。啟動：.venv/bin/uvicorn main:app --reload  文件：http://127.0.0.1:8000/docs"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

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
    max_gradient: float | None = Form(None, gt=0, description="第一階段可容忍的平滑漸層幅度 ΔE00（如打光不均）；"
                                                              "設定後各格改與擬合漸層比較，殘差 ≤ threshold 才通過"),
    spot_mode: Literal["global", "local"] | None = Form(None, description="第二階段比較對象：global = 整張主色；local = 周圍背景"),
    background_size: float | None = Form(None, ge=1, le=50, description="spot_mode=local 時的周圍背景範圍（長邊百分比）"),
    heatmap: bool = Form(True, description="是否在 result 資料夾輸出熱圖 PNG"),
):
    overrides = dict(rows=rows, cols=cols, threshold=threshold, reference=reference or None,
                     margin=margin, spot_threshold=spot_threshold, min_spot_size=min_spot_size,
                     max_gradient=max_gradient, spot_mode=spot_mode, background_size=background_size)
    return run_uniformity(await file.read(), file.filename, site, roi, overrides, heatmap)


def run_uniformity(data: bytes, filename: str | None, site: str | None, roi: str | None,
                   overrides: dict, heatmap: bool = True) -> dict:
    """/v1/uniformity 與 agent 共用的判定流程；輸入有誤時拋 HTTPException。"""
    try:
        cfg, sources = registry.resolve(site or None, overrides)
    except UnknownSiteError:
        raise HTTPException(422, f"未知的 site：{site!r}；可用：{registry.names()}")
    except SiteConfigError as e:
        raise _site_config_error(e)

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
                         min_spot_size=cfg["min_spot_size"], max_gradient=cfg["max_gradient"],
                         spot_mode=cfg["spot_mode"], background_size=cfg["background_size"])
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

    body = {"filename": filename, "site": site or "default",
            "applied_settings": {k: {"value": cfg[k], "source": sources[k]} for k in cfg},
            "analyzed_size": list(img.size), **result.to_dict()}
    out = {"summary": summarize(body), **body}

    if heatmap:
        # bbox 已換算為原圖座標，畫圖時換回分析影像座標
        def to_analyzed(b):
            return [(b[0] - ox) / k, (b[1] - oy) / k, b[2] / k, b[3] / k]
        path = RESULT_DIR / f"{safe_stem(filename)}_heatmap.png"
        try:
            out["heatmap_path"] = str(render_report(img, out, result.pixel_delta_e, result.defect_map, path, to_analyzed))
        except OSError as e:
            out["heatmap_error"] = f"熱圖輸出失敗：{e}"
    return out


# ---------------------------------------------------------------- agent

import anthropic  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import agent  # noqa: E402


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="要問 agent 的話")
    session_id: str | None = Field(None, max_length=64, description="延續對話時帶入上一次回傳的 session_id")


@app.post("/v1/agent")
def agent_chat(req: AgentRequest):
    """用自然語言操作判定功能，例：「用 site_a 檢查所有影像，列出不合格的並說明原因」。

    影像需放在影像資料夾（環境變數 IMAGE_DIR，預設 images/）。同一 session 會保留對話歷史。
    """
    try:
        sid, r = agent.sessions.chat(req.session_id, req.message)
    except agent.SessionBusy:
        raise HTTPException(409, "此 session 上一個請求尚未完成")
    except anthropic.AuthenticationError:
        raise HTTPException(503, "agent 未設定 ANTHROPIC_API_KEY 或金鑰無效")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Claude API 請求過多，請稍後再試")
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        raise HTTPException(502, f"Claude API 錯誤：{e}")
    return {"session_id": sid, "reply": r.reply, "stop_reason": r.stop_reason, "tool_calls": r.tool_calls}


@app.delete("/v1/agent/{session_id}")
def agent_reset(session_id: str):
    """清除對話歷史。"""
    if not agent.sessions.delete(session_id):
        raise HTTPException(404, "找不到此 session")
    return {"deleted": session_id}
