"""色彩均勻度檢測 agent：Claude 透過工具呼叫本專案的判定功能，再以中文回答。

合格與否一律由工具（uniformity.py / pressure.py）判定，Claude 只負責選工具、彙整與解讀。
需要環境變數 ANTHROPIC_API_KEY（或 `ant auth login`）。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import yaml
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("AGENT_CONFIG", ROOT / "config.yaml"))
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
SUPPORTED_PROVIDERS = {"anthropic"}  # 新增廠商：實作對應的 run_agent 轉接器後加入


class AgentConfigError(Exception):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    """讀取 config.yaml 並套用環境變數覆蓋。檔案不存在時全部用預設值。"""
    raw = {}
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise AgentConfigError(f"{path} 格式錯誤：{e}")
    sec = lambda d, name: (d.get(name) or {}) if isinstance(d, dict) else {}  # noqa: E731
    ag, paths, ses = sec(raw, "agent"), sec(raw, "paths"), sec(raw, "session")
    env = os.environ.get
    provider = env("AGENT_PROVIDER") or ag.get("provider", "anthropic")
    if provider not in SUPPORTED_PROVIDERS:
        raise AgentConfigError(f"provider {provider!r} 尚未實作轉接器；目前支援：{sorted(SUPPORTED_PROVIDERS)}")
    pv = sec(sec(raw, "providers"), provider)
    key_env = pv.get("api_key_env") or "ANTHROPIC_API_KEY"
    cfg = {
        "provider": provider,
        "api_key": pv.get("api_key") or env(key_env) or None,  # None → SDK 依預設方式找憑證
        "api_key_env": key_env,
        "base_url": pv.get("base_url") or None,
        "timeout": float(pv.get("timeout", 300)),
        "max_retries": int(pv.get("max_retries", 2)),
        "effort": env("AGENT_EFFORT") or pv.get("effort", "medium"),
        "fallbacks": bool(pv.get("fallbacks", True)),
        "model": env("AGENT_MODEL") or ag.get("model", "claude-opus-5-5"),
        "max_tokens": int(ag.get("max_tokens", 16000)),
        "max_steps": int(ag.get("max_steps", 25)),
        "max_items": int(ag.get("max_items", 10)),
        "image_dir": (ROOT / (env("IMAGE_DIR") or paths.get("image_dir", "images"))).resolve(),
        "session_ttl": float(ses.get("ttl", 3600)),
        "max_sessions": int(ses.get("max_sessions", 200)),
    }
    if cfg["effort"] not in EFFORTS:
        raise AgentConfigError(f"effort 須為 {sorted(EFFORTS)} 之一，目前為 {cfg['effort']!r}")
    for k in ("timeout", "max_tokens", "max_steps", "max_items", "session_ttl", "max_sessions"):
        if cfg[k] <= 0:
            raise AgentConfigError(f"{k} 須大於 0")
    if cfg["max_retries"] < 0:
        raise AgentConfigError("max_retries 不可小於 0")
    return cfg


CONFIG = load_config()
MODEL = CONFIG["model"]
EFFORT = CONFIG["effort"]
IMAGE_DIR = CONFIG["image_dir"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_STEPS = CONFIG["max_steps"]
MAX_ITEMS = CONFIG["max_items"]
SESSION_TTL = CONFIG["session_ttl"]
MAX_SESSIONS = CONFIG["max_sessions"]

SYSTEM = f"""你是面板色彩均勻度檢測助理，使用者是產線品管人員。
- 合格與否一律以工具回傳的 passed 為準，不可自行推翻或猜測；沒有檢查過的影像不可下結論。
- 判定結果一律寫成 Pass／Fail（與報告圖、summary 一致），不要翻成「合格／不合格」。
- 主要分析方式是 check_uniformity（uniformity.py 色差判定），所有影像預設都用它。
- 只有使用者明確提到「感壓紙」或「壓力分析」時才用 check_pressure；不可自行判斷影像是否為感壓紙而改用它。
- 需要檢查多張影像時，在同一回合同時呼叫多個工具。
- 影像路徑一律相對於影像資料夾（{IMAGE_DIR.name}/），先用 list_images 確認檔名。
- 回答用繁體中文，精簡條列。說明不合格原因時引用瑕疵位置、tone、ΔE（或壓力指標）與熱圖／報告路徑。
- 若使用者要求調整門檻，說明這只是試算，正式標準需修改 sites.yaml。"""

TOOLS = [
    {
        "name": "list_sites",
        "description": "列出所有 site 的判定標準（threshold、spot_threshold、min_spot_size、rows、cols 等）。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_images",
        "description": "列出影像資料夾（或其子資料夾）中的影像檔。",
        "input_schema": {
            "type": "object",
            "properties": {"subdir": {"type": "string", "description": "子資料夾，空字串表示根目錄"}},
        },
    },
    {
        "name": "check_uniformity",
        "description": "【主要分析工具，預設使用】以 uniformity.py 的 CIEDE2000 色差判定影像顏色是否均勻"
                       "（第一階段整體切格、第二階段局部小點），"
                       "並輸出熱圖。回傳 summary、passed、score、worst_cell、spots、heatmap_path。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "影像路徑，相對於影像資料夾"},
                "site": {"type": "string", "description": "site 名稱；省略則用 default"},
                "roi": {"type": "string", "description": "感興趣區域 x,y,w,h（像素）"},
                "threshold": {"type": "number", "description": "覆蓋整體 ΔE00 門檻（試算用）"},
                "spot_threshold": {"type": "number", "description": "覆蓋小點 ΔE00 門檻（試算用）"},
                "min_spot_size": {"type": "number", "description": "覆蓋最小瑕疵尺寸，長邊百分比（試算用）"},
                "max_gradient": {"type": "number", "description": "第一階段可容忍的平滑漸層幅度 ΔE（如打光不均）；"
                                 "設定後各格改與擬合漸層比較（試算用）"},
                "spot_mode": {"type": "string", "enum": ["global", "local"],
                              "description": "第二階段比較對象：global 整張主色、local 周圍背景（試算用）"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "check_pressure",
        "description": "【僅在使用者明確要求感壓紙／壓力分析時使用】感壓紙掃描影像的壓力均勻度分析："
                       "CV、p5/p95、接觸面積、傾斜、中凸凹、局部瑕疵，並輸出報告圖。一般色彩均勻度請用 check_uniformity。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "掃描影像路徑，相對於影像資料夾"},
                "dpi": {"type": "number", "description": "掃描 DPI；省略則讀影像檔"},
                "roi": {"type": "string", "description": "x,y,w,h（原圖像素）"},
            },
            "required": ["path"],
        },
    },
]


class ToolError(Exception):
    """回給 Claude 的工具錯誤（is_error=True），讓它自行修正參數。"""


def _resolve(rel: str, want_dir: bool = False) -> Path:
    """路徑限制在 IMAGE_DIR 內，防止讀取任意檔案。"""
    p = (IMAGE_DIR / (rel or ".")).resolve()
    if not p.is_relative_to(IMAGE_DIR):
        raise ToolError(f"路徑必須在影像資料夾內：{rel}")
    if want_dir:
        if not p.is_dir():
            raise ToolError(f"找不到資料夾：{rel}")
    elif not p.is_file():
        raise ToolError(f"找不到影像：{rel}（請先用 list_images 確認）")
    return p


def _roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise ToolError("roi 格式須為 x,y,w,h")
    return x, y, w, h


def list_sites() -> dict:
    import main
    return main.list_sites()


def list_images(subdir: str = "") -> list[str]:
    d = _resolve(subdir, want_dir=True)
    return [str(p.relative_to(IMAGE_DIR)) for p in sorted(d.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS][:500]


def check_uniformity(path: str, site: str | None = None, roi: str | None = None,
                     threshold: float | None = None, spot_threshold: float | None = None,
                     min_spot_size: float | None = None, max_gradient: float | None = None,
                     spot_mode: str | None = None) -> dict:
    import main
    p = _resolve(path)
    if spot_mode not in (None, "global", "local"):
        raise ToolError("spot_mode 須為 global 或 local")
    overrides = dict(rows=None, cols=None, threshold=threshold, reference=None, margin=None,
                     spot_threshold=spot_threshold, min_spot_size=min_spot_size,
                     max_gradient=max_gradient, spot_mode=spot_mode)
    r = main.run_uniformity(p.read_bytes(), p.name, site, roi, overrides, heatmap=True)
    out = {k: r[k] for k in ("summary", "passed", "score", "grid_passed", "worst_cell",
                             "cells_over_threshold", "delta_e_max", "grid_mode", "gradient", "reference_hex",
                             "heatmap_path", "heatmap_error") if k in r}
    out["applied_settings"] = {k: v["value"] for k, v in r["applied_settings"].items()}
    sp = r["spots"]
    out["spots"] = {"passed": sp["passed"], "count": sp["count"],
                    "items": sp["items"][:MAX_ITEMS], "defect_area_ratio": sp["defect_area_ratio"]}
    return out


def check_pressure(path: str, dpi: float | None = None, roi: str | None = None) -> dict:
    import main
    import pressure
    from heatmap import safe_stem
    p = _resolve(path)
    try:
        img = Image.open(p)
        result, extra = pressure.analyze_pressure(img, dpi=dpi, roi=_roi(roi))
    except UnidentifiedImageError:
        raise ToolError(f"無法辨識的影像檔：{path}")
    except ValueError as e:
        raise ToolError(str(e))
    try:
        result["report_path"] = str(pressure.render_pressure_report(
            result, extra, main.RESULT_DIR / f"{safe_stem(p.name)}_pressure.png", title=p.name))
    except OSError as e:
        result["report_error"] = f"報告圖輸出失敗：{e}"
    result["defects"] = result["defects"][:MAX_ITEMS]
    return result


HANDLERS = {"list_sites": list_sites, "list_images": list_images,
            "check_uniformity": check_uniformity, "check_pressure": check_pressure}


def execute_tool(name: str, args: dict) -> tuple[str, bool]:
    """回傳 (結果文字, is_error)。錯誤也回給 Claude，而不是讓整個請求失敗。"""
    fn = HANDLERS.get(name)
    if fn is None:
        return f"未知的工具：{name}", True
    try:
        return json.dumps(fn(**args), ensure_ascii=False, default=str), False
    except ToolError as e:
        return str(e), True
    except HTTPException as e:
        return f"{e.status_code}: {e.detail}", True
    except TypeError as e:  # 參數名稱或型別錯誤
        return f"參數錯誤：{e}", True


# ---------------------------------------------------------------- agent 循環

@dataclass
class AgentResult:
    reply: str
    stop_reason: str
    tool_calls: list[dict]
    new_messages: list[dict]


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"],
                                      timeout=CONFIG["timeout"], max_retries=CONFIG["max_retries"])
    return _client


def run_agent(history: list[dict], user_message: str) -> AgentResult:
    """執行一輪對話。不修改 history；成功後由呼叫端把 new_messages 接上。"""
    client = _get_client()
    messages = [*history, {"role": "user", "content": user_message}]
    start = len(history)
    tool_calls: list[dict] = []

    for _ in range(MAX_STEPS):
        fallback = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                    if CONFIG["fallbacks"] else {})
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=CONFIG["max_tokens"],
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            **fallback,
        )
        # 完整保留 content（含 thinking 區塊），下一輪原樣送回
        messages.append({"role": "assistant", "content": [b.to_dict() for b in response.content]})

        if response.stop_reason == "pause_turn":
            continue
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "refusal":
                text = text or "此請求無法處理，請換個方式描述。"
            elif response.stop_reason == "max_tokens":
                text += "\n\n（回覆過長被截斷）"
            return AgentResult(text, response.stop_reason, tool_calls, messages[start:])

        results = []
        for b in response.content:
            if b.type != "tool_use":
                continue
            content, is_error = execute_tool(b.name, b.input)
            tool_calls.append({"name": b.name, "input": b.input, "is_error": is_error})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})  # 所有結果放同一則訊息

    return AgentResult("工具呼叫次數超過上限，請把問題拆小一點再試。", "max_steps",
                       tool_calls, messages[start:])


# ---------------------------------------------------------------- session（記憶體內）

@dataclass
class Session:
    messages: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    updated: float = field(default_factory=time.monotonic)


class SessionBusy(Exception):
    pass


class SessionStore:
    """單一程序內的對話歷史。多 worker 部署需改成 Redis 等共用儲存。"""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _evict(self):
        now = time.monotonic()
        for sid in [s for s, v in self._sessions.items() if now - v.updated > SESSION_TTL]:
            del self._sessions[sid]
        while len(self._sessions) >= MAX_SESSIONS:
            del self._sessions[min(self._sessions, key=lambda s: self._sessions[s].updated)]

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        with self._lock:
            if session_id and session_id in self._sessions:
                return session_id, self._sessions[session_id]
            self._evict()
            sid = session_id or uuid.uuid4().hex
            self._sessions[sid] = Session()
            return sid, self._sessions[sid]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def chat(self, session_id: str | None, message: str) -> tuple[str, AgentResult]:
        sid, s = self.get_or_create(session_id)
        if not s.lock.acquire(blocking=False):
            raise SessionBusy(sid)
        try:
            result = run_agent(s.messages, message)
            s.messages.extend(result.new_messages)  # 只有成功才寫入，失敗不會留下殘缺歷史
            s.updated = time.monotonic()
            return sid, result
        finally:
            s.lock.release()


sessions = SessionStore()
