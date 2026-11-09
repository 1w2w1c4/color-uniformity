"""依 site 載入判定標準。設定檔路徑可用環境變數 SITES_CONFIG 指定，預設為同目錄 sites.yaml。

設定檔修改後會在下一次請求自動重新載入（依檔案修改時間判斷）。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml

from uniformity import hex_to_rgb

CONFIG_PATH = Path(os.environ.get("SITES_CONFIG", Path(__file__).resolve().parent / "sites.yaml"))

# 欄位 → (型別, 驗證函式)；None 值一律允許的欄位列在 NULLABLE
_FIELDS = {
    "threshold": (float, lambda v: v > 0),
    "spot_threshold": (float, lambda v: v > 0),
    "min_spot_size": (float, lambda v: 0 <= v <= 100),
    "rows": (int, lambda v: 1 <= v <= 50),
    "cols": (int, lambda v: 1 <= v <= 50),
    "margin": (float, lambda v: 0 <= v < 0.5),
    "reference": (str, lambda v: hex_to_rgb(v) is not None),
    "max_gradient": (float, lambda v: v > 0),
    "spot_mode": (str, lambda v: v in SPOT_MODES),
    "background_size": (float, lambda v: 1 <= v <= 50),
}
NULLABLE = {"spot_threshold", "reference", "max_gradient"}
SETTING_KEYS = tuple(_FIELDS)
SPOT_MODES = ("global", "local")
# 後來新增的欄位：default 區塊沒寫時採用此值（＝原本的判定方式），舊設定檔不需修改
OPTIONAL_DEFAULTS = {"max_gradient": None, "spot_mode": "global", "background_size": 10.0}


class SiteConfigError(Exception):
    """設定檔本身有誤（伺服器端問題）。"""


class UnknownSiteError(KeyError):
    pass


def _validate(name: str, block: dict, require_all: bool) -> dict:
    if not isinstance(block, dict):
        raise SiteConfigError(f"{name} 須為鍵值設定")
    unknown = set(block) - set(_FIELDS)
    if unknown:
        raise SiteConfigError(f"{name} 含未知欄位：{sorted(unknown)}")
    if require_all and (missing := set(_FIELDS) - set(block)):
        raise SiteConfigError(f"{name} 缺少欄位：{sorted(missing)}")
    out = {}
    for key, value in block.items():
        typ, ok = _FIELDS[key]
        if value is None:
            if key not in NULLABLE:
                raise SiteConfigError(f"{name}.{key} 不可為 null")
            out[key] = None
            continue
        try:
            value = typ(value)
            valid = ok(value)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise SiteConfigError(f"{name}.{key} 數值不合法：{block[key]!r}")
        out[key] = value
    return out


def parse_config(raw: object) -> tuple[dict, dict[str, dict]]:
    """回傳 (default 設定, {site: 該 site 明確寫出的欄位})。"""
    if not isinstance(raw, dict) or "default" not in raw:
        raise SiteConfigError("設定檔須包含 default 區塊")
    unknown = set(raw) - {"default", "sites"}
    if unknown:
        raise SiteConfigError(f"設定檔含未知區塊：{sorted(unknown)}")
    block = raw["default"]
    if isinstance(block, dict):
        block = {**OPTIONAL_DEFAULTS, **block}
    default = _validate("default", block, require_all=True)
    sites = raw.get("sites") or {}
    if not isinstance(sites, dict):
        raise SiteConfigError("sites 須為鍵值設定")
    if "default" in sites:
        raise SiteConfigError("site 名稱不可為 default")
    return default, {str(n): _validate(f"sites.{n}", b or {}, require_all=False) for n, b in sites.items()}


class SiteRegistry:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self._mtime: float | None = None
        self._default: dict = {}
        self._sites: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _reload_if_changed(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            raise SiteConfigError(f"找不到設定檔 {self.path}")
        with self._lock:
            if mtime == self._mtime:
                return
            try:
                raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                raise SiteConfigError(f"設定檔 YAML 格式錯誤：{e}")
            self._default, self._sites = parse_config(raw)
            self._mtime = mtime

    def sites(self) -> dict[str, dict]:
        """各 site 合併 default 後的完整標準。"""
        self._reload_if_changed()
        return {"default": dict(self._default),
                **{k: {**self._default, **v} for k, v in self._sites.items()}}

    def resolve(self, site: str | None, overrides: dict) -> tuple[dict, dict]:
        """回傳 (最終設定, 每個欄位的來源)。優先序：request > site:<名稱> > default。"""
        self._reload_if_changed()
        if site is not None and site != "default" and site not in self._sites:
            raise UnknownSiteError(site)
        own = self._sites.get(site, {}) if site else {}
        settings, sources = {}, {}
        for key in SETTING_KEYS:
            if overrides.get(key) is not None:
                settings[key], sources[key] = overrides[key], "request"
            elif key in own:
                settings[key], sources[key] = own[key], f"site:{site}"
            else:
                settings[key], sources[key] = self._default[key], "default"
        return settings, sources

    def names(self) -> list[str]:
        self._reload_if_changed()
        return sorted(self._sites)
