import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from main import app
from sites import SiteConfigError, SiteRegistry, parse_config
from uniformity import analyze, delta_e_2000, srgb_to_lab

client = TestClient(app)

TEST_SITES = """
default: {threshold: 3.0, spot_threshold: null, min_spot_size: 0.5, rows: 5, cols: 5, margin: 0.1, reference: null}
sites:
  strict: {threshold: 1.0, rows: 8}
  loose:  {threshold: 10.0, min_spot_size: 2.0}
"""


@pytest.fixture(autouse=True)
def sites_config(tmp_path, monkeypatch):
    """測試一律使用固定的 site 設定，不受實際 sites.yaml 修改影響。"""
    cfg = tmp_path / "sites.yaml"
    cfg.write_text(TEST_SITES, encoding="utf-8")
    monkeypatch.setattr(main, "registry", SiteRegistry(cfg))
    monkeypatch.setattr(main, "RESULT_DIR", tmp_path / "result")  # 測試不寫入專案的 result/
    return cfg


def png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


# Sharma et al. (2005) CIEDE2000 官方測試資料節錄
@pytest.mark.parametrize("lab1,lab2,expected", [
    ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
    ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
    ((50, 0, 0), (50, -1, 2), 2.3669),
    ((50, 2.5, 0), (73, 25, -18), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
])
def test_ciede2000(lab1, lab2, expected):
    assert delta_e_2000(np.array(lab1), np.array(lab2)) == pytest.approx(expected, abs=1e-4)


def test_lab_roundtrip():
    from uniformity import lab_to_srgb
    rgb = np.array([[200, 22, 34], [0, 0, 0], [255, 255, 255], [120, 160, 200]], float)
    assert lab_to_srgb(srgb_to_lab(rgb)) == pytest.approx(rgb, abs=0.01)


def test_srgb_white_and_black():
    assert srgb_to_lab(np.array([255, 255, 255])) == pytest.approx([100, 0, 0], abs=0.01)
    assert srgb_to_lab(np.array([0, 0, 0])) == pytest.approx([0, 0, 0], abs=0.01)


def test_uniform_image_passes():
    img = np.full((200, 200, 3), (120, 160, 200))
    r = client.post("/v1/uniformity", files={"file": ("a.png", png(img))}).json()
    assert r["passed"] and r["delta_e_max"] == 0 and r["score"] == 100


def test_defect_located():
    img = np.full((200, 200, 3), 128)
    img[80:120, 160:200] = (200, 80, 80)  # 第 2 列、第 4 欄有色斑
    r = client.post("/v1/uniformity", files={"file": ("a.png", png(img))}).json()
    assert not r["passed"]
    assert (r["worst_cell"]["row"], r["worst_cell"]["col"]) == (2, 4)


def test_gradient_and_reference():
    x = np.linspace(110, 150, 300)
    img = np.repeat(np.stack([x, x, x], -1)[None], 100, axis=0)
    r = client.post("/v1/uniformity", files={"file": ("g.png", png(img))},
                    data={"rows": 1, "cols": 6, "reference": "#808080"}).json()
    assert r["delta_e_grid"][0][0] > 0 and r["luminance_uniformity"] < 1
    assert r["reference_source"] == "指定色 #808080"


def test_roi_crop():
    img = np.full((100, 200, 3), 50)
    img[:, 100:] = 220
    r = client.post("/v1/uniformity", files={"file": ("a.png", png(img))},
                    data={"roi": "100,0,100,100"}).json()
    assert r["passed"]


# ---- 單一小白點：整體（格子）判定會通過，需靠第二階段逐像素偵測抓出 ----
RED, WHITE = (200, 22, 34), (246, 240, 238)


def red_with_spot(cx, cy, diameter, size=(1200, 900), seed=0):
    W, H = size
    rng = np.random.default_rng(seed)
    a = np.full((H, W, 3), RED, float) + rng.normal(0, 4, (H, W, 3))  # 模擬相機雜訊
    yy, xx = np.mgrid[0:H, 0:W]
    a[(xx - cx) ** 2 + (yy - cy) ** 2 <= (diameter / 2) ** 2] = WHITE
    return np.clip(a, 0, 255)


# 5×5 格時每格 240×180：(600,450) 為格子中心、(720,540) 在格線交點、(482,370) 在格子內縮邊緣
@pytest.mark.parametrize("cx,cy", [(600, 450), (720, 540), (482, 370)],
                         ids=["cell-center", "grid-line", "cell-margin"])
@pytest.mark.parametrize("diameter", [10, 40, 120])
def test_single_small_spot_detected(cx, cy, diameter):
    r = client.post("/v1/uniformity", files={"file": ("s.png", png(red_with_spot(cx, cy, diameter)))}).json()
    assert r["grid_passed"], "整體偵測不受小點影響，維持原判定"
    assert not r["passed"] and not r["spots"]["passed"]
    assert r["spots"]["count"] == 1 and r["spots"]["spot_count"] == 1
    s = r["spots"]["items"][0]
    assert s["type"] == "spot"
    assert abs(s["center"][0] - cx) <= 4 and abs(s["center"][1] - cy) <= 4
    assert s["area_px"] == pytest.approx(np.pi * (diameter / 2) ** 2, rel=0.6)


def test_noise_and_dust_ignored():
    """只有相機雜訊與小於 min_spot_size 的灰塵（直徑 4px = 長邊 0.33%）→ 兩階段皆通過。"""
    r = client.post("/v1/uniformity", files={"file": ("d.png", png(red_with_spot(300, 300, 4)))}).json()
    assert r["grid_passed"] and r["spots"]["passed"] and r["passed"]


def test_spot_coordinates_with_roi_and_downscale():
    """大圖（會被縮到 2000）＋ ROI 時，小點座標仍回報為原圖座標。"""
    img = red_with_spot(2500, 1500, 30, size=(3000, 2000))
    r = client.post("/v1/uniformity", files={"file": ("big.png", png(img))},
                    data={"roi": "1000,500,2000,1500"}).json()
    s = r["spots"]["items"][0]
    assert abs(s["center"][0] - 2500) <= 6 and abs(s["center"][1] - 1500) <= 6


def test_large_area_reported_as_region():
    img = np.full((300, 400, 3), RED)
    img[:, 300:] = WHITE  # 右側 1/4 泛白
    r = client.post("/v1/uniformity", files={"file": ("e.png", png(img))}).json()
    assert not r["grid_passed"] and r["spots"]["region_count"] == 1


@pytest.mark.parametrize("data,code", [
    ({"spot_threshold": 0}, 422),
    ({"min_spot_size": -1}, 422),
    ({"roi": "0,0,999,999"}, 422),
    ({"roi": "abc"}, 422),
    ({"reference": "#12"}, 422),
    ({"rows": 1, "cols": 1}, 422),
    ({"threshold": 0}, 422),
])
def test_bad_params(data, code):
    img = np.full((100, 100, 3), 100)
    assert client.post("/v1/uniformity", files={"file": ("a.png", png(img))}, data=data).status_code == code


def test_not_an_image():
    assert client.post("/v1/uniformity", files={"file": ("a.txt", b"hello")}).status_code == 400


# ---- site 標準 ----
def two_tone():
    """左側略暗的灰色圖：整體最大 ΔE00 ≈ 3.18。"""
    a = np.full((300, 400, 3), (180, 180, 175))
    a[:, :80] = (170, 170, 168)
    return png(a)


def post_site(data):
    return client.post("/v1/uniformity", files={"file": ("t.png", two_tone())}, data=data)


@pytest.mark.parametrize("site,passed", [(None, False), ("default", False), ("strict", False), ("loose", True)])
def test_site_changes_verdict(site, passed):
    r = post_site({"site": site} if site else {}).json()
    assert r["site"] == (site or "default")
    assert r["passed"] is passed


def test_site_settings_and_sources():
    r = post_site({"site": "strict"}).json()
    a = r["applied_settings"]
    assert a["threshold"] == {"value": 1.0, "source": "site:strict"}
    assert a["rows"] == {"value": 8, "source": "site:strict"}
    assert a["cols"] == {"value": 5, "source": "default"}
    assert r["rows"] == 8 and r["threshold"] == 1.0


def test_request_overrides_site():
    r = post_site({"site": "loose", "threshold": 2, "reference": "#B4B4AF"}).json()
    assert r["applied_settings"]["threshold"] == {"value": 2.0, "source": "request"}
    assert r["applied_settings"]["min_spot_size"] == {"value": 2.0, "source": "site:loose"}
    assert r["reference_source"] == "指定色 #B4B4AF" and not r["passed"]


def test_unknown_site():
    r = post_site({"site": "nowhere"})
    assert r.status_code == 422 and "loose" in r.json()["detail"] and "strict" in r.json()["detail"]


def test_list_sites():
    r = client.get("/v1/sites").json()
    assert set(r) == {"default", "strict", "loose"}
    assert r["strict"]["threshold"] == 1.0 and r["strict"]["cols"] == 5  # 已合併 default


def test_config_hot_reload(sites_config):
    assert post_site({"site": "loose"}).json()["passed"]
    sites_config.write_text(TEST_SITES.replace("threshold: 10.0", "threshold: 2.0"), encoding="utf-8")
    import os; st = sites_config.stat(); os.utime(sites_config, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert not post_site({"site": "loose"}).json()["passed"]


def test_broken_config_returns_500(sites_config):
    sites_config.write_text(TEST_SITES.replace("threshold: 1.0", "threshold: -1"), encoding="utf-8")
    import os; st = sites_config.stat(); os.utime(sites_config, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    r = post_site({})
    assert r.status_code == 500 and "sites.strict.threshold" in r.json()["detail"]


@pytest.mark.parametrize("raw,msg", [
    ({}, "default"),
    ({"default": {"threshold": 3}}, "缺少欄位"),
    ({"default": {**dict(threshold=3, spot_threshold=None, min_spot_size=0.5, rows=5, cols=5,
                         margin=0.1, reference=None)}, "sites": {"x": {"thresh": 1}}}, "未知欄位"),
    ({"default": {**dict(threshold=3, spot_threshold=None, min_spot_size=0.5, rows=5, cols=5,
                         margin=0.1, reference=None)}, "sites": {"x": {"reference": "red"}}}, "reference"),
    ({"default": {**dict(threshold=3, spot_threshold=None, min_spot_size=0.5, rows=5, cols=5,
                         margin=0.1, reference=None)}, "sites": {"x": {"rows": None}}}, "不可為 null"),
])
def test_parse_config_rejects(raw, msg):
    with pytest.raises(SiteConfigError, match=msg):
        parse_config(raw)


def test_real_sites_yaml_is_valid():
    """專案內實際的 sites.yaml 必須能正確載入。"""
    SiteRegistry().sites()


# ---- 無標準色：每張圖以自身主色為基準 ----
def mixed(red_frac, pink_frac, size=(400, 300), seed=1):
    """紅、粉紅（過渡）、白三色直條混合；用來模擬大面積泛白。"""
    W, H = size
    a = np.empty((H, W, 3))
    r, p = int(W * red_frac), int(W * (red_frac + pink_frac))
    a[:, :r], a[:, r:p], a[:, p:] = RED, (225, 130, 135), WHITE
    a += np.random.default_rng(seed).normal(0, 3, a.shape)
    return np.clip(a, 0, 255)


def test_reference_is_dominant_color_not_median():
    """紅 45%、粉紅 25%、白 30%：中位數會落在粉紅附近，主色估計須找回紅色。"""
    r = client.post("/v1/uniformity", files={"file": ("m.png", png(mixed(0.45, 0.25)))}).json()
    assert r["reference_source"] == "影像主色（自動）"
    assert delta_e_2000(np.array(r["reference_lab"]), srgb_to_lab(np.array(RED, float))) < 1.5
    assert 0.40 <= r["reference_coverage"] <= 0.50
    assert r["spots"]["defect_area_ratio"] == pytest.approx(0.55, abs=0.05)  # 粉紅 + 白都被標出
    assert not r["passed"]


@pytest.mark.parametrize("color", [(200, 22, 34), (30, 90, 160), (128, 128, 128), (240, 200, 60)])
def test_each_image_uses_its_own_color(color):
    """不同顏色的均勻產品，不需給標準色都應通過，且主色即為該顏色。"""
    a = np.full((300, 400, 3), color, float) + np.random.default_rng(2).normal(0, 3, (300, 400, 3))
    r = client.post("/v1/uniformity", files={"file": ("c.png", png(np.clip(a, 0, 255)))}).json()
    assert r["passed"] and r["reference_coverage"] > 0.95  # 雜訊會讓少數孤立像素超標
    assert delta_e_2000(np.array(r["reference_lab"]), srgb_to_lab(np.array(color, float))) < 1


def test_reference_hex_reported():
    r = client.post("/v1/uniformity", files={"file": ("s.png", png(red_with_spot(600, 450, 30)))}).json()
    assert r["reference_hex"] == "#C81622"


# ---- 綜合說明 summary ----
def summarize_of(img, **data):
    return client.post("/v1/uniformity", files={"file": ("x.png", png(img))}, data=data).json()


def test_summary_pass():
    a = np.clip(np.full((300, 400, 3), RED, float) + np.random.default_rng(0).normal(0, 3, (300, 400, 3)), 0, 255)
    s = summarize_of(a)["summary"]
    assert s.startswith("【Pass】整體與局部皆均勻。")
    assert "第一階段整體均勻度通過" in s and "第二階段局部瑕疵通過" in s


def test_summary_grid_ok_but_spot():
    r = summarize_of(red_with_spot(600, 450, 30))
    s = r["summary"]
    assert s.startswith("【Fail】整體顏色均勻，但有局部瑕疵。")
    assert "第一階段整體均勻度通過" in s and "第二階段局部瑕疵不通過" in s
    assert "中央的泛白小點" in s and "(600, 450)" in s.replace("(601", "(600").replace(", 451)", ", 450)")


def test_summary_both_fail_mentions_position_and_tone():
    img = np.full((300, 400, 3), RED, float)
    img[:, 300:] = WHITE  # 右側泛白
    s = summarize_of(img)["summary"]
    assert s.startswith("【Fail】整體與局部皆不均勻。")
    assert "第一階段整體均勻度不通過" in s and "右" in s and "泛白" in s
    assert "1 處大面積" in s


def test_summary_dark_defect():
    img = np.full((300, 400, 3), RED, float)
    img[100:200, 150:250] = (110, 10, 18)  # 中央偏暗（亮度與彩度皆下降）
    r = summarize_of(img)
    assert r["spots"]["items"][0]["tone"] == "偏暗" and "偏暗" in r["summary"]


def test_summary_low_coverage_warning():
    s = summarize_of(mixed(0.45, 0.25))["summary"]
    assert "注意：與基準色相符的面積僅" in s


@pytest.mark.parametrize("fx,fy,name", [(0.1, 0.1, "左上方"), (0.5, 0.5, "中央"), (0.9, 0.5, "右側"),
                                         (0.5, 0.9, "下方"), (0.1, 0.9, "左下方")])
def test_position_name(fx, fy, name):
    from uniformity import position_name
    assert position_name(fx, fy) == name


@pytest.mark.parametrize("main_color,other", [((200, 22, 34), (110, 10, 18)), ((200, 22, 34), WHITE),
                                              ((128, 128, 128), (90, 90, 90)), ((30, 90, 160), (60, 120, 190))])
def test_reference_exact_on_noise_free_image(main_color, other):
    """無雜訊影像：主色須精確等於佔多數的顏色（避免平滑直方圖峰值落在鄰格）。"""
    img = np.full((300, 400, 3), main_color, np.uint8)
    img[100:200, 150:250] = other
    r = summarize_of(img)
    assert delta_e_2000(np.array(r["reference_lab"]), srgb_to_lab(np.array(main_color, float))) < 0.1
    assert r["grid_passed"] is False or r["spots"]["count"] == 1
    assert r["spots"]["defect_area_ratio"] == pytest.approx(100 * 100 / (300 * 400), abs=0.01)


@pytest.mark.parametrize("color,tone", [((246, 240, 238), "泛白"), ((225, 130, 135), "泛白"),
                                        ((110, 10, 18), "偏暗"), ((150, 70, 75), "變灰"),
                                        ((200, 90, 20), "偏色"), ((160, 20, 120), "偏色")])
def test_classify_tone(color, tone):
    from uniformity import classify_tone
    assert classify_tone(srgb_to_lab(np.array(color, float)), srgb_to_lab(np.array(RED, float))) == tone


# ---- 熱圖輸出 ----
def test_heatmap_written(tmp_path):
    r = summarize_of(red_with_spot(600, 450, 30))
    p = Path(r["heatmap_path"])
    assert p.parent == tmp_path / "result" and p.name == "x_heatmap.png"
    im = Image.open(p)
    assert im.format == "PNG" and im.width > 1000


def test_heatmap_can_be_disabled(tmp_path):
    r = summarize_of(red_with_spot(600, 450, 30), heatmap="false")
    assert "heatmap_path" not in r and not (tmp_path / "result").exists()


def test_heatmap_filename_is_sanitized(tmp_path):
    img = np.full((100, 100, 3), 100)
    r = client.post("/v1/uniformity", files={"file": ("../../etc/evil name.png", png(img))}).json()
    p = Path(r["heatmap_path"])
    assert p.parent == tmp_path / "result" and p.name == "evil_name_heatmap.png"


def test_colorize_monotonic():
    from heatmap import colorize
    c = colorize(np.linspace(0, 12, 50)).astype(float)
    lum = c @ [0.2126, 0.7152, 0.0722]
    assert np.all(np.diff(lum) <= 0.5)  # 色差越大顏色越深


# ---- 照片尺寸不固定：同一場景不同解析度，判定須一致 ----
def scene(W, H, seed=3):
    """同一場景依比例縮放：白點直徑 = 長邊 1%（瑕疵）、灰塵直徑 = 長邊 0.3%（應忽略）。"""
    rng = np.random.default_rng(seed)
    a = np.full((H, W, 3), RED, float) + rng.normal(0, 4, (H, W, 3))
    yy, xx = np.mgrid[0:H, 0:W]
    for fx, fy, pct in [(0.3, 0.4, 1.0), (0.7, 0.6, 0.3)]:
        a[(xx - fx * W) ** 2 + (yy - fy * H) ** 2 <= (pct / 100 * W / 2) ** 2] = WHITE
    return np.clip(a, 0, 255)


def jpg(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.mark.parametrize("W,H", [(800, 600), (1200, 900), (2400, 1800), (4000, 3000)])
def test_same_scene_same_verdict_at_any_resolution(W, H):
    r = client.post("/v1/uniformity", files={"file": ("s.jpg", jpg(scene(W, H)))}).json()
    assert r["grid_passed"] and not r["passed"]
    assert r["spots"]["count"] == 1                       # 只抓到 1% 白點，0.3% 灰塵被忽略
    s = r["spots"]["items"][0]
    assert s["size_pct"] == pytest.approx(1.0, abs=0.15)  # 回報的相對尺寸與解析度無關
    assert abs(s["center"][0] - 0.3 * W) <= W * 0.005 and abs(s["center"][1] - 0.4 * H) <= H * 0.005
    assert r["spots"]["min_area_px"] == pytest.approx(np.pi / 4 * (0.005 * W) ** 2, rel=0.02)


NOISE_CASES = [((128, 128, 128), 400, 300, 3), ((128, 128, 128), 300, 200, 4), ((128, 128, 128), 1200, 900, 4),
               ((128, 128, 128), 2400, 1800, 4), ((200, 22, 34), 400, 300, 4), ((30, 90, 160), 800, 600, 4),
               ((240, 200, 60), 1600, 1200, 5)]


@pytest.mark.parametrize("seed", [2, 5, 9])
@pytest.mark.parametrize("color,W,H,sd", NOISE_CASES)
def test_noise_not_reported_at_any_size(color, W, H, sd, seed):
    """各種尺寸、顏色的均勻影像加上相機雜訊，不可誤判出瑕疵。"""
    from uniformity import analyze
    a = np.clip(np.full((H, W, 3), color, float) + np.random.default_rng(seed).normal(0, sd, (H, W, 3)), 0, 255)
    r = analyze(a.astype(np.uint8))
    assert r.passed and r.spots["count"] == 0


@pytest.mark.parametrize("W,expected", [(800, 0.62), (1200, 0.42), (4000, 0.25)])
def test_detectable_min_reported(W, expected):
    r = summarize_of(np.full((W * 3 // 4, W, 3), RED), heatmap="false") if W < 4000 else None
    if r is None:  # 大圖用 JPEG 避免超過上傳上限
        r = client.post("/v1/uniformity", files={"file": ("b.jpg", jpg(np.full((3000, 4000, 3), RED)))},
                        data={"heatmap": "false"}).json()
    assert r["spots"]["detectable_min_pct"] == pytest.approx(expected, abs=0.02)


def test_low_resolution_warning_in_summary():
    r = summarize_of(np.full((300, 400, 3), RED), min_spot_size=0.5)
    assert r["spots"]["detectable_min_pct"] > 0.5 and "本張照片解析度較低" in r["summary"]
    r = summarize_of(np.full((900, 1200, 3), RED), min_spot_size=0.5)
    assert "本張照片解析度較低" not in r["summary"]


# ---------------------------------------------------------------- 漸層拆解（max_gradient）與周圍比較（spot_mode=local）

def panel(H=450, W=600, shade=(1.0, 1.0), faint=None, white=None, blotch=None, seed=0):
    """紅色面板：shade = 左、右亮度倍率（打光漸層）；faint／white／blotch = (cx, cy, r)。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    img = np.array([200, 22, 34], float) * np.linspace(*shade, W)[None, :, None] * np.ones((H, W, 3))
    disk = lambda cx, cy, r: ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)  # noqa: E731
    if blotch:
        img[disk(*blotch)] *= 1.22
    if faint:
        img[disk(*faint)] *= 1.22
    if white:
        img[disk(*white)] = (250, 240, 240)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


LIT = dict(shade=(1.25, 0.62))                       # 左亮右暗，ΔE 約 20
NEW = dict(threshold=3.0, spot_threshold=5.0, min_spot_size=0.5)


def test_new_site_fields_default_to_original_behaviour():
    cfg, src = main.registry.resolve(None, {})       # 測試設定檔沒寫新欄位
    assert (cfg["max_gradient"], cfg["spot_mode"], cfg["background_size"]) == (None, "global", 10.0)
    assert src["spot_mode"] == "default"


@pytest.mark.parametrize("field,value", [("spot_mode", "nearby"), ("background_size", 0),
                                         ("background_size", 80), ("max_gradient", -1)])
def test_new_site_fields_validated(field, value):
    raw = {"default": {"threshold": 3.0, "spot_threshold": None, "min_spot_size": 0.5, "rows": 5,
                       "cols": 5, "margin": 0.1, "reference": None, field: value}}
    with pytest.raises(SiteConfigError, match=field):
        parse_config(raw)


def test_project_sites_yaml_site_c():
    reg = SiteRegistry(Path(main.__file__).resolve().parent / "sites.yaml")
    cfg, _ = reg.resolve("site_c", {})
    assert (cfg["max_gradient"], cfg["spot_mode"], cfg["spot_threshold"]) == (25.0, "local", 5.0)
    assert reg.resolve(None, {})[0]["spot_mode"] == "global"


def test_lighting_gradient_fails_reference_mode_but_passes_detrended():
    rgb = panel(**LIT)
    old = analyze(rgb, **NEW)
    assert not old.grid_passed and old.delta_e_max > 15 and old.grid_mode == "reference" and old.gradient is None
    new = analyze(rgb, **NEW, max_gradient=25)
    assert new.grid_passed and new.delta_e_max < 1 and new.grid_mode == "detrended"
    g = new.gradient
    assert 15 < g["span"] < 25 and g["passed"] and g["bright_side"] == "左側" and g["dark_side"] == "右側"


def test_gradient_beyond_limit_fails():
    r = analyze(panel(**LIT), **NEW, max_gradient=10)
    assert not r.grid_passed and not r.gradient["passed"] and r.cells_over_threshold == 0


def test_detrended_still_catches_central_blotch():
    r = analyze(panel(blotch=(300, 225, 75)), **NEW, max_gradient=25)
    assert not r.grid_passed and r.delta_e_max > 5
    assert r.worst_cell["position"] == "中央" and r.worst_cell["tone"] in ("偏亮", "泛白")


def test_local_mode_ignores_gradient_and_finds_faint_spot():
    rgb = panel(**LIT, faint=(450, 100, 10), white=(150, 330, 8))
    glob = analyze(rgb, **NEW)
    assert glob.spots["region_count"] >= 1                     # 打光漸層被當成大面積瑕疵
    loc = analyze(rgb, **NEW, spot_mode="local")
    pos = sorted((it["position"], it["tone"]) for it in loc.spots["items"])
    assert loc.spots["count"] == 2 and loc.spots["region_count"] == 0
    assert [p for p, _ in pos] == ["右上方", "左下方"]
    assert dict(pos)["右上方"] in ("偏亮", "泛白") and dict(pos)["左下方"] == "泛白"
    assert loc.spots["mode"] == "local" and loc.spots["background_size_pct"] == 10
    assert glob.spots["mode"] == "global" and glob.spots["background_size_pct"] is None


def test_local_mode_misses_large_blotch_but_stage1_catches_it():
    r = analyze(panel(blotch=(300, 225, 120)), **NEW, max_gradient=25, spot_mode="local")
    assert r.spots["region_count"] == 0 and not r.grid_passed


@pytest.mark.parametrize("kw", [dict(max_gradient=0), dict(spot_mode="nearby"), dict(background_size=0.5)])
def test_analyze_rejects_bad_new_params(kw):
    with pytest.raises(ValueError):
        analyze(panel(), **kw)


def test_api_new_params_and_summary():
    data = png(panel(**LIT, white=(150, 300, 8)))
    r = client.post("/v1/uniformity", files={"file": ("lit.png", data, "image/png")},
                    data={"max_gradient": "25", "spot_mode": "local", "spot_threshold": "5"})
    body = r.json()
    assert r.status_code == 200 and body["grid_passed"] and not body["spots"]["passed"]
    assert body["applied_settings"]["spot_mode"] == {"value": "local", "source": "request"}
    assert body["gradient"]["limit"] == 25 and body["grid_mode"] == "detrended"
    s = body["summary"]
    assert "扣除平滑漸層後" in s and "左側較亮、右側較暗" in s and "在容許範圍內" in s
    assert "與周圍比較（範圍長邊 10%）" in s


def test_api_gradient_over_limit_summary():
    body = client.post("/v1/uniformity", files={"file": ("lit.png", png(panel(**LIT)), "image/png")},
                       data={"max_gradient": "10"}).json()
    assert "第一階段整體均勻度不通過" in body["summary"] and "超過上限" in body["summary"]
    assert "未超過" in body["summary"]


def test_api_rejects_bad_spot_mode():
    r = client.post("/v1/uniformity", files={"file": ("a.png", png(panel()), "image/png")},
                    data={"spot_mode": "nearby"})
    assert r.status_code == 422


def test_summary_skips_coverage_warning_when_both_local():
    data = png(panel(**LIT))
    both = client.post("/v1/uniformity", files={"file": ("lit.png", data, "image/png")},
                       data={"max_gradient": "25", "spot_mode": "local", "spot_threshold": "5"}).json()
    assert both["reference_coverage"] < 0.5
    assert "注意" not in both["summary"] and "未使用整張主色" in both["summary"]
    only_spot = client.post("/v1/uniformity", files={"file": ("lit.png", data, "image/png")},
                            data={"spot_mode": "local"}).json()       # 第一階段仍對主色 → 保留提醒
    assert "基準色為影像主色" in only_spot["summary"]
