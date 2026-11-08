"""pressure.py 測試：以合成的感壓紙掃描影像驗證各判定項目。執行：.venv/bin/pytest -q test_pressure.py"""
import numpy as np
import pytest
from PIL import Image

from pressure import PressureSettings, analyze_pressure, main

RED = np.array([200, 22, 34], float)
WHITE = np.array([246, 240, 238], float)
H, W = 450, 600
YY, XX = np.mgrid[0:H, 0:W]


def scan(p: np.ndarray, paper_margin: bool = False, dpi: int = 300, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    p = np.clip(p, 0, 1)
    if paper_margin:
        p = np.pad(p, 40)
    rgb = WHITE + (RED - WHITE) * p[..., None] ** 1.6 + rng.normal(0, 3, p.shape + (3,))
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img.info["dpi"] = (dpi, dpi)
    return img


def run(p, **kw):
    return analyze_pressure(scan(p, **{k: v for k, v in kw.items() if k in ("paper_margin", "dpi")}),
                            **{k: v for k, v in kw.items() if k not in ("paper_margin", "dpi")})[0]


def test_uniform_passes():
    r = run(np.full((H, W), 0.9))
    assert r["passed"]
    assert r["checks"]["cv"]["value"] < 0.05
    assert r["defects"] == []


def test_tilt_detected_with_low_side():
    r = run(0.95 - 0.4 * XX / W)
    assert not r["checks"]["tilt"]["passed"]
    assert r["shape"]["tilt_low_side"] == "右側"


def test_concave_detected():
    rr = np.hypot((XX - W / 2) / (W / 2), (YY - H / 2) / (H / 2))
    r = run(0.55 + 0.4 * rr ** 2 / 2)
    assert not r["checks"]["bow"]["passed"]
    assert r["shape"]["bow"] < 0 and "中凹" in r["shape"]["bow_shape"]
    assert r["defects"] == []          # 大範圍形狀不應被當成局部瑕疵


def test_single_spot_reported_once_in_mm():
    p = np.clip((np.hypot(XX - 300, YY - 220) - 8) / 4, 0, 1) * 0.9
    r = run(p, dpi=300)
    assert not r["passed"]
    assert len(r["defects"]) == 1
    d = r["defects"][0]
    assert d["position"] == "中央"
    assert abs(d["center"][0] - 300) < 5 and abs(d["center"][1] - 220) < 5
    assert 1.0 < d["size_mm"] < 4.0    # 直徑約 16–24px @300DPI ≈ 1.4–2 mm


def test_spot_size_scales_with_dpi():
    p = np.clip((np.hypot(XX - 300, YY - 220) - 8) / 4, 0, 1) * 0.9
    a = run(p, dpi=300)["defects"][0]["size_mm"]
    b = run(p, dpi=600)["defects"][0]["size_mm"]
    assert b == pytest.approx(a / 2, rel=0.05)


def test_paper_margin_used_as_white_and_roi():
    img = scan(np.full((H, W), 0.9), paper_margin=True)
    r, _ = analyze_pressure(img)
    assert "白紙" in r["white_source"] and "預設" not in r["white_source"]
    r2, _ = analyze_pressure(img, roi=(40, 40, W, H))
    assert r2["passed"] and r2["checks"]["contact_ratio"]["value"] == 1.0


def test_settings_override():
    p = 0.95 - 0.15 * XX / W
    assert not run(p, settings=PressureSettings(max_tilt=0.05))["passed"]
    assert run(p, settings=PressureSettings(max_tilt=1.0))["checks"]["tilt"]["passed"]


def test_blank_image_rejected():
    with pytest.raises(ValueError):
        analyze_pressure(scan(np.zeros((H, W))))


def test_cli_writes_report(tmp_path):
    f = tmp_path / "scan.png"
    scan(np.full((H, W), 0.9)).save(f, dpi=(300, 300))
    assert main([str(f), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "scan_pressure.png").exists()
