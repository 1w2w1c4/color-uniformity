"""把兩階段判定結果組成一段中文說明。"""
from __future__ import annotations

TYPE_NAME = {"spot": "小點", "region": "大面積"}


def visibility(de: float) -> str:
    """ΔE00 → 人眼可見程度。"""
    if de < 1:
        return "幾乎看不出"
    if de < 2:
        return "仔細看才看得出"
    if de < 3.5:
        return "一般人可察覺"
    if de < 5:
        return "明顯可見"
    return "非常明顯"


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def summarize(r: dict) -> str:
    """r：API 回傳的完整結果（座標已換算回原圖）。"""
    verdict = "合格" if r["passed"] else "不合格"

    # 第一階段：整體
    grid = f"{r['rows']}×{r['cols']} 格"
    wc = r["worst_cell"]
    if r["grid_passed"]:
        stage1 = (f"第一階段整體均勻度通過：{grid}中最大色差 ΔE {_fmt(r['delta_e_max'])}"
                  f"（門檻 {_fmt(r['threshold'])}，{visibility(r['delta_e_max'])}）。")
    else:
        stage1 = (f"第一階段整體均勻度不通過：{grid}中有 {r['cells_over_threshold']} 格超過門檻 "
                  f"ΔE {_fmt(r['threshold'])}，最嚴重在畫面{wc['position']}"
                  f"（第 {wc['row'] + 1} 列第 {wc['col'] + 1} 欄），{wc['tone']}，"
                  f"色差 ΔE {_fmt(wc['delta_e'])}（{visibility(wc['delta_e'])}）。")

    # 第二階段：局部瑕疵
    sp = r["spots"]
    rule = f"門檻 ΔE {_fmt(sp['threshold'])}、最小尺寸為長邊 {_fmt(sp['min_spot_size_pct'])}%"
    if sp["passed"]:
        stage2 = f"第二階段局部瑕疵通過：未發現超過標準的瑕疵（{rule}）。"
    else:
        kinds = "、".join(f"{n} 處{TYPE_NAME[t]}" for t, n in
                         (("region", sp["region_count"]), ("spot", sp["spot_count"])) if n)
        top = sp["items"][0]
        x, y, w, h = top["bbox"]
        stage2 = (f"第二階段局部瑕疵不通過：發現 {kinds}（{rule}），瑕疵總面積佔畫面 "
                  f"{sp['defect_area_ratio']:.2%}；最嚴重的是畫面{top['position']}的{top['tone']}"
                  f"{TYPE_NAME[top['type']]}，中心 ({top['center'][0]}, {top['center'][1]})、約 {w}×{h} px"
                  f"（長邊 {_fmt(top['size_pct'])}%），"
                  f"色差 ΔE {_fmt(top['delta_e_max'])}（{visibility(top['delta_e_max'])}）。")

    # 綜合結論
    if r["passed"]:
        conclusion = "整體與局部皆均勻。"
    elif r["grid_passed"]:
        conclusion = "整體顏色均勻，但有局部瑕疵。"
    elif sp["passed"]:
        conclusion = "局部無明顯瑕疵，但整體有大範圍色差。"
    else:
        conclusion = "整體與局部皆不均勻。"

    if sp["min_spot_size_pct"] < sp["detectable_min_pct"]:
        stage2 += (f"（本張照片解析度較低，小於長邊 {_fmt(sp['detectable_min_pct'])}% 的瑕疵可能無法偵測；"
                   "如需偵測更小的瑕疵，請提高拍攝解析度或以 roi 只框產品範圍。）")

    ref = f"基準色為{'影像主色' if r['reference_source'].startswith('影像') else '指定色'} {r['reference_hex']}"
    note = ""
    if r["reference_coverage"] < 0.5:
        note = (f"注意：與基準色相符的面積僅 {r['reference_coverage']:.0%}，"
                "異常面積可能大於正常面積，請確認基準色是否為正常顏色。")

    return f"【{verdict}】{conclusion}{stage1}{stage2}{ref}，佔畫面 {r['reference_coverage']:.1%}。{note}"
