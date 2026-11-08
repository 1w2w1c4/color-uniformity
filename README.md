# Color Uniformity API

**不需要標準色**：每張圖會自動找出自己的「主色」（佔最多像素的那群相近顏色）作為基準，再判斷其他區域是否偏離。
前提是正常顏色在畫面中佔比最大；若泛白面積超過正常色，主色會變成白色，此時仍會判為不均勻，但瑕疵位置會標在紅色區。

上傳影像後分兩階段判定，兩者皆通過才算 `passed`：

1. **整體偵測**：切成 rows×cols 格 → 每格取中位數色並轉 CIELAB → 以 CIEDE2000 計算與參考色的色差。抓大範圍漸層、單側泛白。
2. **小點偵測**：整體偵測後，再逐像素與同一參考色比對，超過門檻的相鄰像素合併成瑕疵區塊。抓單一小白點等格子平均會漏掉的局部瑕疵。

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --reload        # 互動文件：http://127.0.0.1:8000/docs
.venv/bin/pip install pytest httpx && .venv/bin/pytest -q
```

```bash
curl -F file=@panel.jpg -F site=site_a http://127.0.0.1:8000/v1/uniformity

# 個別參數可覆蓋 site 標準（測試／調參用）
curl -F file=@panel.jpg -F site=site_a -F threshold=2.5 [-F roi=100,50,800,600] \
     http://127.0.0.1:8000/v1/uniformity
```

## 照片尺寸不固定

所有標準都以「相對照片大小」表示，同一場景不論用多少解析度拍，判定結果相同：

- 第一階段切格（rows×cols）本來就是比例。
- `min_spot_size`：最小瑕疵尺寸 = **照片長邊的百分比**（等面積圓直徑）。0.5 代表 1200px 寬的照片忽略 < 6px 的點、4000px 寬的忽略 < 20px 的點。
- 每個瑕疵回報 `size_pct`（同樣以長邊百分比表示），可直接與 `min_spot_size` 比較。
- 小點偵測一律至少縮小 2 倍並做開運算去雜訊，避免小照片把相機雜訊誤判成瑕疵。
- 照片越小，能分辨的瑕疵越大：回傳 `spots.detectable_min_pct` 為本張照片的偵測下限（約：800px → 0.62%、1200px → 0.42%、4000px → 0.25%）。若 `min_spot_size` 低於此值，`summary` 會提醒。

> 標準是相對於「照片」而非「產品」：若拍攝距離不同、產品在畫面中佔比不同，請用 `roi` 只框產品範圍，標準才會對應到產品上的同一實際大小。

## 熱圖輸出

每次呼叫會在 `result/` 輸出 `<檔名>_heatmap.png`（路徑見回傳的 `heatmap_path`；`-F heatmap=false` 可關閉；環境變數 `RESULT_DIR` 可改輸出位置）。同檔名會覆蓋。

- 標題列：合格／不合格（圖示＋文字）、檔名、site、基準色
- 左：原圖，局部瑕疵以描邊標示，過小的另加方框，編號與 JSON `spots.items` 順序相同
- 中：第一階段格子熱圖，每格標 ΔE 值，粗黑框 = 超過整體門檻
- 右：第二階段逐像素熱圖
- 色階固定 0–10（淺藍 = 均勻、深藍 = 不均勻，≥10 皆最深），不同圖片可直接比較；標出兩階段門檻
- 底部：`summary` 綜合說明

## Site 標準

各 site 的判定標準寫在 [sites.yaml](sites.yaml)（也可用環境變數 `SITES_CONFIG` 指定其他路徑），修改後不需重啟，下一次請求即生效。

- 優先序：**請求參數 > site 標準 > default**；site 沒寫的欄位沿用 default。
- 可設定欄位：`threshold`、`spot_threshold`、`min_spot_size`、`rows`、`cols`、`margin`、`reference`。
- 回傳的 `site` 與 `applied_settings` 會列出每個欄位實際採用的值與來源（`request` / `site:<名稱>` / `default`），方便追溯。
- `GET /v1/sites` 查詢所有 site 的完整標準；未知的 site 回 422 並列出可用名稱；設定檔有誤回 500 並指出錯誤欄位。

| 回傳欄位 | 意義 |
|---|---|
| `summary` | 一段中文說明：結論＋第一階段（整體）結果＋第二階段（局部瑕疵）結果＋基準色 |
| `passed` / `score` | 整體與小點皆通過才為 true；score 0–100（達門檻 = 60），取兩階段較低者 |
| `grid_passed` | 第一階段（整體）結果：最大格子 ΔE00 ≤ threshold |
| `spots.passed` / `spots.count` | 第二階段（小點）結果：沒有任何尺寸 ≥ min_spot_size 的瑕疵即通過 |
| `spots.items[]` | 每個瑕疵：`type`（spot 小點／region 佔畫面 ≥5% 的大面積）、`tone`（泛白／偏亮／偏暗／變灰／偏濃／偏色）、`position`（左上方…中央…右下方）、`center`、`bbox` [x,y,w,h]、`area_px`、`delta_e_max`，座標皆為原圖像素 |
| `spots.defect_area_ratio` | 瑕疵總面積佔比 |
| `reference_hex` / `reference_lab` | 本張圖採用的基準色（自動找出的主色） |
| `reference_coverage` | 與主色相符（ΔE00 ≤ 小點門檻）的像素比例 |
| `delta_e_grid` | 每格 ΔE00，可直接畫熱圖找出不均位置 |
| `worst_cell` | 色差最大的格子（row, col 從 0 起算），含 `tone`、`position` |
| `cells_over_threshold` | 超過門檻的格數 |
| `delta_e_max / mean / p95` | 色差統計 |
| `lightness_std` | 各格 L* 標準差 |
| `luminance_uniformity` | 最暗格 Y ÷ 最亮格 Y（顯示器/照明常用） |

`summary` 範例：

> 【不合格】整體顏色均勻，但有局部瑕疵。第一階段整體均勻度通過：8×8 格中最大色差 ΔE 0.28（門檻 2，幾乎看不出）。第二階段局部瑕疵不通過：發現 1 處小點（門檻 ΔE 2.5、最小面積 15 px²），瑕疵總面積佔畫面 0.07%；最嚴重的是畫面中央的泛白小點，中心 (721, 541)、約 30×30 px，色差 ΔE 49.44（非常明顯）。基準色為影像主色 #C81622，佔畫面 99.9%。

ΔE00 參考：< 1 幾乎看不出、1–2 仔細看才看得出、2–3.5 一般人可察覺、> 5 明顯不同。
