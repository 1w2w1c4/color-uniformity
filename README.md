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

- 標題列：Pass／Fail（圖示＋文字）、檔名、site、基準色
- 左：原圖，局部瑕疵以描邊標示，過小的另加方框，編號與 JSON `spots.items` 順序相同
- 中：第一階段格子熱圖，每格標 ΔE 值，粗黑框 = 超過整體門檻
- 右：第二階段逐像素熱圖
- 色階固定 0–20（淺藍 = 均勻、深藍 = 不均勻，≥20 皆最深），不同圖片可直接比較；標出兩階段門檻
- 底部：`summary` 綜合說明

## Site 標準

各 site 的判定標準寫在 [sites.yaml](sites.yaml)（也可用環境變數 `SITES_CONFIG` 指定其他路徑），修改後不需重啟，下一次請求即生效。

- 優先序：**請求參數 > site 標準 > default**；site 沒寫的欄位沿用 default。
- 可設定欄位：`threshold`、`spot_threshold`、`min_spot_size`、`rows`、`cols`、`margin`、`reference`、`max_gradient`、`spot_mode`、`background_size`（新欄位在 default 沒寫時採原本做法，舊設定檔不需修改）。
- 回傳的 `site` 與 `applied_settings` 會列出每個欄位實際採用的值與來源（`request` / `site:<名稱>` / `default`），方便追溯。
- `GET /v1/sites` 查詢所有 site 的完整標準；未知的 site 回 422 並列出可用名稱；設定檔有誤回 500 並指出錯誤欄位。

### 打光不均時：漸層拆解與周圍比較

原本兩階段都和「整張照片的主色」比較，打光不均（例如左亮右暗）會讓正常區域也超過門檻。兩個選項可分別開啟（範例見 `sites.yaml` 的 `site_c`）：

| 欄位 | 作用 | 盲點 |
|---|---|---|
| `max_gradient`（第一階段） | 以平面擬合各格顏色，拆成「平滑漸層」與「殘差」。殘差 ≤ `threshold` 且漸層幅度 ≤ `max_gradient` 才通過。中央泛色、斑駁仍會被殘差抓到 | 幅度在上限內的平順單側褪色與打光無法區分，會被放過。根本解法是改善打光或平場校正 |
| `spot_mode: local`（第二階段） | 每個像素改和周圍背景比較（範圍 `background_size`，長邊 %），漸層被抵消，`spot_threshold` 可降回約 5，淡色斑也抓得到 | 比背景範圍大的瑕疵會被當成背景，只剩邊緣；需由第一階段負責 |

兩者一起開時，大面積由第一階段（殘差）負責、小點由第二階段（周圍）負責。也可用請求參數試算：`-F max_gradient=25 -F spot_mode=local -F spot_threshold=5`。

| 回傳欄位 | 意義 |
|---|---|
| `summary` | 一段中文說明：結論＋第一階段（整體）結果＋第二階段（局部瑕疵）結果＋基準色 |
| `passed` / `score` | 整體與小點皆通過才為 true；score 0–100（達門檻 = 60），取兩階段較低者 |
| `grid_passed` | 第一階段（整體）結果：最大格子 ΔE00 ≤ threshold（`max_gradient` 有值時另需漸層幅度 ≤ max_gradient） |
| `grid_mode` / `gradient` | `reference`＝各格對主色；`detrended`＝各格對擬合漸層，此時 `delta_e_grid` 為殘差，`gradient` 含 `span`（漸層幅度）、`limit`、`passed`、`bright_side`／`dark_side` |
| `spots.mode` | 第二階段比較對象：`global` 整張主色／`local` 周圍背景（`spots.background_size_pct` 為範圍） |
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

> 【Fail】整體顏色均勻，但有局部瑕疵。第一階段整體均勻度通過：8×8 格中最大色差 ΔE 0.28（門檻 2，幾乎看不出）。第二階段局部瑕疵不通過：發現 1 處小點（門檻 ΔE 2.5、最小面積 15 px²），瑕疵總面積佔畫面 0.07%；最嚴重的是畫面中央的泛白小點，中心 (721, 541)、約 30×30 px，色差 ΔE 49.44（非常明顯）。基準色為影像主色 #C81622，佔畫面 99.9%。

ΔE00 參考：< 1 幾乎看不出、1–2 仔細看才看得出、2–3.5 一般人可察覺、> 5 明顯不同。

## 壓力模式（pressure.py，感壓紙掃描影像）

另一種判定方式：不比較色差，而是把顏色換算成**濃度**（≈ 壓力），分析壓力分布。適用於感壓紙等「顏色深淺代表壓力大小」的掃描影像。

```bash
.venv/bin/python pressure.py scan.png [--dpi 600] [--roi x,y,w,h] [--white #F6F0EE] [--json]
.venv/bin/python pressure.py images/*.png --max-cv 0.1 --min-defect-mm 0.5   # 所有判定標準都可用參數覆蓋
```

1. **濃度**：取互補色通道（紅色感壓紙 → G）算 `OD = -log10(I / I_白紙)`。白紙基準自動取畫面中未受壓的白紙，找不到時用預設白紙色，所以**掃描時保留一小塊白紙**會比較準。
2. **相對壓力**：濃度 ÷ 受壓區中位數。型號未知、沒有校正曲線時，只判斷相對均勻度，不換算成 MPa。
3. **判定**（全部通過才合格）：

| 項目 | 預設標準 | 意義 |
|---|---|---|
| `cv` | ≤ 15% | 相對壓力變異係數 |
| `uniformity_p5_p95` | ≥ 0.70 | 低壓處 ÷ 高壓處 |
| `contact_ratio` | ≥ 95% | 有壓到的面積比例 |
| `tilt` | ≤ 20% | 二次曲面的平面部分：傾斜（平行度不良），`tilt_low_side` 指出偏低的一側 |
| `bow` | \|值\| ≤ 20% | 二次曲面的曲率部分：中凸／中凹（壓板變形） |
| `local_defects` | 0 處 | 與周圍 5 mm 局部背景相比偏差 > 30%、直徑 ≥ 1 mm 的未接觸／低壓點／高壓點 |

4. **尺寸以 mm 表示**：DPI 讀自影像檔（讀不到時用 `--dpi`，預設 300）。
5. **警告**：受壓區濃度 ≥ 1.2 的比例超過 5% 時，提醒可能接近感壓紙飽和；通道值被掃描器截斷時，提醒關閉自動曝光／自動色彩校正。

報告圖輸出到 `result/<檔名>_pressure.png`：左邊是掃描影像加瑕疵編號，右邊是相對壓力圖（藍 = 偏低、紅 = 偏高），附 3×3 分區百分比和指向低壓側的箭頭。

掃描建議：關閉自動色彩校正、自動曝光、去網紋和銳化；固定掃描器、DPI 和擺放位置。

## Agent（自然語言操作，agent.py）

`POST /v1/agent`：用一句話操作上述判定功能，Claude 會自己呼叫工具（`list_sites`、`list_images`、`check_uniformity`、`check_pressure`）並用中文彙整結果。**合格與否仍由判定程式決定**，Claude 只負責選工具與解讀。

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/uvicorn main:app

curl -X POST http://127.0.0.1:8000/v1/agent -H 'Content-Type: application/json' \
     -d '{"message": "用 site_a 檢查所有影像，列出不合格的並說明原因"}'
# 延續對話：帶入上次回傳的 session_id
curl -X POST http://127.0.0.1:8000/v1/agent -H 'Content-Type: application/json' \
     -d '{"message": "如果門檻放寬到 3 呢？", "session_id": "<上次的 session_id>"}'
curl -X DELETE http://127.0.0.1:8000/v1/agent/<session_id>   # 清除對話
```

- 回傳：`reply`（中文回答）、`session_id`、`tool_calls`（實際呼叫了哪些工具、參數、是否出錯）。
- 影像需放在 `IMAGE_DIR`（預設 `images/`），agent 只能讀取此資料夾內的檔案。
- 設定寫在 [config.yaml](config.yaml)（路徑可用 `AGENT_CONFIG` 指定），修改後需重啟。優先序：環境變數（`AGENT_PROVIDER`、`AGENT_MODEL`、`AGENT_EFFORT`、`IMAGE_DIR`）> config.yaml > 預設值。
  - `agent`：用哪個 `provider`、`model`，以及與廠商無關的上限（max_tokens、max_steps…）。
  - `providers.<名稱>`：各 LLM 廠商的金鑰、`base_url`、逾時、重試與專屬參數；只讀取 `agent.provider` 指定的那段。目前已實作 `anthropic`，選用其他名稱會在啟動時報錯。
- 金鑰建議用環境變數（名稱由 `api_key_env` 指定，預設 `ANTHROPIC_API_KEY`），`api_key` 保持 null；若填在檔案裡，請勿 commit。
- 對話歷史存在記憶體，閒置 1 小時清除、重啟即消失；多 worker 部署需改用 Redis 等共用儲存。
- 錯誤：未設定金鑰 503、同一 session 前一個請求未完成 409、Claude API 限流 429、其他 Claude API 錯誤 502。
