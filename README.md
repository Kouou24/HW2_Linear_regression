# 雙資料集線性回歸與特徵選擇分析 (Dual-Dataset Linear Regression & CRISP-DM)
> 基於 CRISP-DM 流程之線性回歸 (Linear Regression) 建模、特徵選擇甜點區、區間估計 (95% CI & PI) 與雙資料集互動儀表板

---

## 🔗 專案連結與線上展示 (Links & Live Demo)
* 🌐 **GitHub Pages (Live Demo)**: [https://kouou24.github.io/HW2_Linear_regression/](https://kouou24.github.io/HW2_Linear_regression/)
* 📦 **GitHub Repository**: [https://github.com/Kouou24/HW2_Linear_regression](https://github.com/Kouou24/HW2_Linear_regression)
* 🌐 **Result**:
<img width="1920" height="4659" alt="image" src="https://github.com/user-attachments/assets/e7d682e0-a0ef-42b8-b8ca-0a4d13e7e719" />

---

## 🌐 互動展示網頁 (Interactive Demo)
本專案附帶可獨立執行的現代化響應式互動網頁應用程式 [`index.html`](./index.html)，具備 **雙資料集即時切換下拉式選單**、**即時推論計算機 (Inference Calculator)**、**動態統計區間估計 (95% CI & PI)** 與 **高解析圖表畫廊**：

* 🚀 **線上直接體驗 (GitHub Pages)**: [https://kouou24.github.io/HW2_Linear_regression/](https://kouou24.github.io/HW2_Linear_regression/)
* 💻 **本地即時預覽**：直接雙擊開啟工作目錄中的 [`index.html`](./index.html)。
* 🖥️ **本地 HTTP 伺服器啟動方式**：
  ```bash
  python -m http.server 8000
  ```
  啟動後於瀏覽器訪問：[http://localhost:8000/index.html](http://localhost:8000/index.html)

---

## 📊 雙資料集核心實驗結論與甜點區對比 (Sweet Spot Summary)

| 資料集 (Dataset) | 樣本數 | 候選特徵數 | 最佳甜點區 ($k$) | 選取核心特徵 | 測試集 $R^2$ | 測試集 RMSE | 改善與實證說明 |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **1. 學生考試成績**<br>`StudentPerformanceFactors.csv` | 6,607 筆 | 27 個 (One-Hot) | **$k=24$ ⭐** | `Attendance`, `Hours_Studied`, `Previous_Scores`, `Tutoring_Sessions`, `Peer_Influence` 等 | **0.7697** | **1.8041 分** | 剔除性別、公私立與睡眠時數 3 個雜訊特徵，Adj-$R^2$ 達 0.7655，MAE 僅 0.45 分。 |
| **2. 各國預期壽命**<br>`Life_Expectancy_Data.csv` | 2,938 筆 | 19 個 (WHO健康經濟) | **$k=14$ ⭐** | `Schooling`, `Income composition`, `Adult Mortality`, `HIV/AIDS`, `BMI`, `Polio`, `GDP` 等 | **0.8204** | **3.9413 歲** | 剔除支出比率、總支出、人口與消瘦等 5 個無顯著變數，泛化能力超越全特徵模型。 |

---

## 專案目錄與架構

```text
HW2_Linear_regression/
│
├── StudentPerformanceFactors.csv       # 資料集一：學生考試成績 (6,607 筆，20 欄位)
├── Life_Expectancy_Data.csv            # 資料集二：各國預期壽命 (2,938 筆，22 欄位)
├── main.py                             # 主程式 (雙資料集 CRISP-DM 一體化執行)
├── index.html                          # 雙資料集切換互動式視覺化儀表板 (Live Demo)
├── README.md                           # 完整分析與成果總整理報告
│
├── figures/                            # 成果視覺化圖表
│   ├── correlation_heatmap.png         # [資料集一] 學生特徵相關係數熱圖
│   ├── simple_regression_ci_pi.png     # [資料集一] 單純回歸預測圖 (含 95% CI 與 PI)
│   ├── multiple_regression_prediction_intervals.png # [資料集一] 多元回歸預測區間帶 (PI Ribbon)
│   ├── feature_importance.png          # [資料集一] 特徵標準化係數 (Beta) 排序圖
│   ├── residual_diagnostics.png        # [資料集一] 殘差四合一診斷圖
│   ├── model_comparison.png            # [資料集一] 模型評估指標柱狀圖
│   │
│   ├── life_correlation_heatmap.png    # [資料集二] 預期壽命特徵相關係數熱圖
│   ├── life_simple_regression_ci_pi.png # [資料集二] 單純回歸預測圖 (Schooling vs. Life)
│   ├── life_multiple_regression_prediction_intervals.png # [資料集二] 多元回歸預測區間帶
│   ├── life_feature_importance.png     # [資料集二] 特徵標準化係數 (Beta) 排序圖
│   ├── life_residual_diagnostics.png   # [資料集二] 殘差四合一診斷圖
│   └── life_model_comparison.png       # [資料集二] 模型評估指標柱狀圖
│
└── models/                             # 評估量化指標記錄
    ├── metrics.json                    # 資料集一評估數據
    └── life_metrics.json               # 資料集二評估數據
```

---

## CRISP-DM 流程剖析 (雙資料集實踐)

```mermaid
flowchart LR
    A["1. 業務理解<br>(Business Understanding)"] --> B["2. 資料理解<br>(Data Understanding)"]
    B --> C["3. 資料準備<br>(Data Preparation)"]
    C --> D["4. 模型建置<br>(Modeling)"]
    D --> E["5. 模型評估<br>(Evaluation)"]
    E --> F["6. 模型部署<br>(Deployment)"]
    E -.->|特徵工程優化| D
```

---

### 第一階段：業務理解 (Business Understanding)

#### 1. 資料集一：學生學業表現
* **核心問題**：學生學業表現受學習投入、家庭背景與學校環境多重影響。本研究旨在建立透明、高解釋力之線性回歸模型，提供 **95% 信賴區間 (CI)** 與 **95% 預測區間 (PI)**，為教育現場提供早期預警。
* **成功指標**：$R^2 \ge 0.75$、$\text{RMSE} \le 2.0$。

#### 2. 資料集二：WHO 各國預期壽命
* **核心問題**：各國國民預期壽命存在顯著跨國不平等。分析免疫接種（Polio、Diphtheria、Hepatitis B）、死亡率（成人、嬰兒）、經濟發展（GDP、所得結構）與教育年數對國民壽命之邊際影響。
* **成功指標**：$R^2 \ge 0.80$、$\text{RMSE} \le 4.5$ 歲。

---

### 第二階段：資料理解 (Data Understanding)

#### 1. 資料概況與目標變數分佈
* **學生資料集**：6,607 筆樣本，目標變數 `Exam_Score`（均值 67.24，標準差 3.89，範圍 55~101 分）。
* **預期壽命資料集**：2,938 筆樣本（涵蓋 193 國），目標變數 `Life expectancy`（均值 69.22，標準差 9.52，範圍 36.3~89.0 歲）。

#### 2. 特徵相關性矩陣
* **學生資料集最強相關**：
  * `Attendance` ($r = +0.581$)
  * `Hours_Studied` ($r = +0.445$)
* **預期壽命資料集最強相關**：
  * `Schooling` (平均受教育年數, $r = +0.717$)
  * `Income composition of resources` (所得資源結構, $r = +0.689$)
  * `Adult Mortality` (成人死亡率, $r = -0.696$)
  * `HIV/AIDS` (愛滋病死亡率, $r = -0.557$)

---

### 第三階段：資料準備 (Data Preparation)

1. **缺失值處理**：
   * 學生資料集：`Teacher_Quality` (78筆)、`Parental_Education_Level` (90筆)、`Distance_from_Home` (67筆) 採眾數 (Mode) 補齊。
   * 預期壽命資料集：排除 10 筆標籤缺失樣本，特徵欄位採中位數 (Median) 補齊。
2. **類別特徵編碼**：
   * 採用虛擬變數編碼 (Dummy Encoding, `drop_first=True`) 消除共線性陷阱。
3. **資料分割**：
   * 兩資料集皆以 80/20 比例劃分訓練集與獨立測試集 (`random_state=42`)。

---

### 第四階段：模型建置與特徵選擇 (Modeling & Feature Selection)

本專案於雙資料集皆建置並比較四大回歸架構：

1. **Model 1: 單純線性回歸 (Simple Linear Regression)**
   * 學生資料集：以 `Attendance` 單變數建模。
   * 預期壽命資料集：以 `Schooling` 單變數建模。
2. **Model 2: 多元線性回歸全特徵 (Multiple Linear Regression - Full)**
   * 普通最小平方法 (OLS) 擬合全候選變數。
3. **Model 3: 特徵選擇精簡模型 (Feature Selection - Reduced Model ★ 最佳)**
   * 結合 **p-value 反向逐步淘汰法** 與 **Lasso L1 正則化**：
     * **學生資料集**：淘汰 `Sleep_Hours` ($p=0.53$)、`School_Type_Public` ($p=0.78$)、`Gender_Male` ($p=0.66$) 等 3 個雜訊特徵，精簡為 24 特徵。
     * **預期壽命資料集**：淘汰 `percentage expenditure` ($p=0.54$)、`Total expenditure` ($p=0.19$)、`Population` ($p=0.56$)、`thinness 1-19 years` ($p=0.07$)、`thinness 5-9 years` ($p=0.90$) 等 5 個無效特徵，精簡為 14 特徵。
4. **Model 4: Auto Regression (自動化正則化 & 基線消融實驗)**
   * 實作 `ElasticNetCV` 進行 L1/L2 雙重正則化超參數自動尋優。
   * **消融實驗**：
     * 學生資料集移除歷史成績 `Previous_Scores` 後，$R^2$ 仍達 0.7355（73.6% 變異由當前學習行為解釋）。
     * 預期壽命資料集移除 `Schooling` 與 `Income composition` 後，$R^2$ 從 0.8204 驟降至 0.7421，實證教育與經濟基線對長壽貢獻超過 7.8% 邊際解釋力。

---

### 第五階段：模型評估 (Model Evaluation)

#### 1. 雙資料集評估指標對照表

##### (A) 資料集一：學生學業表現 (`StudentPerformanceFactors.csv`)
| 模型名稱 | 使用特徵數 | 測試集 $R^2$ | 調整後 $R^2_{\text{adj}}$ | 均方根誤差 (RMSE) | 平均絕對誤差 (MAE) | 備註說明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1: 單純線性 (Attendance)** | 1 | 0.3476 | — | 3.0367 | 2.1079 | 僅能解釋約 35% 變異 |
| **Model 2: 多元全特徵 OLS** | 27 | 0.7696 | 0.7648 | 1.8044 | 0.4524 | 含 3 個無顯著性雜訊變數 |
| **Model 3: 特徵選擇精簡 (最佳 ★)** | **24** | **0.7697** | **0.7655** | **1.8041** | **0.4515** | **剔除噪音，保留最佳解釋力** |
| **Model 4A: ElasticNetCV 自動化** | 24 | 0.7544 | — | 1.8634 | 0.6464 | 具抗過擬合能力 |
| **Model 4B: 去除自回歸項** | 23 | 0.7355 | 0.7308 | 1.9332 | 0.5012 | 當前投入仍佔 73.6% 解釋力 |

##### (B) 資料集二：WHO 各國預期壽命 (`Life_Expectancy_Data.csv`)
| 模型名稱 | 使用特徵數 | 測試集 $R^2$ | 調整後 $R^2_{\text{adj}}$ | 均方根誤差 (RMSE) | 平均絕對誤差 (MAE) | 備註說明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1: 單純線性 (Schooling)** | 1 | 0.5517 | — | 6.2269 歲 | 4.6396 歲 | 單一受教育年數即解釋 55.2% |
| **Model 2: 多元全特徵 OLS** | 19 | 0.8194 | 0.8133 | 3.9527 歲 | 2.9281 歲 | 含 5 個不顯著無效變數 |
| **Model 3: 特徵選擇精簡 (最佳 ★)** | **14** | **0.8204** | **0.8160** | **3.9413 歲** | **2.9280 歲** | **精簡 26% 特徵，泛化表現超越全特徵** |
| **Model 4A: ElasticNetCV (標準化)** | 14 | 0.8136 | — | 4.0159 歲 | 2.9776 歲 | 正則化收縮，抗共線性能力強 |
| **Model 4B: 去除教育經濟基線** | 12 | 0.7421 | 0.7365 | 4.7120 歲 | 3.5102 歲 | 證實教育與所得為壽命基石 (+7.8%) |

#### 2. 信賴區間 (CI) 與 預測區間 (PI) 統計意涵
* **95% 信賴區間 (Confidence Interval, CI)**：估計的是「特定條件下群體的**平均期望值 $E(Y \mid X)$**」，隨樣本增加而收斂（帶狀極窄）。
* **95% 預測區間 (Prediction Interval, PI)**：估計的是「**單一特定觀測體**未來的實際值 $Y_0$」，計入個體未觀測隨機誤差 $\sigma^2$（帶狀較寬），能真實量化極端風險。

---

### 第六階段：模型部署與應用 (Deployment)

#### 1. 互動式網頁儀表板 (`index.html`)
網頁頂部設有下拉式切換選單，可在 **學生學業成績** 與 **各國預期壽命** 之間即時切換：
* 包含完整即時推論計算機，隨動調整滑桿與輸入值，前端即時輸出預測值、95% CI 與 95% PI。
* 內建高學習投入/高風險學生、已開發/開發中國家情境按鈕。

#### 2. 本地執行與重現
```bash
# 執行雙資料集完整分析與圖表生成
python -X utf8 main.py

# 啟動本地網頁預覽
python -m http.server 8000
```
瀏覽器造訪：[http://localhost:8000/index.html](http://localhost:8000/index.html)

---

## 🚀 部署至 GitHub Pages (Deploy to github.io) 指引

本專案之互動儀表板為無伺服器純靜態前端架構，可直接免費託管於 **GitHub Pages** (`https://<username>.github.io/<repo>/`)：

### 步驟 1：提交並推送程式碼至 GitHub
在專案根目錄開啟終端機（PowerShell），執行 Git 指令：
```bash
git add .
git commit -m "feat: add dual-dataset support and interactive dashboard"
git push origin main
```

### 步驟 2：啟用 GitHub Pages
1. 開啟瀏覽器進入儲存庫：[https://github.com/Kouou24/HW2_Linear_regression](https://github.com/Kouou24/HW2_Linear_regression)
2. 點選儲存庫頂部選單的 **Settings**（設定）。
3. 在左側側邊欄中點選 **Pages**（頁面）。
4. 在 **Build and deployment** 區塊：
   - **Source** 下拉選單選擇：`Deploy from a branch`。
   - **Branch** 選擇：`main`，資料夾選擇：`/ (root)`。
   - 點選 **Save**（儲存）。

### 步驟 3：查看線上即時成果
等待約 1~2 分鐘 GitHub Actions 自動建置完成後，即可直接透過專屬網址存取：
👉 **[https://kouou24.github.io/HW2_Linear_regression/](https://kouou24.github.io/HW2_Linear_regression/)**
