# Smart Pilot - 投資組合再平衡系統

基於 PD 控制器 + 卡爾曼濾波的投資組合再平衡系統。

## 架構總覽（v3.0）

```
原始價格 → Log-KF 卡爾曼濾波（兩個通道）→ 乾淨速度信號
                              ↓
                誤差 = 目標權重 - 當前權重
                              ↓
                PD 控制器（用 KF 速度算 D 項）
                              ↓
                死區判斷 → 執行交易或不動
```

三階段驗證流程：
1. Grid Search + SLSQP → 找最佳參數
2. 敏感度分析 → 確認參數穩定性（三維切片熱力圖）
3. 蒙地卡羅模擬 + 三策略對比 + 帕雷托前線圖 → 最終驗證

## 專案結構

```
smart_pilot/
├── app.py                          # Streamlit 主程式（4 分頁）✅
├── core/
│   ├── __init__.py
│   ├── kalman_filter.py           # Log-Space 卡爾曼濾波器 ✅
│   ├── pd_controller.py           # PD 控制器（KF 速度驅動 D 項）✅
│   ├── backtest_engine.py         # 回測引擎（KF + PD）✅
│   ├── benchmark.py               # 三策略回測 + 帕雷托前線掃描 ✅
│   ├── optimizer.py               # Grid Search + SLSQP 最佳化 ✅
│   ├── metrics.py                 # 績效指標計算 ✅
│   ├── portfolio.py               # 投資組合管理
│   └── pid_controller_deprecated.py  # 舊版 PID（已棄用）
├── scripts/
│   ├── run_grid_search.py         # Grid Search 預計算腳本 ✅
│   └── run_slsqp.py               # SLSQP 精確最佳化腳本 ✅
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # yfinance 資料下載 ✅
│   └── cache/                     # 快取資料夾（CSV + JSON）
│       ├── *.csv                  # 價格資料快取
│       ├── grid_search_results.json  # Grid Search 結果快取
│       └── slsqp_results.json     # SLSQP 結果快取
├── validation/
│   ├── __init__.py
│   ├── monte_carlo.py             # 蒙地卡羅模擬 ✅
│   └── out_of_sample_deprecated.py   # 舊版樣本外測試（已棄用）
├── tests/
│   ├── __init__.py
│   ├── test_pid.py                # PID 控制器測試
│   ├── test_data_loader.py        # 資料載入器測試 ✅
│   └── test_backtest_engine.py    # 回測引擎測試
├── requirements.txt               # 依賴套件
├── .streamlit/
│   └── config.toml                # Streamlit 設定
└── README.md
```

## 預設參數

| 參數 | 值 | 說明 |
|------|-----|------|
| 目標資產 | VTI + BND | 股票 + 債券 |
| 目標權重 | 60:40 | target_w = 0.6 |
| 手續費率 | 0.003 | 0.3% |
| KF 暖機期 | 30 天 | 前 30 天不回測 |
| 權重格式 | 小數 | 0.6（非 60） |
| RMSE / Cost | 小數 | 顯示時才乘以 100 |

## 快速上手

### 前置作業

**1. 安裝 Git**

前往 https://git-scm.com/downloads 下載並安裝 Git。

**2. 安裝 Python**

前往 https://www.python.org/downloads/ 下載 Python 3.10 以上版本。
安裝時勾選「Add Python to PATH」。

**3. 下載專案**

打開終端機（Windows 請用 PowerShell），輸入：

```bash
git clone https://github.com/Jueruili/smart_pilot.git
```
**4. 到指定路徑**
```bash
cd ./smart_pilot/
```

### 安裝環境

**5. 建立虛擬環境**

```bash
python -m venv venv
```

**6. 啟動虛擬環境**

Windows：

```powershell
.\venv\Scripts\activate
```

如果出現紅字錯誤（執行原則不允許），請先輸入：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

再重新執行 `.\venv\Scripts\activate`

啟動成功後，終端機最左邊會出現 `(venv)` 字樣。

**7. 安裝套件**

```bash
pip install -r requirements.txt
```

### 啟動系統

**8. 執行應用程式**

```bash
streamlit run app.py
```

瀏覽器會自動開啟，網址是 http://localhost:8501

### 已下載後快速進入
**1. 到指定路徑**
```bash
cd ./smart_pilot/
```
**2. 啟動虛擬環境**
```powershell
.\venv\Scripts\activate
```
**3. 執行應用程式**
```bash
streamlit run app.py
```

### 使用步驟

1. 在左側側邊欄設定股票標的（預設 VTI + BND）
2. 點擊「▶ 執行 Grid Search」產生熱力圖（約 2-5 分鐘）
3. 點擊「▶ 執行 CMA-ES 全域最佳化」找最佳參數（約 5-10 分鐘）
4. 查看四個分頁的分析結果


## 核心概念

### PD + 卡爾曼濾波架構（v3.0）

**信號處理流程：**
1. 原始價格取 log → Log-KF 濾波 → 輸出乾淨的 log 價格和速度
2. 誤差 = 目標權重 - 當前權重
3. PD 控制器：P 項用誤差，D 項用 KF 速度差（vel_stock - vel_bond）
4. 死區判斷 → 執行交易或不動

**卡爾曼濾波器狀態模型：**
```
x(k) = F * x(k-1) + w    （慣性模型：log_price = 前一天 + 速度）
z(k) = H * x(k) + v      （只能觀測價格）
```

### 參數建議

| 參數 | 建議範圍 | 說明 |
|------|----------|------|
| Kp | 0.1 ~ 2.0 | 比例增益，控制響應速度 |
| Kd | 0.1 ~ 2.0 | 微分增益，使用 KF 速度差 |
| Q | 0.0001 ~ 1.0 | KF 過程雜訊（SLSQP 用 log scale） |
| deadband | 0.001 ~ 0.10 | 死區閾值 |

### 最佳化流程

1. **Grid Search**：每組 (Kp, Kd, Q) 掃描多個 deadband，計算超體積作為評分（joblib 平行化）
2. **SLSQP**：對每個 deadband 分別最佳化 (Kp, Kd, Q)，Q 用 log scale
3. **超體積**：面積越大 → 策略在 RMSE-Cost 空間支配越多 → 參數越好

### 超體積指標

用於比較 Pareto Frontier 的優劣：
- 面積越大 → 策略支配的空間越大 → 策略越有價值
- 參考點動態設置：所有策略中最差點 × 1.1

## 開發進度

- [x] Phase 1: 專案結構建立
  - [x] 資料載入模組 (`data/data_loader.py`)
  - [x] 績效指標計算 (`core/metrics.py`)
  - [x] 蒙地卡羅模擬模組 (`validation/monte_carlo.py`)
- [x] Phase 2: v1.0 PID 架構（已棄用）
  - [x] PID 控制器 → `pid_controller_deprecated.py`
  - [x] 樣本外測試 → `out_of_sample_deprecated.py`
- [x] Phase 3: v3.0 PD + KF 架構
  - [x] Log-Space 卡爾曼濾波器 (`core/kalman_filter.py`)
  - [x] PD 控制器 (`core/pd_controller.py`)
  - [x] 回測引擎整合 KF + PD (`core/backtest_engine.py`)
  - [x] 三策略回測 + 帕雷托前線 (`core/benchmark.py`)
  - [x] Grid Search + SLSQP 最佳化 (`core/optimizer.py`)
  - [x] 超體積指標比較
  - [x] Streamlit UI 重構（4 分頁：Pareto、Heatmap、Rolling、Monte Carlo）
  - [x] 滾動窗口分析（Rolling Window Analysis）
  - [x] 預計算腳本 (`scripts/run_grid_search.py`, `scripts/run_slsqp.py`)
  - [ ] 敏感度分析（三維切片熱力圖）

## 授權

MIT License
