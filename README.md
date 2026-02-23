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
git clone https://github.com/你的帳號/smart_pilot.git
cd smart_pilot/smart_pilot
```

### 安裝環境

**4. 建立虛擬環境**

```bash
python -m venv venv
```

**5. 啟動虛擬環境**

Windows：

```powershell
.\venv\Scripts\activate
```

如果出現紅字錯誤（執行原則不允許），請先輸入：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

再重新執行 `.\venv\Scripts\activate`

Mac / Linux：

```bash
source venv/bin/activate
```

啟動成功後，終端機最左邊會出現 `(venv)` 字樣。

**6. 安裝套件**

```bash
pip install -r requirements.txt
```

### 啟動系統

**7. 執行應用程式**

```bash
streamlit run app.py
```

瀏覽器會自動開啟，網址是 http://localhost:8501

### 使用步驟

1. 在左側側邊欄設定股票標的（預設 VTI + BND）
2. 點擊「▶ 執行 Grid Search」產生熱力圖（約 2-5 分鐘）
3. 點擊「▶ 執行 CMA-ES 全域最佳化」找最佳參數（約 5-10 分鐘）
4. 查看四個分頁的分析結果

### 應用程式四個分頁

| 分頁 | 功能 |
|------|------|
| Pareto Frontier | 三策略帕雷托前線對比，超體積指標比較 |
| Heatmap | 參數空間熱力圖（需先執行 Grid Search） |
| Rolling Window | 滾動窗口分析，評估策略穩定性 |
| Monte Carlo | Bootstrap 蒙地卡羅模擬，風險分析 |

## 使用方式

### 使用 BacktestEngine 執行回測（v3.0）

```python
from data.data_loader import DataLoader
from core.backtest_engine import BacktestEngine

# 載入資料
loader = DataLoader()
data = loader.load_and_process(["VTI", "BND"], "2013-01-01", "2025-12-31")
data = data[["VTI", "BND"]]  # 強制排序

# 初始化回測引擎
engine = BacktestEngine(
    target_ratio=0.6,
    kf_q=0.001,
    kp=0.5,
    kd=0.5,
    deadband=0.0125,
    fee_rate=0.003,
    stock_ticker="VTI",
    bond_ticker="BND",
)

# 執行回測
result = engine.run(data)

# 查看結果
print(f"年化報酬率: {result['metrics']['ann_return_pct']:.2f}%")
print(f"Sharpe: {result['metrics']['sharpe']:.2f}")
print(f"RMSE: {result['rmse']:.4f}")
print(f"Cost: {result['cost']:.4f}")
print(f"交易次數: {result['trade_count']}")
```

**BacktestEngine 回傳格式：**

| 欄位 | 說明 |
|------|------|
| `nav_list` | 每日淨值（從 1.0 開始） |
| `weights` | 每日股票權重（小數） |
| `trade_count` | 總交易次數 |
| `turnover` | 總週轉率 |
| `actions` | 每日交易動作 |
| `rmse` | 追蹤誤差（小數） |
| `cost` | 總交易成本 = turnover * fee_rate（小數） |
| `metrics` | 績效指標（ann_return, volatility, sharpe, max_drawdown） |
| `kf_stock_velocities` | 每天的 KF 股票速度 |
| `kf_bond_velocities` | 每天的 KF 債券速度 |
| `p_terms` | 每天的 P 項 |
| `d_terms` | 每天的 D 項 |
| `u_values` | 每天的總控制量 u |
| `dates` | 回測日期列表 |

### 使用 Benchmark 策略

```python
import numpy as np
from core.benchmark import run_bangbang, run_yearly, run_smart_pilot, scan_pareto_frontier

# 準備資料
prices_stock = data["VTI"].values
prices_bond = data["BND"].values
rets_stock = np.zeros(len(prices_stock))
rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
rets_bond = np.zeros(len(prices_bond))
rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]
dates = data.index.tolist()

# Bang-Bang Control
bb = run_bangbang(rets_stock, rets_bond, dates, target_w=0.6, drift_tolerance=0.05, fee_rate=0.003)

# Yearly Rebalance
yr = run_yearly(rets_stock, rets_bond, dates, target_w=0.6, fee_rate=0.003)

# Smart Pilot
sp = run_smart_pilot(rets_stock, rets_bond, prices_stock, prices_bond, dates,
                     target_w=0.6, fee_rate=0.003, kf_q=0.001, kp=0.5, kd=0.5, deadband=0.0125)

# 帕雷托前線掃描
pareto = scan_pareto_frontier(rets_stock, rets_bond, prices_stock, prices_bond, dates,
                               target_w=0.6, fee_rate=0.003, kf_q=0.001, kp=0.5, kd=0.5)
```

**策略回傳格式（共用）：**

| 欄位 | 說明 |
|------|------|
| `nav_list` | 每日淨值（從 1.0 開始） |
| `weights` | 每日股票權重（小數） |
| `trade_count` | 總交易次數 |
| `turnover` | 總週轉率 |
| `actions` | 每日交易動作 |
| `rmse` | 追蹤誤差（小數） |
| `cost` | 總交易成本（小數） |
| `metrics` | 年化績效指標 |

### 使用 Optimizer 最佳化參數

```python
from core.optimizer import run_grid_search, find_best_from_grid, run_slsqp, compare_hypervolumes
from core.benchmark import scan_pareto_frontier

# 1. Grid Search（每組 (Kp,Kd,Q) 掃描多個 deadband，計算超體積）
grid_results = run_grid_search(
    rets_stock, rets_bond, prices_stock, prices_bond, dates,
    target_w=0.6, fee_rate=0.003,
    kp_range=[0.1, 0.3, 0.5, 0.7, 1.0],
    kd_range=[0.1, 0.3, 0.5, 0.7, 1.0],
    q_values=[0.0001, 0.001, 0.01],
    deadband_values=[0.01, 0.025, 0.05],
)
best = find_best_from_grid(grid_results)  # 超體積最大
print(f"最佳: Kp={best['kp']}, Kd={best['kd']}, Q={best['q']}, HV={best['hypervolume']:.6f}")

# 2. SLSQP（對每個 deadband 分別最佳化 Kp,Kd,Q）
slsqp_results = run_slsqp(
    rets_stock, rets_bond, prices_stock, prices_bond, dates,
    initial_params={"kp": best["kp"], "kd": best["kd"], "q": best["q"]},
    deadband_values=[0.01, 0.025, 0.05],
)
for r in slsqp_results:
    print(f"  db={r['deadband']:.3f}: RMSE={r['rmse']:.4f}, AnnCost={r['ann_cost']:.4f}")

# 3. 超體積比較
pareto = scan_pareto_frontier(rets_stock, rets_bond, prices_stock, prices_bond, dates)
hv = compare_hypervolumes(pareto)
print(f"Winner: {hv['winner']}")
```

### 使用 LogKalmanFilter 卡爾曼濾波器

```python
import numpy as np
from core.kalman_filter import LogKalmanFilter

kf_stock = LogKalmanFilter(initial_log_price=np.log(400.0), q=0.001, r=0.005)

for price in stock_prices:
    kf_stock.predict()
    kf_stock.update(np.log(price))
    print(f"濾波價格: {kf_stock.filtered_price:.2f}, 速度: {kf_stock.velocity:.6f}")
```

### 使用 PDController PD 控制器

```python
from core.pd_controller import PDController

pd = PDController(kp=0.3, kd=0.1)
error = target_weight - current_weight
u = pd.calculate(error, vel_stock=kf_stock.velocity, vel_bond=kf_bond.velocity)
```

**PD 控制器公式：**
```
P = Kp * error
D = clip(Kd * (vel_stock - vel_bond), -0.15, 0.15)
u = clip(P + D, -0.2, 0.2)
```

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
