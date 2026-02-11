# Smart Pilot - 投資組合再平衡系統

基於增量型 PID 控制理論的投資組合再平衡系統。

## 專案結構

```
smart_pilot/
├── app.py                          # Streamlit 主程式 ✅
├── core/
│   ├── __init__.py
│   ├── pid_controller.py          # 增量型 PID 控制器 ✅
│   ├── backtest_engine.py         # 回測引擎 ✅
│   ├── benchmark.py               # 對照策略（Threshold/Yearly）✅
│   ├── portfolio.py               # 投資組合管理
│   └── metrics.py                 # 績效指標計算 ✅
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # yfinance 資料下載 ✅
│   └── cache/                     # 快取資料夾（CSV 格式）
├── validation/
│   ├── __init__.py
│   ├── out_of_sample.py           # 樣本外測試 ✅
│   └── monte_carlo.py             # 蒙地卡羅模擬 ✅
├── visualization/
│   ├── __init__.py
│   └── charts.py                  # 圖表生成
├── config/
│   └── parameters.yaml            # 參數配置檔
├── tests/
│   ├── __init__.py
│   ├── test_pid.py                # PID 控制器測試 ✅
│   ├── test_data_loader.py        # 資料載入器測試 ✅
│   └── test_backtest_engine.py    # 回測引擎測試 ✅
├── requirements.txt               # 依賴套件
├── .streamlit/
│   └── config.toml                # Streamlit 設定
└── README.md
```

## 安裝

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

## 使用方式

### 啟動 Streamlit 應用

```bash
streamlit run app.py
```

**Streamlit 應用功能（v2.0）：**

1. **側邊欄參數設定：**
   - 初始資金（預設 1,000,000 USD）
   - 目標股票比例（滑桿 0-100%，預設 60%）
   - 風險偏好（保守/穩健/積極）
   - **PID 參數可直接編輯**（Kp, Ki, Kd）
   - 進階設定（死區閾值、手續費率）

2. **風險偏好對應 PID 參數（新版）：**

   | 風險偏好 | Kp | Ki | Kd | Deadband |
   |----------|-----|------|-----|----------|
   | 保守 | 0.15 | 0.02 | 0.05 | 3% |
   | 穩健 | 0.3 | 0.05 | 0.1 | 2% |
   | 積極 | 0.5 | 0.1 | 0.2 | 1% |

3. **四個分頁顯示結果：**

   - **📈 回測結果**：績效指標、淨值曲線、PID 控制訊號圖、下載 CSV
   - **⚔️ 策略對比**：Smart Pilot vs Threshold 5% vs Yearly Rebalance
   - **🎲 蒙地卡羅模擬**：10,000 次模擬、報酬分布、智慧診斷
   - **🔬 樣本外測試**：過擬合檢測、一致性分析、智慧建議

4. **智慧診斷提示：**
   - 根據分析結果自動提供參數調整建議
   - 識別潛在風險並給出改善方向

### 使用 DataLoader 載入資料

```python
from data.data_loader import DataLoader, quick_load

# 方法 1：完整流程
loader = DataLoader()
raw_data = loader.download_data(["SPY", "TLT"], "2020-01-01", "2023-12-31")
processed = loader.process_data(raw_data)
dates = loader.align_trading_dates(processed)

# 方法 2：便捷方法（下載 + 處理）
data = loader.load_and_process()

# 方法 3：快速函數（使用預設參數：SPY+TLT, 2013-2025）
data = quick_load()

# 計算報酬率
returns = loader.get_returns(data, method="simple")
```

**DataLoader 功能：**

| 方法 | 功能 |
|------|------|
| `download_data(tickers, start_date, end_date)` | 下載收盤價（auto_adjust=True） |
| `process_data(data)` | ffill() + dropna() 處理資料 |
| `align_trading_dates(data)` | 取得對齊的交易日期列表 |
| `load_and_process()` | 便捷方法，一次完成下載+處理 |
| `get_returns(data, method)` | 計算報酬率（simple/log） |
| `clear_cache()` | 清除快取 |

**快取機制：**
- 格式：CSV
- 檔名：`{ticker1}_{ticker2}_{start}_{end}.csv`
- 位置：`data/cache/`

**預設值：**
- 標的：`["SPY", "TLT"]`
- 期間：`2013-01-01` ~ `2025-12-31`

### 使用 BacktestEngine 執行回測

```python
from data.data_loader import DataLoader
from core.backtest_engine import BacktestEngine

# 1. 載入資料
loader = DataLoader()
data = loader.load_and_process(["SPY", "TLT"], "2015-01-01", "2023-12-31")

# 2. 初始化回測引擎
engine = BacktestEngine(
    initial_cash=1_000_000,        # 初始資金
    target_ratio=0.6,              # 目標股票比例 60%
    pid_params={"kp": 1.0, "ki": 0.1, "kd": 2.0},
    deadband=0.01,                 # 死區閾值 1%
    commission_rate=0.001          # 手續費率 0.1%
)

# 3. 執行回測
result = engine.run(data)

# 4. 查看結果
print(f"總報酬率: {result.total_return:.2%}")
print(f"年化報酬率: {result.annualized_return:.2%}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")

# 5. 取得詳細報告
print(engine.get_summary(result))

# 6. 存取歷史記錄
history = result.history
print(history[["nav", "ratio", "trade_flag"]].tail())
```

**BacktestEngine 參數：**

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `initial_cash` | 1,000,000 | 初始資金 |
| `target_ratio` | 0.6 | 目標股票比例（0-1） |
| `pid_params` | {"kp":1.0, "ki":0.1, "kd":2.0} | PID 參數 |
| `deadband` | 0.01 | 死區閾值（調整量低於此值不交易） |
| `commission_rate` | 0.001 | 手續費率 |

**回測結果欄位：**

| 欄位 | 說明 |
|------|------|
| `nav` | 淨值 |
| `ratio` | 實際股票比例 |
| `error` | 誤差（目標 - 實際） |
| `delta_u` | PID 調整量 |
| `trade_flag` | 是否執行交易 |
| `commission` | 當日手續費 |

### 使用 MetricsCalculator 計算績效指標

```python
from core.metrics import MetricsCalculator
from core.backtest_engine import BacktestEngine

# 執行回測
engine = BacktestEngine()
result = engine.run(data)

# 初始化績效計算器
calculator = MetricsCalculator(risk_free_rate=0.02)

# 計算所有指標
metrics = calculator.calculate_all_metrics(result.history)

# 查看指標
print(f"總報酬率: {metrics['total_return']:.2%}")
print(f"年化報酬率: {metrics['annual_return']:.2%}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"索提諾比率: {metrics['sortino_ratio']:.2f}")
print(f"勝率: {metrics['win_rate']:.1%}")

# 格式化報告
print(calculator.format_report(metrics))
```

**MetricsCalculator 計算的指標：**

| 類別 | 指標 | 說明 |
|------|------|------|
| **基礎** | `total_return` | 總報酬率 |
| | `annual_return` | 年化報酬率 |
| | `total_trades` | 總交易次數 |
| | `total_commission` | 總手續費 |
| **風險** | `max_drawdown` | 最大回撤 (MDD) |
| | `volatility` | 年化波動率 |
| | `downside_volatility` | 下行波動率 |
| | `var_95` / `var_99` | 風險值 (VaR) |
| **風險調整** | `sharpe_ratio` | 夏普比率 |
| | `sortino_ratio` | 索提諾比率 |
| | `calmar_ratio` | 卡瑪比率 |
| **交易分析** | `win_rate` | 勝率 |
| | `average_gain` | 平均獲利 |
| | `average_loss` | 平均虧損 |

### 使用 OutOfSampleValidator 執行樣本外驗證

```python
from data.data_loader import DataLoader
from core.backtest_engine import BacktestEngine
from validation.out_of_sample import OutOfSampleValidator, quick_validate

# 1. 載入資料
loader = DataLoader()
data = loader.load_and_process(["SPY", "TLT"], "2015-01-01", "2025-12-31")

# 2. 建立回測引擎
engine = BacktestEngine(
    target_ratio=0.6,
    pid_params={"kp": 1.0, "ki": 0.1, "kd": 2.0}
)

# 3. 建立驗證器並執行驗證
validator = OutOfSampleValidator(overfitting_threshold=0.5)
result = validator.validate(engine, data, split_date="2024-01-01")

# 4. 查看結果
print(f"是否過擬合: {result.is_overfitted}")
print(f"樣本內報酬: {result.in_sample_metrics['total_return']:.2%}")
print(f"樣本外報酬: {result.out_of_sample_metrics['total_return']:.2%}")

# 5. 取得完整報告
print(validator.get_report(result))

# 方法 2：使用便捷函數
result = quick_validate(data, split_date="2024-01-01", target_ratio=0.6)
```

**OutOfSampleValidator 方法：**

| 方法 | 功能 |
|------|------|
| `split_data(data, split_date)` | 分割資料為樣本內/樣本外 |
| `validate(engine, data, split_date)` | 執行完整驗證流程 |
| `compare_performance(is_metrics, oos_metrics)` | 比較兩個時期的績效 |
| `get_report(result)` | 生成格式化報告 |

**ValidationResult 欄位：**

| 欄位 | 說明 |
|------|------|
| `in_sample_result` | 樣本內回測結果 |
| `out_of_sample_result` | 樣本外回測結果 |
| `in_sample_metrics` | 樣本內績效指標 |
| `out_of_sample_metrics` | 樣本外績效指標 |
| `comparison` | 績效比較報告 |
| `is_overfitted` | 是否過擬合 |
| `split_date` | 分割日期 |

**過擬合判斷標準：**
- 夏普比率比率 < 0.5（樣本外/樣本內）
- 報酬率比率 < 0.5（樣本外/樣本內）
- 一致性分數 < 0.3

### 使用 MonteCarloSimulator 執行蒙地卡羅模擬

```python
from core.backtest_engine import BacktestEngine
from data.data_loader import DataLoader
from validation.monte_carlo import MonteCarloSimulator, quick_simulate

# 1. 載入資料並執行回測
loader = DataLoader()
data = loader.load_and_process()
engine = BacktestEngine(target_ratio=0.6)
backtest_result = engine.run(data)

# 2. 建立模擬器（設定隨機種子可重現結果）
simulator = MonteCarloSimulator(random_seed=42)

# 3. 執行蒙地卡羅模擬
result = simulator.simulate(backtest_result, n_simulations=10000)

# 4. 查看統計結果
stats = result.statistics
print(f"平均報酬率: {stats['mean_return']:.2%}")
print(f"標準差: {stats['std_return']:.2%}")
print(f"賺錢機率: {stats['prob_profit']:.1%}")
print(f"破產風險: {stats['ruin_risk']:.1%}")
print(f"95% CI: [{stats['ci_95_lower']:.2%}, {stats['ci_95_upper']:.2%}]")

# 5. 取得完整報告
print(simulator.get_report(result))

# 方法 2：使用便捷函數
result = quick_simulate(backtest_result, n_simulations=10000, random_seed=42)
```

**MonteCarloSimulator 方法：**

| 方法 | 功能 |
|------|------|
| `simulate(backtest_result, n_simulations)` | 執行 Bootstrap 蒙地卡羅模擬 |
| `analyze_results(simulations)` | 分析模擬結果，計算統計指標 |
| `get_report(result)` | 生成格式化報告 |
| `reset_seed(new_seed)` | 重設隨機種子 |

**統計指標：**

| 類別 | 指標 | 說明 |
|------|------|------|
| **基本統計** | `mean_return` | 平均報酬率 |
| | `std_return` | 標準差 |
| | `median_return` | 中位數報酬率 |
| **機率** | `prob_profit` | 賺錢機率（報酬 > 0） |
| | `ruin_risk` | 破產風險（虧損 > 50%） |
| **信賴區間** | `ci_95_lower/upper` | 95% 信賴區間（2.5%, 97.5%） |
| | `ci_99_lower/upper` | 99% 信賴區間（0.5%, 99.5%） |
| **極端情況** | `best_case` | 最佳情況報酬 |
| | `worst_case` | 最差情況報酬 |

### 使用對照策略（Benchmark）

```python
from core.benchmark import run_threshold_rebalance, run_yearly_rebalance, calculate_tracking_error
from data.data_loader import DataLoader

# 載入資料
loader = DataLoader()
data = loader.load_and_process()

# 執行門檻再平衡策略（偏離 5% 才交易）
threshold_history = run_threshold_rebalance(
    data=data,
    initial_cash=1_000_000,
    target_ratio=0.6,
    threshold=0.05,
    commission_rate=0.001
)

# 執行年度再平衡策略（每年初再平衡）
yearly_history = run_yearly_rebalance(
    data=data,
    initial_cash=1_000_000,
    target_ratio=0.6,
    commission_rate=0.001
)

# 計算追蹤誤差
rmse = calculate_tracking_error(threshold_history["ratio"], target_ratio=0.6)
print(f"追蹤誤差 RMSE: {rmse:.2%}")
```

**對照策略：**

| 策略 | 說明 |
|------|------|
| `run_threshold_rebalance()` | 門檻再平衡：偏離超過門檻才交易 |
| `run_yearly_rebalance()` | 年度再平衡：每年第一個交易日再平衡 |
| `calculate_tracking_error()` | 計算追蹤誤差 RMSE |

### 使用 PID 控制器

```python
from core.pid_controller import IncrementalPID

# 初始化 PID 控制器
pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

# 計算調整量
target_weight = 0.30  # 目標權重 30%
current_weight = 0.25  # 當前權重 25%
error = target_weight - current_weight

adjustment = pid.calculate(error)
print(f"建議調整量: {adjustment:.4f}")

# 重置歷史誤差
pid.reset()

# 動態調整參數
pid.set_parameters(kp=0.3, ki=0.15)

# 查看狀態
print(pid.get_state())
```

### 執行測試

```bash
# 執行所有測試
pytest tests/ -v

# 只執行 PID 控制器測試
pytest tests/test_pid.py -v

# 只執行 DataLoader 測試
pytest tests/test_data_loader.py -v

# 只執行 BacktestEngine 測試
pytest tests/test_backtest_engine.py -v

# 執行測試並顯示覆蓋率
pytest tests/ -v --cov=core --cov=data
```

**測試涵蓋範圍：**

| 測試檔案 | 測試目標 |
|----------|----------|
| `test_pid.py` | IncrementalPID 類別 |
| `test_data_loader.py` | DataLoader 類別 |
| `test_backtest_engine.py` | BacktestEngine 類別 |

## 核心概念

### 增量型 PID 控制器

公式：
```
Δu(k) = Kp[e(k) - e(k-1)] + Ki*e(k) + Kd[e(k) - 2e(k-1) + e(k-2)]
```

其中：
- `e(k)`: 當前誤差（目標權重 - 實際權重）
- `e(k-1)`: 前一次誤差
- `e(k-2)`: 前兩次誤差
- `Kp`: 比例增益
- `Ki`: 積分增益
- `Kd`: 微分增益

### 參數建議

| 參數 | 建議範圍 | 說明 |
|------|----------|------|
| Kp | 0.1 ~ 1.0 | 響應速度，較大值響應更快 |
| Ki | 0.01 ~ 0.5 | 消除穩態誤差 |
| Kd | 0.01 ~ 0.2 | 抑制振盪，增加穩定性 |

## 開發進度

- [x] Phase 1: 專案結構建立
  - [x] PID 控制器實作 (`core/pid_controller.py`)
  - [x] 資料載入模組 (`data/data_loader.py`)
  - [x] 績效指標計算 (`core/metrics.py`)
  - [x] 單元測試 (`tests/test_pid.py`, `tests/test_data_loader.py`, `tests/test_backtest_engine.py`)
  - [ ] 投資組合管理框架
  - [x] 樣本外驗證模組 (`validation/out_of_sample.py`)
  - [x] 蒙地卡羅模擬模組 (`validation/monte_carlo.py`)
  - [x] 對照策略模組 (`core/benchmark.py`)
  - [ ] 視覺化模組框架
- [x] Phase 2: 回測引擎完整實作 (`core/backtest_engine.py`)
- [x] Phase 3: Streamlit UI 開發 (`app.py`)
  - [x] PID 參數可編輯
  - [x] 策略對比功能
  - [x] 蒙地卡羅模擬整合
  - [x] 樣本外測試整合
  - [x] 智慧診斷提示
- [ ] Phase 4: 進階功能

## 授權

MIT License
