# Smart Pilot - 投資組合再平衡系統

基於增量型 PID 控制理論的投資組合再平衡系統。

## 專案結構

```
smart_pilot/
├── app.py                          # Streamlit 主程式
├── core/
│   ├── __init__.py
│   ├── pid_controller.py          # 增量型 PID 控制器 ✅
│   ├── backtest_engine.py         # 回測引擎 ✅
│   ├── portfolio.py               # 投資組合管理
│   └── metrics.py                 # 績效指標計算 ✅
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # yfinance 資料下載 ✅
│   └── cache/                     # 快取資料夾（CSV 格式）
├── validation/
│   ├── __init__.py
│   ├── out_of_sample.py           # 樣本外測試
│   └── monte_carlo.py             # 蒙地卡羅模擬
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
  - [ ] 驗證模組框架
  - [ ] 視覺化模組框架
- [x] Phase 2: 回測引擎完整實作 (`core/backtest_engine.py`)
- [ ] Phase 3: Streamlit UI 開發
- [ ] Phase 4: 進階功能

## 授權

MIT License
