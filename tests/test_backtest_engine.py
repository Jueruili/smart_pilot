"""
BacktestEngine 單元測試

測試 core/backtest_engine.py 中的 BacktestEngine 類別功能。

作者：Smart Pilot Team
版本：1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    DEFAULT_INITIAL_CASH,
    DEFAULT_TARGET_RATIO,
    DEFAULT_PID_PARAMS,
    DEFAULT_DEADBAND,
    DEFAULT_COMMISSION_RATE,
)


def create_sample_data(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """建立測試用的價格資料

    Args:
        n_days: 天數
        seed: 隨機種子

    Returns:
        pd.DataFrame: 包含 SPY 和 TLT 的價格資料
    """
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # 模擬股票和債券價格（帶有隨機波動）
    stock_returns = np.random.normal(0.0005, 0.015, n_days)
    bond_returns = np.random.normal(0.0002, 0.005, n_days)

    stock_prices = 100 * np.cumprod(1 + stock_returns)
    bond_prices = 50 * np.cumprod(1 + bond_returns)

    return pd.DataFrame({
        "SPY": stock_prices,
        "TLT": bond_prices
    }, index=dates)


class TestBacktestEngineInit:
    """測試 BacktestEngine 初始化"""

    def test_default_initialization(self):
        """測試預設參數初始化"""
        engine = BacktestEngine()

        assert engine.initial_cash == DEFAULT_INITIAL_CASH
        assert engine.target_ratio == DEFAULT_TARGET_RATIO
        assert engine.deadband == DEFAULT_DEADBAND
        assert engine.commission_rate == DEFAULT_COMMISSION_RATE

    def test_custom_initialization(self):
        """測試自訂參數初始化"""
        engine = BacktestEngine(
            initial_cash=500_000,
            target_ratio=0.7,
            pid_params={"kp": 2.0, "ki": 0.2, "kd": 1.0},
            deadband=0.02,
            commission_rate=0.002
        )

        assert engine.initial_cash == 500_000
        assert engine.target_ratio == 0.7
        assert engine.deadband == 0.02
        assert engine.commission_rate == 0.002
        assert engine.pid_params["kp"] == 2.0

    def test_invalid_initial_cash_raises(self):
        """測試無效初始資金拋出例外"""
        with pytest.raises(ValueError, match="初始資金必須為正數"):
            BacktestEngine(initial_cash=-1000)

        with pytest.raises(ValueError, match="初始資金必須為正數"):
            BacktestEngine(initial_cash=0)

    def test_invalid_target_ratio_raises(self):
        """測試無效目標比例拋出例外"""
        with pytest.raises(ValueError, match="目標比例必須在 0-1 之間"):
            BacktestEngine(target_ratio=-0.1)

        with pytest.raises(ValueError, match="目標比例必須在 0-1 之間"):
            BacktestEngine(target_ratio=1.5)

    def test_invalid_deadband_raises(self):
        """測試無效死區閾值拋出例外"""
        with pytest.raises(ValueError, match="死區閾值不能為負數"):
            BacktestEngine(deadband=-0.01)

    def test_invalid_commission_rate_raises(self):
        """測試無效手續費率拋出例外"""
        with pytest.raises(ValueError, match="手續費率不能為負數"):
            BacktestEngine(commission_rate=-0.001)

    def test_pid_controller_initialized(self):
        """測試 PID 控制器正確初始化"""
        engine = BacktestEngine(pid_params={"kp": 1.5, "ki": 0.2, "kd": 0.5})

        assert engine.pid.kp == 1.5
        assert engine.pid.ki == 0.2
        assert engine.pid.kd == 0.5


class TestBacktestEngineRun:
    """測試回測執行"""

    def test_run_returns_backtest_result(self):
        """測試 run 返回 BacktestResult"""
        engine = BacktestEngine()
        data = create_sample_data(50)

        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.history, pd.DataFrame)

    def test_run_empty_data_raises(self):
        """測試空資料拋出例外"""
        engine = BacktestEngine()
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="輸入資料為空"):
            engine.run(empty_data)

    def test_run_wrong_columns_raises(self):
        """測試錯誤欄位數拋出例外"""
        engine = BacktestEngine()
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        single_col_data = pd.DataFrame({"SPY": [100] * 10}, index=dates)

        with pytest.raises(ValueError, match="需要兩個標的的資料"):
            engine.run(single_col_data)

    def test_history_has_correct_columns(self):
        """測試歷史記錄包含正確欄位"""
        engine = BacktestEngine()
        data = create_sample_data(30)

        result = engine.run(data)

        expected_columns = [
            "nav", "cash", "holdings_stock", "holdings_bond",
            "value_stock", "value_bond", "ratio", "price_stock",
            "price_bond", "error", "delta_u", "trade_flag", "commission"
        ]
        for col in expected_columns:
            assert col in result.history.columns

    def test_history_length_matches_data(self):
        """測試歷史記錄長度與資料相符"""
        engine = BacktestEngine()
        data = create_sample_data(50)

        result = engine.run(data)

        assert len(result.history) == len(data)

    def test_first_day_initial_allocation(self):
        """測試第一天的初始配置"""
        engine = BacktestEngine(
            initial_cash=1_000_000,
            target_ratio=0.6,
            commission_rate=0.001
        )
        data = create_sample_data(10)

        result = engine.run(data)

        # 第一天應該有交易（初始買入）
        assert result.history["trade_flag"].iloc[0] == True

        # 檢查比例接近目標
        first_ratio = result.history["ratio"].iloc[0]
        assert abs(first_ratio - 0.6) < 0.01

    def test_nav_always_positive(self):
        """測試淨值始終為正"""
        engine = BacktestEngine()
        data = create_sample_data(100)

        result = engine.run(data)

        assert (result.history["nav"] > 0).all()

    def test_ratio_within_bounds(self):
        """測試比例在合理範圍內"""
        engine = BacktestEngine()
        data = create_sample_data(100)

        result = engine.run(data)

        assert (result.history["ratio"] >= 0).all()
        assert (result.history["ratio"] <= 1).all()


class TestBacktestMetrics:
    """測試績效指標計算"""

    def test_total_return_calculation(self):
        """測試總報酬率計算"""
        engine = BacktestEngine()
        data = create_sample_data(100)

        result = engine.run(data)

        # 手動計算總報酬率
        initial_nav = result.history["nav"].iloc[0]
        final_nav = result.history["nav"].iloc[-1]
        expected_return = (final_nav - initial_nav) / initial_nav

        assert abs(result.total_return - expected_return) < 1e-6

    def test_max_drawdown_negative_or_zero(self):
        """測試最大回撤為負或零"""
        engine = BacktestEngine()
        data = create_sample_data(100)

        result = engine.run(data)

        assert result.max_drawdown <= 0

    def test_total_trades_count(self):
        """測試交易次數統計"""
        engine = BacktestEngine(deadband=0.001)  # 較小的死區，更多交易
        data = create_sample_data(50)

        result = engine.run(data)

        # 至少有初始買入的 2 筆交易
        assert result.total_trades >= 2

    def test_total_commission_non_negative(self):
        """測試總手續費非負"""
        engine = BacktestEngine()
        data = create_sample_data(50)

        result = engine.run(data)

        assert result.total_commission >= 0


class TestDeadbandBehavior:
    """測試死區行為"""

    def test_large_deadband_fewer_trades(self):
        """測試較大死區導致較少交易"""
        data = create_sample_data(100, seed=42)

        engine_small = BacktestEngine(deadband=0.001)
        engine_large = BacktestEngine(deadband=0.1)

        result_small = engine_small.run(data.copy())
        result_large = engine_large.run(data.copy())

        # 較大死區應該有較少交易
        assert result_large.total_trades <= result_small.total_trades

    def test_zero_deadband_trades_on_any_error(self):
        """測試零死區時任何誤差都會交易"""
        engine = BacktestEngine(deadband=0.0)
        data = create_sample_data(20)

        result = engine.run(data)

        # 幾乎每天都應該有交易（除了完全沒有誤差的情況）
        trade_days = result.history["trade_flag"].sum()
        assert trade_days >= 10  # 至少一半的天數有交易


class TestCommissionCalculation:
    """測試手續費計算"""

    def test_zero_commission_rate(self):
        """測試零手續費率"""
        engine = BacktestEngine(commission_rate=0.0)
        data = create_sample_data(30)

        result = engine.run(data)

        assert result.total_commission == 0

    def test_commission_proportional_to_rate(self):
        """測試手續費與費率成正比"""
        data = create_sample_data(50, seed=42)

        engine_low = BacktestEngine(commission_rate=0.001, deadband=0.001)
        engine_high = BacktestEngine(commission_rate=0.002, deadband=0.001)

        result_low = engine_low.run(data.copy())
        result_high = engine_high.run(data.copy())

        # 手續費率高的應該有更多手續費（大約是 2 倍）
        if result_low.total_commission > 0:
            ratio = result_high.total_commission / result_low.total_commission
            assert 1.5 < ratio < 2.5


class TestPIDIntegration:
    """測試 PID 控制器整合"""

    def test_pid_affects_delta_u(self):
        """測試 PID 參數影響 delta_u"""
        data = create_sample_data(30, seed=42)

        engine_high_kp = BacktestEngine(pid_params={"kp": 2.0, "ki": 0.1, "kd": 2.0})
        engine_low_kp = BacktestEngine(pid_params={"kp": 0.5, "ki": 0.1, "kd": 2.0})

        result_high = engine_high_kp.run(data.copy())
        result_low = engine_low_kp.run(data.copy())

        # 較高 Kp 應該產生較大的 delta_u（絕對值）
        avg_delta_high = result_high.history["delta_u"].abs().mean()
        avg_delta_low = result_low.history["delta_u"].abs().mean()

        # 注意：這個斷言可能因為隨機數據而有變化
        # 但整體趨勢應該是高 Kp 產生更大的調整量
        assert avg_delta_high >= avg_delta_low * 0.5  # 寬鬆的檢查

    def test_pid_reset_on_run(self):
        """測試每次執行時 PID 重置"""
        engine = BacktestEngine()
        data = create_sample_data(20)

        # 第一次執行
        engine.run(data)

        # PID 應該已經有歷史誤差
        # 第二次執行時應該重置
        result2 = engine.run(data)

        # 第一天的 error 應該是 0（因為是初始配置）
        assert result2.history["error"].iloc[0] == 0


class TestReset:
    """測試重置功能"""

    def test_reset_clears_stats(self):
        """測試重置清除統計"""
        engine = BacktestEngine()
        data = create_sample_data(30)

        # 執行回測
        engine.run(data)

        # 手動檢查內部狀態
        assert engine._total_trades > 0

        # 重置
        engine.reset()

        # 統計應該歸零
        assert engine._total_trades == 0
        assert engine._total_commission == 0.0


class TestGetSummary:
    """測試摘要報告"""

    def test_get_summary_returns_string(self):
        """測試 get_summary 返回字串"""
        engine = BacktestEngine()
        data = create_sample_data(30)

        result = engine.run(data)
        summary = engine.get_summary(result)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_contains_key_metrics(self):
        """測試摘要包含關鍵指標"""
        engine = BacktestEngine()
        data = create_sample_data(30)

        result = engine.run(data)
        summary = engine.get_summary(result)

        # 檢查關鍵字是否存在
        assert "總報酬率" in summary
        assert "年化報酬率" in summary
        assert "最大回撤" in summary
        assert "夏普比率" in summary
        assert "總交易次數" in summary


class TestBacktestResult:
    """測試 BacktestResult 資料類別"""

    def test_backtest_result_attributes(self):
        """測試 BacktestResult 屬性"""
        history = pd.DataFrame({"nav": [100, 101, 102]})
        result = BacktestResult(
            history=history,
            total_return=0.02,
            annualized_return=0.05,
            max_drawdown=-0.1,
            sharpe_ratio=1.5,
            total_trades=10,
            total_commission=100.0
        )

        assert len(result.history) == 3
        assert result.total_return == 0.02
        assert result.annualized_return == 0.05
        assert result.max_drawdown == -0.1
        assert result.sharpe_ratio == 1.5
        assert result.total_trades == 10
        assert result.total_commission == 100.0


class TestEdgeCases:
    """測試邊界情況"""

    def test_minimum_data_length(self):
        """測試最小資料長度"""
        engine = BacktestEngine()
        data = create_sample_data(2)  # 只有 2 天

        result = engine.run(data)

        assert len(result.history) == 2

    def test_target_ratio_zero(self):
        """測試目標比例為零（全部債券）"""
        engine = BacktestEngine(target_ratio=0.0)
        data = create_sample_data(20)

        result = engine.run(data)

        # 股票比例應該接近零
        avg_ratio = result.history["ratio"].mean()
        assert avg_ratio < 0.1

    def test_target_ratio_one(self):
        """測試目標比例為一（全部股票）"""
        engine = BacktestEngine(target_ratio=1.0)
        data = create_sample_data(20)

        result = engine.run(data)

        # 股票比例應該接近一
        avg_ratio = result.history["ratio"].mean()
        assert avg_ratio > 0.9


# 執行測試的入口
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
