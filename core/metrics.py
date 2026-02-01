"""
績效指標計算模組

計算投資組合的各種績效指標，包括：
- 基礎指標（總報酬、年化報酬、交易統計）
- 風險指標（波動率、最大回撤、下行波動率、VaR）
- 風險調整報酬（Sharpe Ratio、Sortino Ratio、Calmar Ratio）
- 交易分析（勝率、平均獲利/虧損）

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np


# 預設參數
DEFAULT_RISK_FREE_RATE: float = 0.02  # 無風險利率 2%
DEFAULT_TRADING_DAYS: int = 252       # 一年交易日數


@dataclass
class MetricsResult:
    """績效指標結果

    Attributes:
        total_return: 總報酬率
        annual_return: 年化報酬率
        total_trades: 總交易次數
        total_commission: 總手續費
        max_drawdown: 最大回撤
        volatility: 年化波動率
        downside_volatility: 下行波動率
        sharpe_ratio: 夏普比率
        sortino_ratio: 索提諾比率
        calmar_ratio: 卡瑪比率
        win_rate: 勝率
        average_gain: 平均獲利
        average_loss: 平均虧損
        var_95: 95% VaR
        var_99: 99% VaR
        trading_days: 交易天數
    """
    total_return: float
    annual_return: float
    total_trades: int
    total_commission: float
    max_drawdown: float
    volatility: float
    downside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    average_gain: float
    average_loss: float
    var_95: float
    var_99: float
    trading_days: int


class MetricsCalculator:
    """績效指標計算器

    計算投資組合回測結果的各種績效指標。

    Attributes:
        risk_free_rate: 無風險利率（年化）
        trading_days: 一年的交易日數

    Example:
        >>> from core.backtest_engine import BacktestEngine
        >>> result = engine.run(data)
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.calculate_all_metrics(result.history)
        >>> print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        trading_days: int = DEFAULT_TRADING_DAYS
    ) -> None:
        """初始化績效指標計算器

        Args:
            risk_free_rate: 無風險利率（年化），預設為 0.02 (2%)
            trading_days: 一年的交易日數，預設為 252

        Raises:
            ValueError: 當交易日數為非正數時拋出
        """
        if trading_days <= 0:
            raise ValueError(f"交易日數必須為正數，收到：{trading_days}")

        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_all_metrics(
        self,
        result_df: pd.DataFrame
    ) -> dict[str, Union[float, int]]:
        """計算所有績效指標

        Args:
            result_df: 回測結果 DataFrame，必須包含以下欄位：
                - nav: 淨值序列
                - trade_flag: 交易標記
                - commission: 手續費

        Returns:
            dict: 包含所有績效指標的字典

        Raises:
            ValueError: 當資料格式不正確時拋出

        Example:
            >>> metrics = calculator.calculate_all_metrics(result.history)
            >>> print(metrics)
        """
        # 驗證資料
        self._validate_dataframe(result_df)

        # 取得淨值序列
        nav_series = result_df["nav"]

        # 計算報酬率序列
        returns = self._calculate_returns(nav_series)

        # 計算回撤序列
        drawdown_series = self._calculate_drawdown(nav_series)

        # === 基礎指標 ===
        total_return = self._calculate_total_return(nav_series)
        annual_return = self._calculate_annual_return(returns)

        # 交易統計
        total_trades = int(result_df["trade_flag"].sum())
        total_commission = float(result_df["commission"].sum())

        # === 風險指標 ===
        max_drawdown = float(drawdown_series.min())
        volatility = self._calculate_volatility(returns)
        downside_volatility = self._calculate_downside_volatility(returns)

        # === 風險調整報酬 ===
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)

        # === 交易分析 ===
        win_rate, average_gain, average_loss = self._calculate_trade_stats(returns)

        # === VaR ===
        var_95 = self._calculate_var(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)

        return {
            # 基礎指標
            "total_return": total_return,
            "annual_return": annual_return,
            "total_trades": total_trades,
            "total_commission": total_commission,

            # 風險指標
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "downside_volatility": downside_volatility,

            # 風險調整報酬
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,

            # 交易分析
            "win_rate": win_rate,
            "average_gain": average_gain,
            "average_loss": average_loss,

            # VaR
            "var_95": var_95,
            "var_99": var_99,

            # 其他
            "trading_days": len(result_df),
        }

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """驗證 DataFrame 格式

        Args:
            df: 要驗證的 DataFrame

        Raises:
            ValueError: 當格式不正確時拋出
        """
        if df.empty:
            raise ValueError("輸入資料為空")

        required_columns = ["nav", "trade_flag", "commission"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"缺少必要欄位: {missing_columns}")

    def _calculate_returns(self, nav_series: pd.Series) -> pd.Series:
        """計算日報酬率序列

        公式：r_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}

        Args:
            nav_series: 淨值序列

        Returns:
            pd.Series: 日報酬率序列（已移除第一個 NaN）
        """
        returns = nav_series.pct_change().dropna()
        return returns

    def _calculate_drawdown(self, nav_series: pd.Series) -> pd.Series:
        """計算回撤序列

        回撤公式：DD_t = (NAV_t - max(NAV_0, ..., NAV_t)) / max(NAV_0, ..., NAV_t)

        Args:
            nav_series: 淨值序列

        Returns:
            pd.Series: 回撤序列（負值表示回撤）
        """
        # 計算歷史高點（累積最大值）
        running_max = nav_series.cummax()

        # 計算回撤
        # 避免除以零：當 running_max 為 0 時，回撤為 0
        drawdown = np.where(
            running_max != 0,
            (nav_series - running_max) / running_max,
            0.0
        )

        return pd.Series(drawdown, index=nav_series.index)

    def _annualize(
        self,
        value: float,
        periods: int,
        is_volatility: bool = False
    ) -> float:
        """年化計算

        報酬率年化公式：r_annual = (1 + r_total)^(trading_days / periods) - 1
        波動率年化公式：σ_annual = σ_daily * sqrt(trading_days)

        Args:
            value: 要年化的值
            periods: 實際期數
            is_volatility: 是否為波動率（使用不同的年化公式）

        Returns:
            float: 年化後的值
        """
        if periods == 0:
            return 0.0

        if is_volatility:
            # 波動率年化：乘以 sqrt(交易日數)
            return value * np.sqrt(self.trading_days)
        else:
            # 報酬率年化：複利計算
            return (1 + value) ** (self.trading_days / periods) - 1

    def _calculate_total_return(self, nav_series: pd.Series) -> float:
        """計算總報酬率

        公式：r_total = (NAV_end - NAV_start) / NAV_start

        Args:
            nav_series: 淨值序列

        Returns:
            float: 總報酬率
        """
        if len(nav_series) < 2:
            return 0.0

        initial_nav = nav_series.iloc[0]
        final_nav = nav_series.iloc[-1]

        # 避免除以零
        if initial_nav == 0:
            return 0.0

        return (final_nav - initial_nav) / initial_nav

    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """計算年化報酬率

        公式：r_annual = (1 + r_total)^(252 / n_days) - 1

        其中 r_total = ∏(1 + r_t) - 1

        Args:
            returns: 日報酬率序列

        Returns:
            float: 年化報酬率
        """
        if len(returns) == 0:
            return 0.0

        # 計算總報酬率（複利）
        total_return = (1 + returns).prod() - 1

        # 年化
        return self._annualize(total_return, len(returns), is_volatility=False)

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """計算年化波動率

        公式：σ_annual = σ_daily * sqrt(252)

        其中 σ_daily = sqrt(1/(n-1) * Σ(r_t - r̄)²)

        Args:
            returns: 日報酬率序列

        Returns:
            float: 年化波動率
        """
        if len(returns) < 2:
            return 0.0

        daily_vol = returns.std()
        return self._annualize(daily_vol, len(returns), is_volatility=True)

    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """計算下行波動率

        只考慮負報酬的波動率，更能反映下行風險。

        公式：σ_down = sqrt(1/(n-1) * Σ(min(r_t, 0))²) * sqrt(252)

        Args:
            returns: 日報酬率序列

        Returns:
            float: 年化下行波動率
        """
        # 只取負報酬
        negative_returns = returns[returns < 0]

        if len(negative_returns) < 2:
            return 0.0

        daily_downside_vol = negative_returns.std()
        return self._annualize(daily_downside_vol, len(returns), is_volatility=True)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """計算夏普比率 (Sharpe Ratio)

        衡量每單位總風險所獲得的超額報酬。

        公式：Sharpe = (r_annual - r_f) / σ_annual

        其中：
        - r_annual: 年化報酬率
        - r_f: 無風險利率（年化）
        - σ_annual: 年化波動率

        Args:
            returns: 日報酬率序列

        Returns:
            float: 夏普比率
        """
        annual_return = self._calculate_annual_return(returns)
        volatility = self._calculate_volatility(returns)

        # 避免除以零
        if volatility == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / volatility

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """計算索提諾比率 (Sortino Ratio)

        類似夏普比率，但只考慮下行風險，對投資者更有意義。

        公式：Sortino = (r_annual - r_f) / σ_downside

        其中：
        - r_annual: 年化報酬率
        - r_f: 無風險利率（年化）
        - σ_downside: 年化下行波動率

        Args:
            returns: 日報酬率序列

        Returns:
            float: 索提諾比率
        """
        annual_return = self._calculate_annual_return(returns)
        downside_vol = self._calculate_downside_volatility(returns)

        # 避免除以零
        if downside_vol == 0:
            # 如果沒有下行波動，且有正報酬，返回較大值
            if annual_return > self.risk_free_rate:
                return float('inf')
            return 0.0

        return (annual_return - self.risk_free_rate) / downside_vol

    def _calculate_calmar_ratio(
        self,
        annual_return: float,
        max_drawdown: float
    ) -> float:
        """計算卡瑪比率 (Calmar Ratio)

        衡量報酬與最大回撤的關係，反映風險調整後的績效。

        公式：Calmar = r_annual / |MDD|

        其中：
        - r_annual: 年化報酬率
        - MDD: 最大回撤（負值）

        Args:
            annual_return: 年化報酬率
            max_drawdown: 最大回撤（負值）

        Returns:
            float: 卡瑪比率
        """
        # 避免除以零
        if max_drawdown == 0:
            if annual_return > 0:
                return float('inf')
            return 0.0

        # 最大回撤是負值，取絕對值
        return annual_return / abs(max_drawdown)

    def _calculate_trade_stats(
        self,
        returns: pd.Series
    ) -> tuple[float, float, float]:
        """計算交易統計

        Args:
            returns: 日報酬率序列

        Returns:
            tuple: (勝率, 平均獲利, 平均虧損)
                - 勝率: 正報酬天數 / 總天數
                - 平均獲利: 正報酬的平均值
                - 平均虧損: 負報酬的平均值（負值）
        """
        if len(returns) == 0:
            return 0.0, 0.0, 0.0

        # 正報酬和負報酬
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        # 勝率
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

        # 平均獲利
        average_gain = positive_returns.mean() if len(positive_returns) > 0 else 0.0

        # 平均虧損
        average_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0

        return win_rate, average_gain, average_loss

    def _calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float
    ) -> float:
        """計算風險值 (Value at Risk, VaR)

        在給定信心水準下，可能的最大損失。

        公式：VaR_α = Percentile(returns, 1-α)

        例如：95% VaR 是報酬率分布的第 5 百分位數

        Args:
            returns: 日報酬率序列
            confidence_level: 信心水準（例如 0.95 表示 95%）

        Returns:
            float: VaR 值（通常為負值，表示損失）
        """
        if len(returns) == 0:
            return 0.0

        return float(returns.quantile(1 - confidence_level))

    def get_metrics_as_dataclass(
        self,
        result_df: pd.DataFrame
    ) -> MetricsResult:
        """取得績效指標（資料類別格式）

        Args:
            result_df: 回測結果 DataFrame

        Returns:
            MetricsResult: 績效指標資料類別
        """
        metrics = self.calculate_all_metrics(result_df)
        return MetricsResult(**metrics)

    def format_report(
        self,
        metrics: dict[str, Union[float, int]]
    ) -> str:
        """格式化績效指標報告

        Args:
            metrics: 績效指標字典

        Returns:
            str: 格式化的報告文字
        """
        # 處理 inf 值
        def safe_format(value: float, format_spec: str) -> str:
            if value == float('inf'):
                return "∞"
            elif value == float('-inf'):
                return "-∞"
            return format(value, format_spec)

        report = f"""
=====================================
        績效指標報告
=====================================

【基礎指標】
  總報酬率:          {metrics['total_return']:>12.2%}
  年化報酬率:        {metrics['annual_return']:>12.2%}
  總交易次數:        {metrics['total_trades']:>12,d}
  總手續費:          ${metrics['total_commission']:>11,.2f}

【風險指標】
  最大回撤 (MDD):    {metrics['max_drawdown']:>12.2%}
  年化波動率:        {metrics['volatility']:>12.2%}
  下行波動率:        {metrics['downside_volatility']:>12.2%}
  95% VaR:          {metrics['var_95']:>12.2%}
  99% VaR:          {metrics['var_99']:>12.2%}

【風險調整報酬】
  夏普比率:          {safe_format(metrics['sharpe_ratio'], '>12.2f')}
  索提諾比率:        {safe_format(metrics['sortino_ratio'], '>12.2f')}
  卡瑪比率:          {safe_format(metrics['calmar_ratio'], '>12.2f')}

【交易分析】
  勝率:              {metrics['win_rate']:>12.1%}
  平均獲利:          {metrics['average_gain']:>12.4%}
  平均虧損:          {metrics['average_loss']:>12.4%}

【其他資訊】
  交易天數:          {metrics['trading_days']:>12,d}

=====================================
"""
        return report


# === 便捷函數（保持向後相容） ===

def calculate_returns(prices: pd.Series) -> pd.Series:
    """計算日報酬率（便捷函數）"""
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """計算累積報酬率（便捷函數）"""
    return (1 + returns).cumprod() - 1


def calculate_total_return(returns: pd.Series) -> float:
    """計算總報酬率（便捷函數）"""
    return float((1 + returns).prod() - 1)


def calculate_annualized_return(
    returns: pd.Series,
    trading_days: int = 252
) -> float:
    """計算年化報酬率（便捷函數）"""
    if len(returns) == 0:
        return 0.0
    total_return = calculate_total_return(returns)
    return (1 + total_return) ** (trading_days / len(returns)) - 1


def calculate_volatility(
    returns: pd.Series,
    trading_days: int = 252
) -> float:
    """計算年化波動率（便捷函數）"""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days))


def calculate_max_drawdown(returns: pd.Series) -> float:
    """計算最大回撤（便捷函數）"""
    if len(returns) == 0:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> float:
    """計算夏普比率（便捷函數）"""
    calculator = MetricsCalculator(risk_free_rate, trading_days)
    return calculator._calculate_sharpe_ratio(returns)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> float:
    """計算索提諾比率（便捷函數）"""
    calculator = MetricsCalculator(risk_free_rate, trading_days)
    return calculator._calculate_sortino_ratio(returns)


def calculate_calmar_ratio(
    returns: pd.Series,
    trading_days: int = 252
) -> float:
    """計算卡瑪比率（便捷函數）"""
    if len(returns) == 0:
        return 0.0
    calculator = MetricsCalculator(trading_days=trading_days)
    annual_return = calculator._calculate_annual_return(returns)
    max_dd = calculate_max_drawdown(returns)
    return calculator._calculate_calmar_ratio(annual_return, max_dd)


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """計算 VaR（便捷函數）"""
    if len(returns) == 0:
        return 0.0
    return float(returns.quantile(1 - confidence_level))
