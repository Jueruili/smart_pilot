"""
績效指標計算模組

計算投資組合的各種績效指標，包括：
- 報酬率指標（總報酬、年化報酬、累積報酬）
- 風險指標（波動率、最大回撤、VaR）
- 風險調整報酬（Sharpe Ratio、Sortino Ratio、Calmar Ratio）

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional

import pandas as pd
import numpy as np


def calculate_returns(prices: pd.Series) -> pd.Series:
    """計算日報酬率

    Args:
        prices: 價格序列，index 為日期

    Returns:
        pd.Series: 日報酬率序列

    Example:
        >>> prices = pd.Series([100, 102, 101, 105])
        >>> returns = calculate_returns(prices)
    """
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """計算累積報酬率

    Args:
        returns: 日報酬率序列

    Returns:
        pd.Series: 累積報酬率序列

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.03])
        >>> cum_returns = calculate_cumulative_returns(returns)
    """
    return (1 + returns).cumprod() - 1


def calculate_total_return(returns: pd.Series) -> float:
    """計算總報酬率

    Args:
        returns: 日報酬率序列

    Returns:
        float: 總報酬率
    """
    return (1 + returns).prod() - 1


def calculate_annualized_return(
    returns: pd.Series,
    trading_days: int = 252
) -> float:
    """計算年化報酬率

    Args:
        returns: 日報酬率序列
        trading_days: 一年的交易日數，預設為 252

    Returns:
        float: 年化報酬率

    Formula:
        年化報酬率 = (1 + 總報酬率) ^ (交易日數 / 實際天數) - 1
    """
    total_return = calculate_total_return(returns)
    n_days = len(returns)

    if n_days == 0:
        return 0.0

    annualized = (1 + total_return) ** (trading_days / n_days) - 1
    return annualized


def calculate_volatility(
    returns: pd.Series,
    trading_days: int = 252
) -> float:
    """計算年化波動率

    Args:
        returns: 日報酬率序列
        trading_days: 一年的交易日數，預設為 252

    Returns:
        float: 年化波動率（標準差）
    """
    if len(returns) < 2:
        return 0.0

    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(trading_days)
    return annualized_vol


def calculate_max_drawdown(returns: pd.Series) -> float:
    """計算最大回撤

    最大回撤是從歷史高點到最低點的最大跌幅。

    Args:
        returns: 日報酬率序列

    Returns:
        float: 最大回撤（負值）

    Example:
        >>> returns = pd.Series([0.05, -0.10, 0.03, -0.15, 0.08])
        >>> mdd = calculate_max_drawdown(returns)
        >>> print(f"最大回撤: {mdd:.2%}")
    """
    if len(returns) == 0:
        return 0.0

    # 計算累積淨值
    cumulative = (1 + returns).cumprod()

    # 計算歷史高點
    running_max = cumulative.cummax()

    # 計算回撤
    drawdown = (cumulative - running_max) / running_max

    # 返回最大回撤
    return drawdown.min()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> float:
    """計算夏普比率 (Sharpe Ratio)

    衡量每單位風險的超額報酬。

    Args:
        returns: 日報酬率序列
        risk_free_rate: 無風險利率（年化），預設為 2%
        trading_days: 一年的交易日數，預設為 252

    Returns:
        float: 夏普比率

    Formula:
        Sharpe Ratio = (年化報酬率 - 無風險利率) / 年化波動率
    """
    ann_return = calculate_annualized_return(returns, trading_days)
    ann_vol = calculate_volatility(returns, trading_days)

    if ann_vol == 0:
        return 0.0

    sharpe = (ann_return - risk_free_rate) / ann_vol
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> float:
    """計算索提諾比率 (Sortino Ratio)

    類似夏普比率，但只考慮下行風險（負報酬的波動）。

    Args:
        returns: 日報酬率序列
        risk_free_rate: 無風險利率（年化），預設為 2%
        trading_days: 一年的交易日數，預設為 252

    Returns:
        float: 索提諾比率

    Formula:
        Sortino Ratio = (年化報酬率 - 無風險利率) / 下行波動率
    """
    ann_return = calculate_annualized_return(returns, trading_days)

    # 計算下行波動率（只考慮負報酬）
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return float('inf') if ann_return > risk_free_rate else 0.0

    downside_vol = negative_returns.std() * np.sqrt(trading_days)

    if downside_vol == 0:
        return float('inf') if ann_return > risk_free_rate else 0.0

    sortino = (ann_return - risk_free_rate) / downside_vol
    return sortino


def calculate_calmar_ratio(
    returns: pd.Series,
    trading_days: int = 252
) -> float:
    """計算卡瑪比率 (Calmar Ratio)

    衡量報酬與最大回撤的關係。

    Args:
        returns: 日報酬率序列
        trading_days: 一年的交易日數，預設為 252

    Returns:
        float: 卡瑪比率

    Formula:
        Calmar Ratio = 年化報酬率 / |最大回撤|
    """
    ann_return = calculate_annualized_return(returns, trading_days)
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0:
        return float('inf') if ann_return > 0 else 0.0

    # 最大回撤是負值，所以用絕對值
    calmar = ann_return / abs(max_dd)
    return calmar


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """計算風險值 (Value at Risk, VaR)

    在給定信心水準下，可能的最大損失。

    Args:
        returns: 日報酬率序列
        confidence_level: 信心水準，預設為 95%

    Returns:
        float: VaR 值（負值表示損失）

    Example:
        >>> var = calculate_var(returns, 0.95)
        >>> print(f"95% VaR: {var:.2%}")
    """
    if len(returns) == 0:
        return 0.0

    var = returns.quantile(1 - confidence_level)
    return var


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> dict[str, float]:
    """計算所有績效指標

    Args:
        returns: 日報酬率序列
        risk_free_rate: 無風險利率（年化）
        trading_days: 一年的交易日數

    Returns:
        dict: 包含所有績效指標的字典
    """
    return {
        "total_return": calculate_total_return(returns),
        "annualized_return": calculate_annualized_return(returns, trading_days),
        "volatility": calculate_volatility(returns, trading_days),
        "max_drawdown": calculate_max_drawdown(returns),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, trading_days),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate, trading_days),
        "calmar_ratio": calculate_calmar_ratio(returns, trading_days),
        "var_95": calculate_var(returns, 0.95),
        "var_99": calculate_var(returns, 0.99),
        "trading_days": len(returns),
    }


def format_metrics_report(metrics: dict[str, float]) -> str:
    """格式化績效指標報告

    Args:
        metrics: 績效指標字典

    Returns:
        str: 格式化的報告文字
    """
    report = """
=====================================
        投資組合績效報告
=====================================

【報酬指標】
  總報酬率:        {total_return:>10.2%}
  年化報酬率:      {annualized_return:>10.2%}

【風險指標】
  年化波動率:      {volatility:>10.2%}
  最大回撤:        {max_drawdown:>10.2%}
  95% VaR:        {var_95:>10.2%}
  99% VaR:        {var_99:>10.2%}

【風險調整報酬】
  夏普比率:        {sharpe_ratio:>10.2f}
  索提諾比率:      {sortino_ratio:>10.2f}
  卡瑪比率:        {calmar_ratio:>10.2f}

【其他資訊】
  交易天數:        {trading_days:>10d}
=====================================
""".format(**metrics)

    return report
