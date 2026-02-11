"""
對照策略模組

提供兩種傳統再平衡策略作為 Smart Pilot (PID) 的對照基準：
1. Threshold Rebalance（門檻再平衡）：偏離超過門檻才交易
2. Yearly Rebalance（年度再平衡）：每年第一個交易日再平衡

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional
import pandas as pd
import numpy as np


def run_threshold_rebalance(
    data: pd.DataFrame,
    initial_cash: float,
    target_ratio: float,
    threshold: float = 0.05,
    commission_rate: float = 0.001
) -> pd.DataFrame:
    """執行門檻再平衡策略

    每天檢查股票權重偏離目標比例多少，偏離超過門檻才執行交易，
    直接拉回目標比例。

    Args:
        data: 價格資料 DataFrame，columns 為股票和債券代碼
        initial_cash: 初始資金
        target_ratio: 目標股票比例（0-1）
        threshold: 偏離門檻（預設 5%）
        commission_rate: 手續費率

    Returns:
        pd.DataFrame: 回測歷史，包含 nav, ratio, trade_flag, commission
    """
    if data.empty or len(data.columns) < 2:
        raise ValueError("資料必須包含至少兩個標的")

    stock_col = data.columns[0]
    bond_col = data.columns[1]

    # 初始化
    dates = data.index
    n_days = len(dates)

    # 歷史記錄
    history = {
        "nav": np.zeros(n_days),
        "ratio": np.zeros(n_days),
        "trade_flag": np.zeros(n_days, dtype=bool),
        "commission": np.zeros(n_days),
    }

    # 第一天：按目標比例買入
    first_price_stock = data[stock_col].iloc[0]
    first_price_bond = data[bond_col].iloc[0]

    stock_value = initial_cash * target_ratio
    bond_value = initial_cash * (1 - target_ratio)

    stock_shares = stock_value / first_price_stock
    bond_shares = bond_value / first_price_bond

    # 扣除初始買入手續費
    init_commission = initial_cash * commission_rate
    nav = initial_cash - init_commission

    # 重新計算持股（扣除手續費後）
    stock_value = nav * target_ratio
    bond_value = nav * (1 - target_ratio)
    stock_shares = stock_value / first_price_stock
    bond_shares = bond_value / first_price_bond

    history["nav"][0] = nav
    history["ratio"][0] = target_ratio
    history["trade_flag"][0] = True
    history["commission"][0] = init_commission

    total_commission = init_commission

    # 逐日模擬
    for i in range(1, n_days):
        price_stock = data[stock_col].iloc[i]
        price_bond = data[bond_col].iloc[i]

        # 計算當前市值
        stock_value = stock_shares * price_stock
        bond_value = bond_shares * price_bond
        nav = stock_value + bond_value

        # 計算當前股票比例
        current_ratio = stock_value / nav if nav > 0 else 0

        # 檢查是否需要再平衡
        deviation = abs(current_ratio - target_ratio)
        trade_flag = False
        commission = 0.0

        if deviation > threshold:
            # 需要再平衡：計算交易金額
            target_stock_value = nav * target_ratio
            target_bond_value = nav * (1 - target_ratio)

            trade_amount = abs(target_stock_value - stock_value) + abs(target_bond_value - bond_value)
            commission = trade_amount * commission_rate

            # 扣除手續費後重新分配
            nav_after_commission = nav - commission
            stock_value = nav_after_commission * target_ratio
            bond_value = nav_after_commission * (1 - target_ratio)

            stock_shares = stock_value / price_stock
            bond_shares = bond_value / price_bond

            nav = nav_after_commission
            current_ratio = target_ratio
            trade_flag = True
            total_commission += commission

        history["nav"][i] = nav
        history["ratio"][i] = current_ratio
        history["trade_flag"][i] = trade_flag
        history["commission"][i] = commission

    # 建立 DataFrame
    result_df = pd.DataFrame(history, index=dates)
    result_df.index.name = "date"

    return result_df


def run_yearly_rebalance(
    data: pd.DataFrame,
    initial_cash: float,
    target_ratio: float,
    commission_rate: float = 0.001
) -> pd.DataFrame:
    """執行年度再平衡策略

    平時完全不動，每年第一個交易日不管偏離多少，直接拉回目標比例。

    Args:
        data: 價格資料 DataFrame，columns 為股票和債券代碼
        initial_cash: 初始資金
        target_ratio: 目標股票比例（0-1）
        commission_rate: 手續費率

    Returns:
        pd.DataFrame: 回測歷史，包含 nav, ratio, trade_flag, commission
    """
    if data.empty or len(data.columns) < 2:
        raise ValueError("資料必須包含至少兩個標的")

    stock_col = data.columns[0]
    bond_col = data.columns[1]

    # 初始化
    dates = data.index
    n_days = len(dates)

    # 歷史記錄
    history = {
        "nav": np.zeros(n_days),
        "ratio": np.zeros(n_days),
        "trade_flag": np.zeros(n_days, dtype=bool),
        "commission": np.zeros(n_days),
    }

    # 第一天：按目標比例買入
    first_price_stock = data[stock_col].iloc[0]
    first_price_bond = data[bond_col].iloc[0]

    # 扣除初始買入手續費
    init_commission = initial_cash * commission_rate
    nav = initial_cash - init_commission

    stock_value = nav * target_ratio
    bond_value = nav * (1 - target_ratio)
    stock_shares = stock_value / first_price_stock
    bond_shares = bond_value / first_price_bond

    history["nav"][0] = nav
    history["ratio"][0] = target_ratio
    history["trade_flag"][0] = True
    history["commission"][0] = init_commission

    total_commission = init_commission
    current_year = dates[0].year

    # 逐日模擬
    for i in range(1, n_days):
        price_stock = data[stock_col].iloc[i]
        price_bond = data[bond_col].iloc[i]

        # 計算當前市值
        stock_value = stock_shares * price_stock
        bond_value = bond_shares * price_bond
        nav = stock_value + bond_value

        # 計算當前股票比例
        current_ratio = stock_value / nav if nav > 0 else 0

        # 檢查是否為新年度的第一個交易日
        this_year = dates[i].year
        trade_flag = False
        commission = 0.0

        if this_year != current_year:
            # 新年度：執行再平衡
            current_year = this_year

            # 計算交易金額
            target_stock_value = nav * target_ratio
            target_bond_value = nav * (1 - target_ratio)

            trade_amount = abs(target_stock_value - stock_value) + abs(target_bond_value - bond_value)
            commission = trade_amount * commission_rate

            # 扣除手續費後重新分配
            nav_after_commission = nav - commission
            stock_value = nav_after_commission * target_ratio
            bond_value = nav_after_commission * (1 - target_ratio)

            stock_shares = stock_value / price_stock
            bond_shares = bond_value / price_bond

            nav = nav_after_commission
            current_ratio = target_ratio
            trade_flag = True
            total_commission += commission

        history["nav"][i] = nav
        history["ratio"][i] = current_ratio
        history["trade_flag"][i] = trade_flag
        history["commission"][i] = commission

    # 建立 DataFrame
    result_df = pd.DataFrame(history, index=dates)
    result_df.index.name = "date"

    return result_df


def calculate_tracking_error(ratio_series: pd.Series, target_ratio: float) -> float:
    """計算追蹤誤差 (RMSE)

    Args:
        ratio_series: 實際比例序列
        target_ratio: 目標比例

    Returns:
        float: 追蹤誤差 RMSE
    """
    return np.sqrt(np.mean((ratio_series - target_ratio) ** 2))
