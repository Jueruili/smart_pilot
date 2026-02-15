"""
對照策略與帕雷托前線掃描模組

提供兩種傳統再平衡策略 + Smart Pilot 單次回測 + 帕雷托前線掃描：
1. Bang-Bang Control（門檻再平衡）：偏離超過 drift_tolerance 才交易
2. Yearly Rebalance（年度再平衡）：每年第一個交易日再平衡
3. run_smart_pilot_single：簡化版 Smart Pilot 回測（KF + PD）
4. scan_pareto_frontier：掃描不同閾值，生成帕雷托前線數據

作者：Smart Pilot Team
版本：2.0.0
"""

import numpy as np
from typing import Optional

from core.kalman_filter import LogKalmanFilter
from core.pd_controller import PDController


def get_enhanced_metrics(wealth_list: list) -> dict:
    """計算績效指標

    Args:
        wealth_list: 每日淨值列表（從 1.0 開始）

    Returns:
        dict: ann_return, volatility, sharpe, max_drawdown
    """
    wealth_array = np.array(wealth_list)
    rets = np.diff(wealth_array) / wealth_array[:-1]
    if len(rets) == 0:
        return {"ann_return": 0, "volatility": 0, "sharpe": 0, "max_drawdown": 0}
    ann_ret = (wealth_array[-1] ** (252 / len(rets))) - 1
    ann_vol = np.std(rets) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol != 0 else 0
    peak = np.maximum.accumulate(wealth_array)
    drawdown = (wealth_array - peak) / peak
    mdd = np.min(drawdown)
    return {"ann_return": ann_ret, "volatility": ann_vol, "sharpe": sharpe, "max_drawdown": mdd}


def _compute_cer(weights: list, target_w: float, turnover: float, fee_rate: float) -> float:
    """計算 CER 分數

    Args:
        weights: 每日股票權重列表（百分比）
        target_w: 目標權重（0-1）
        turnover: 總週轉率
        fee_rate: 手續費率

    Returns:
        float: CER 分數
    """
    weights_array = np.array(weights)
    rmse = np.sqrt(np.mean((weights_array - target_w * 100) ** 2))
    real_cost = turnover * fee_rate
    if rmse == 0:
        rmse = 0.001
    if real_cost == 0:
        real_cost = 0.00001
    cer = 0.1 / (rmse * real_cost)
    return cer


def _compute_rmse(weights: list, target_w: float) -> float:
    """計算追蹤誤差 RMSE

    Args:
        weights: 每日股票權重列表（百分比）
        target_w: 目標權重（0-1）

    Returns:
        float: RMSE
    """
    weights_array = np.array(weights)
    return np.sqrt(np.mean((weights_array - target_w * 100) ** 2))


def run_bangbang(
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float,
    drift_tolerance: float,
    fee_rate: float
) -> dict:
    """執行 Bang-Bang Control（門檻再平衡）策略

    偏離超過 drift_tolerance 才交易，直接拉回目標比例。
    回測從第 30 天開始（KF 暖機期）。

    Args:
        prices_stock: 股票價格序列
        prices_bond: 債券價格序列
        rets_stock: 股票日報酬率序列
        rets_bond: 債券日報酬率序列
        dates: 日期列表
        target_w: 目標股票權重（0-1）
        drift_tolerance: 偏離門檻
        fee_rate: 手續費率

    Returns:
        dict: nav_list, weights, trade_count, turnover, actions, metrics, rmse, cer
    """
    n = len(prices_stock)
    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []

    for i in range(30, n):
        val_stock = p_stock * wealth * (1 + rets_stock[i])
        val_bond = (1 - p_stock) * wealth * (1 + rets_bond[i])
        total = val_stock + val_bond
        curr_w = val_stock / total

        if abs(curr_w - target_w) > drift_tolerance:
            trade_pct = abs(curr_w - target_w)
            turnover += trade_pct
            total -= trade_pct * total * fee_rate
            direction = -1 if curr_w > target_w else 1
            actions.append(trade_pct * direction)
            p_stock = target_w
            trade_count += 1
        else:
            p_stock = curr_w
            actions.append(0)

        wealth = total
        nav_list.append(wealth)
        weights.append(curr_w * 100)

    metrics = get_enhanced_metrics(nav_list)
    rmse = _compute_rmse(weights, target_w)
    cer = _compute_cer(weights, target_w, turnover, fee_rate)

    return {
        "nav_list": nav_list,
        "weights": weights,
        "trade_count": trade_count,
        "turnover": turnover,
        "actions": actions,
        "metrics": metrics,
        "rmse": rmse,
        "cer": cer,
    }


def run_yearly(
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float,
    fee_rate: float
) -> dict:
    """執行 Yearly Rebalance（年度再平衡）策略

    每年第一個交易日拉回目標比例。
    回測從第 30 天開始（KF 暖機期）。

    Args:
        prices_stock: 股票價格序列
        prices_bond: 債券價格序列
        rets_stock: 股票日報酬率序列
        rets_bond: 債券日報酬率序列
        dates: 日期列表
        target_w: 目標股票權重（0-1）
        fee_rate: 手續費率

    Returns:
        dict: nav_list, weights, trade_count, turnover, actions, metrics, rmse, cer
    """
    n = len(prices_stock)
    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []
    current_year = dates[29].year

    for i in range(30, n):
        val_stock = p_stock * wealth * (1 + rets_stock[i])
        val_bond = (1 - p_stock) * wealth * (1 + rets_bond[i])
        total = val_stock + val_bond
        curr_w = val_stock / total

        if dates[i].year > current_year:
            trade_pct = abs(curr_w - target_w)
            turnover += trade_pct
            total -= trade_pct * total * fee_rate
            direction = -1 if curr_w > target_w else 1
            actions.append(trade_pct * direction)
            p_stock = target_w
            current_year = dates[i].year
            trade_count += 1
        else:
            p_stock = curr_w
            actions.append(0)

        wealth = total
        nav_list.append(wealth)
        weights.append(curr_w * 100)

    metrics = get_enhanced_metrics(nav_list)
    rmse = _compute_rmse(weights, target_w)
    cer = _compute_cer(weights, target_w, turnover, fee_rate)

    return {
        "nav_list": nav_list,
        "weights": weights,
        "trade_count": trade_count,
        "turnover": turnover,
        "actions": actions,
        "metrics": metrics,
        "rmse": rmse,
        "cer": cer,
    }


def run_smart_pilot_single(
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float,
    fee_rate: float,
    kf_q: float,
    kp: float,
    kd: float,
    real_threshold: float
) -> dict:
    """簡化版 Smart Pilot 回測（KF + PD）

    與 BacktestEngine 邏輯相同，但不需要完整 history DataFrame，
    只回傳 rmse, turnover, cer 等數字。用於帕雷托前線掃描。
    回測從第 30 天開始（KF 暖機期）。

    Args:
        prices_stock: 股票價格序列
        prices_bond: 債券價格序列
        rets_stock: 股票日報酬率序列
        rets_bond: 債券日報酬率序列
        dates: 日期列表
        target_w: 目標股票權重（0-1）
        fee_rate: 手續費率
        kf_q: 卡爾曼濾波器 Q 值
        kp: PD 控制器 Kp
        kd: PD 控制器 Kd
        real_threshold: 死區閾值

    Returns:
        dict: nav_list, weights, trade_count, turnover, actions, metrics, rmse, cer
    """
    n = len(prices_stock)

    # 初始化 KF（兩個通道）
    kf_stock = LogKalmanFilter(initial_log_price=np.log(prices_stock[0]), q=kf_q)
    kf_bond = LogKalmanFilter(initial_log_price=np.log(prices_bond[0]), q=kf_q)

    # KF 暖機期（前 30 天）
    for i in range(1, 30):
        kf_stock.predict()
        kf_stock.update(np.log(prices_stock[i]))
        kf_bond.predict()
        kf_bond.update(np.log(prices_bond[i]))

    # 初始化 PD 控制器
    pd_ctrl = PDController(kp=kp, kd=kd)

    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []

    for i in range(30, n):
        # KF 更新
        kf_stock.predict()
        kf_stock.update(np.log(prices_stock[i]))
        kf_bond.predict()
        kf_bond.update(np.log(prices_bond[i]))

        # 計算市值
        val_stock = p_stock * wealth * (1 + rets_stock[i])
        val_bond = (1 - p_stock) * wealth * (1 + rets_bond[i])
        total = val_stock + val_bond
        curr_w = val_stock / total

        # PD 控制
        error = target_w - curr_w
        u = pd_ctrl.calculate(error, kf_stock.velocity, kf_bond.velocity)

        if abs(u) > real_threshold:
            trade_pct = abs(u)
            turnover += trade_pct
            total -= trade_pct * total * fee_rate
            direction = 1 if u > 0 else -1
            actions.append(trade_pct * direction)
            p_stock = curr_w + u
            p_stock = np.clip(p_stock, 0.0, 1.0)
            trade_count += 1
        else:
            p_stock = curr_w
            actions.append(0)

        wealth = total
        nav_list.append(wealth)
        weights.append(curr_w * 100)

    metrics = get_enhanced_metrics(nav_list)
    rmse = _compute_rmse(weights, target_w)
    cer = _compute_cer(weights, target_w, turnover, fee_rate)

    return {
        "nav_list": nav_list,
        "weights": weights,
        "trade_count": trade_count,
        "turnover": turnover,
        "actions": actions,
        "metrics": metrics,
        "rmse": rmse,
        "cer": cer,
    }


def scan_pareto_frontier(
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float,
    fee_rate: float,
    kf_q: float,
    kp: float,
    kd: float
) -> dict:
    """掃描不同閾值，生成帕雷托前線數據

    對 Smart Pilot 和 Bang-Bang 分別掃描多組閾值：
    - Smart Pilot: real_threshold 從 0.001 到 0.15，約 30 個點
    - Bang-Bang: drift_tolerance 從 0.005 到 0.15，約 30 個點

    Args:
        prices_stock: 股票價格序列
        prices_bond: 債券價格序列
        rets_stock: 股票日報酬率序列
        rets_bond: 債券日報酬率序列
        dates: 日期列表
        target_w: 目標股票權重（0-1）
        fee_rate: 手續費率
        kf_q: 卡爾曼濾波器 Q 值
        kp: PD 控制器 Kp
        kd: PD 控制器 Kd

    Returns:
        dict: {
            "smart_pilot": [{"rmse": x, "cost": y, "threshold": t}, ...],
            "bangbang": [{"rmse": x, "cost": y, "threshold": t}, ...],
            "yearly": {"rmse": x, "cost": y},
        }
    """
    results = {"smart_pilot": [], "bangbang": [], "yearly": None}

    # Smart Pilot: 掃描不同 real_threshold
    for threshold in np.linspace(0.001, 0.15, 30):
        result = run_smart_pilot_single(
            prices_stock, prices_bond, rets_stock, rets_bond, dates,
            target_w, fee_rate, kf_q, kp, kd, threshold
        )
        results["smart_pilot"].append({
            "rmse": result["rmse"],
            "cost": result["turnover"] * fee_rate,
            "threshold": threshold
        })

    # Bang-Bang: 掃描不同 drift_tolerance
    for tolerance in np.linspace(0.005, 0.15, 30):
        result = run_bangbang(
            prices_stock, prices_bond, rets_stock, rets_bond, dates,
            target_w, tolerance, fee_rate
        )
        results["bangbang"].append({
            "rmse": result["rmse"],
            "cost": result["turnover"] * fee_rate,
            "threshold": tolerance
        })

    # Yearly: 只有一個點
    yearly = run_yearly(
        prices_stock, prices_bond, rets_stock, rets_bond, dates,
        target_w, fee_rate
    )
    results["yearly"] = {
        "rmse": yearly["rmse"],
        "cost": yearly["turnover"] * fee_rate
    }

    return results
