"""
benchmark.py

三個再平衡策略的回測實作：
- Threshold-only（門檻再平衡）
- Time-and-threshold（時間＋門檻再平衡）
- Smart Pilot（PD + KF 控制器）

以及 Pareto Frontier 掃描函數。

主要指標：
- RMSE：追蹤誤差（小數）
- Cost：總交易成本 = turnover * fee_rate（小數）
- 所有績效指標均為年化數字
"""

import numpy as np
from typing import List, Dict

from core.kalman_filter import LogKalmanFilter
from core.pd_controller import PDController


# ── 共用函數 ──────────────────────────────────────────────


def get_metrics(nav_list: list, turnover: float = 0.0) -> dict:
    """
    計算年化績效指標。
    所有指標均年化，不使用總資金數字。

    Args:
        nav_list: 每日淨值列表，從 1.0 開始
        turnover: 總週轉率（用於計算年化週轉率）

    Returns:
        dict 包含年化報酬率、波動率、Sharpe、最大回撤、年化週轉率、年化資產波動率
    """
    wealth = np.array(nav_list, dtype=np.float64)
    rets = np.diff(wealth) / wealth[:-1]
    n_days = len(rets)

    if n_days == 0:
        return {
            "ann_return": 0.0, "volatility": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
            "ann_return_pct": 0.0, "volatility_pct": 0.0,
            "ann_turnover": 0.0, "ann_wealth_vol": 0.0,
        }

    n_years = len(nav_list) / 252.0

    # 年化報酬率（複利公式）
    ann_ret = float((wealth[-1] ** (252.0 / n_days)) - 1.0)
    # 年化波動率
    ann_vol = float(np.std(rets) * np.sqrt(252))
    # Sharpe（無風險利率 2%）
    sharpe = float((ann_ret - 0.02) / ann_vol) if ann_vol > 0 else 0.0
    # 最大回撤
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.min((wealth - peak) / peak))
    # 年化週轉率
    ann_turnover = turnover / n_years if n_years > 0 else 0.0
    # 年化總資產波動率
    wealth_rets = np.diff(wealth) / wealth[:-1]
    ann_wealth_vol = float(np.std(wealth_rets) * np.sqrt(252))

    return {
        "ann_return": ann_ret,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "ann_return_pct": ann_ret * 100.0,
        "volatility_pct": ann_vol * 100.0,
        "ann_turnover": ann_turnover,
        "ann_wealth_vol": ann_wealth_vol,
    }


def calc_rmse(weights: list, target_w: float) -> float:
    """
    計算追蹤誤差 RMSE。

    Args:
        weights: 每日股票權重列表（小數，例如 0.62）
        target_w: 目標權重（小數，例如 0.60）

    Returns:
        RMSE（小數）
    """
    arr = np.array(weights, dtype=np.float64)
    return float(np.sqrt(np.mean((arr - target_w) ** 2)))


# ── 策略 A：Threshold-only ─────────────────────────────


def run_threshold_only(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    drift_tolerance: float = 0.05,
    fee_rate: float = 0.003,
    warmup: int = 30,
) -> dict:
    """
    Threshold-only 門檻再平衡策略。
    偏離超過 drift_tolerance 時，拉回目標權重。
    """
    n = len(rets_stock)
    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []

    for i in range(warmup, n):
        val_stock = p_stock * wealth * (1.0 + rets_stock[i])
        val_bond = (1.0 - p_stock) * wealth * (1.0 + rets_bond[i])
        total = val_stock + val_bond
        curr_w = val_stock / total

        if abs(curr_w - target_w) > drift_tolerance:
            trade_pct = abs(curr_w - target_w)
            cost = trade_pct * total * fee_rate
            total -= cost
            turnover += trade_pct
            trade_count += 1
            direction = 1.0 if curr_w < target_w else -1.0
            actions.append(trade_pct * direction)
            p_stock = target_w
        else:
            p_stock = curr_w
            actions.append(0.0)

        wealth = total
        nav_list.append(wealth)
        weights.append(curr_w)

    rmse = calc_rmse(weights, target_w)
    cost = turnover * fee_rate

    return {
        "nav_list": nav_list,
        "weights": weights,
        "trade_count": trade_count,
        "turnover": turnover,
        "actions": actions,
        "rmse": rmse,
        "cost": cost,
        "metrics": get_metrics(nav_list, turnover=turnover),
    }


# ── 策略 B：Time-and-threshold ─────────────────────────────


def run_time_and_threshold(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    threshold: float = 0.05,
    warmup: int = 30,
) -> dict:
    """
    Time-and-threshold 再平衡策略。
    每年第一個交易日檢查，只有當 |curr_w - target_w| > threshold 才執行再平衡。
    """
    n = len(rets_stock)
    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []
    current_year = dates[warmup].year

    for i in range(warmup, n):
        val_stock = p_stock * wealth * (1.0 + rets_stock[i])
        val_bond = (1.0 - p_stock) * wealth * (1.0 + rets_bond[i])
        total = val_stock + val_bond
        curr_w = val_stock / total

        if dates[i].year > current_year:
            current_year = dates[i].year
            if abs(curr_w - target_w) > threshold:
                trade_pct = abs(curr_w - target_w)
                cost = trade_pct * total * fee_rate
                total -= cost
                turnover += trade_pct
                trade_count += 1
                direction = 1.0 if curr_w < target_w else -1.0
                actions.append(trade_pct * direction)
                p_stock = target_w
            else:
                p_stock = curr_w
                actions.append(0.0)
        else:
            p_stock = curr_w
            actions.append(0.0)

        wealth = total
        nav_list.append(wealth)
        weights.append(curr_w)

    rmse = calc_rmse(weights, target_w)
    cost = turnover * fee_rate

    return {
        "nav_list": nav_list,
        "weights": weights,
        "trade_count": trade_count,
        "turnover": turnover,
        "actions": actions,
        "rmse": rmse,
        "cost": cost,
        "metrics": get_metrics(nav_list, turnover=turnover),
    }


# ── 策略 C：Smart Pilot ──────────────────────────────────


def run_smart_pilot(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    kf_q: float = 0.001,
    kf_r: float = 0.005,
    kp: float = 0.5,
    kd: float = 0.5,
    deadband: float = 0.0125,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    warmup_prices_stock: np.ndarray = None,
    warmup_prices_bond: np.ndarray = None,
) -> dict:
    """
    Smart Pilot：PD 控制器 + Log-Space 卡爾曼濾波器。

    流程：
    1. 用暖機資料初始化 KF
       - 如果提供 warmup_prices_stock/bond，用這些資料初始化，回測從 index 0 開始
       - 如果不提供，用 prices_stock/bond 前 warmup 天初始化，從 warmup 開始回測
    2. 每天更新 KF，取得速度信號
    3. PD 控制器計算控制量 u
    4. |u| > deadband 才執行交易

    Args:
        warmup_prices_stock: 外部暖機資料（回測起始日前的價格序列）
        warmup_prices_bond: 外部暖機資料（回測起始日前的價格序列）
    """
    n = len(rets_stock)

    # 判斷是否使用外部暖機資料
    use_external_warmup = warmup_prices_stock is not None and warmup_prices_bond is not None

    if use_external_warmup:
        # 使用外部暖機資料初始化 KF
        kf_stock = LogKalmanFilter(np.log(warmup_prices_stock[0]), q=kf_q, r=kf_r)
        kf_bond = LogKalmanFilter(np.log(warmup_prices_bond[0]), q=kf_q, r=kf_r)
        for i in range(len(warmup_prices_stock)):
            kf_stock.predict()
            kf_stock.update(np.log(warmup_prices_stock[i]))
            kf_bond.predict()
            kf_bond.update(np.log(warmup_prices_bond[i]))
        # 外部暖機模式：回測從 index 0 開始
        start_idx = 0
    else:
        # 原本行為：用 prices_stock/bond 前 warmup 天初始化
        kf_stock = LogKalmanFilter(np.log(prices_stock[0]), q=kf_q, r=kf_r)
        kf_bond = LogKalmanFilter(np.log(prices_bond[0]), q=kf_q, r=kf_r)
        for i in range(warmup):
            kf_stock.predict()
            kf_stock.update(np.log(prices_stock[i]))
            kf_bond.predict()
            kf_bond.update(np.log(prices_bond[i]))
        # 原本行為：回測從 warmup 開始
        start_idx = warmup

    pd_ctrl = PDController(kp=kp, kd=kd, d_clip=d_clip, output_clip=output_clip)

    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []

    for i in range(start_idx, n):
        # 更新 KF
        kf_stock.predict()
        kf_stock.update(np.log(prices_stock[i]))
        kf_bond.predict()
        kf_bond.update(np.log(prices_bond[i]))

        vel_stock = kf_stock.velocity
        vel_bond = kf_bond.velocity

        # 更新資產價值
        val_stock = p_stock * wealth * (1.0 + rets_stock[i])
        val_bond = (1.0 - p_stock) * wealth * (1.0 + rets_bond[i])
        total = val_stock + val_bond
        curr_w = val_stock / total

        # PD 控制
        error = target_w - curr_w
        u = pd_ctrl.calculate(error, vel_stock, vel_bond)

        if abs(u) > deadband:
            trade_pct = abs(u)
            cost = trade_pct * total * fee_rate
            total -= cost
            turnover += trade_pct
            trade_count += 1
            new_w = np.clip(curr_w + u, 0.0, 1.0)
            p_stock = new_w
            actions.append(u)
        else:
            p_stock = curr_w
            actions.append(0.0)

        wealth = total
        nav_list.append(wealth)
        weights.append(curr_w)

    rmse = calc_rmse(weights, target_w)
    cost = turnover * fee_rate

    return {
        "nav_list": nav_list,
        "weights": weights,
        "trade_count": trade_count,
        "turnover": turnover,
        "actions": actions,
        "rmse": rmse,
        "cost": cost,
        "metrics": get_metrics(nav_list, turnover=turnover),
    }


# ── Pareto Frontier 掃描 ─────────────────────────────────


def scan_pareto_frontier(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    kf_q: float = 0.001,
    kf_r: float = 0.005,
    kp: float = 0.5,
    kd: float = 0.5,
    deadband_values: List[float] = None,
    n_points: int = 30,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    warmup_prices_stock: np.ndarray = None,
    warmup_prices_bond: np.ndarray = None,
    threshold_values: List[float] = None,
) -> dict:
    """
    掃描不同閾值，生成三個策略的 Pareto Frontier 數據。

    Smart Pilot：掃描 deadband（可傳入自訂列表或用 n_points 自動產生）
    Threshold-only：掃描 drift_tolerance 從 0.005 到 0.15
    Time-and-threshold：掃描 threshold 從 0.005 到 0.15

    Y 軸成本說明：
    - cost：總交易成本（小數）= turnover * fee_rate
    - ann_cost：年化成本 = cost / 回測年數（用於 Pareto 圖 Y 軸）

    Args:
        warmup_prices_stock: 外部暖機資料（回測起始日前的價格序列）
        warmup_prices_bond: 外部暖機資料（回測起始日前的價格序列）
        threshold_values: Time-and-threshold 掃描閾值列表

    Returns:
        {
            "smart_pilot":        [{"rmse", "cost", "ann_cost", "deadband"}, ...],
            "threshold_only":     [{"rmse", "cost", "ann_cost", "tolerance"}, ...],
            "time_and_threshold": [{"rmse", "cost", "ann_cost", "threshold"}, ...],
        }
    """
    n_years = len(dates) / 252.0
    results = {"smart_pilot": [], "threshold_only": [], "time_and_threshold": []}

    if deadband_values is None:
        deadband_values = np.linspace(0.001, 0.10, n_points).tolist()

    if threshold_values is None:
        threshold_values = np.linspace(0.005, 0.15, n_points).tolist()

    # Smart Pilot
    for deadband in deadband_values:
        r = run_smart_pilot(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=kf_q, kf_r=kf_r, kp=kp, kd=kd, deadband=float(deadband),
            warmup=warmup, d_clip=d_clip, output_clip=output_clip,
            warmup_prices_stock=warmup_prices_stock,
            warmup_prices_bond=warmup_prices_bond,
        )
        results["smart_pilot"].append({
            "rmse": r["rmse"],
            "cost": r["cost"],
            "ann_cost": r["cost"] / n_years,
            "deadband": float(deadband),
        })

    # Threshold-only
    for tolerance in np.linspace(0.005, 0.15, n_points):
        r = run_threshold_only(
            rets_stock, rets_bond, dates,
            target_w=target_w, drift_tolerance=float(tolerance), fee_rate=fee_rate,
            warmup=warmup,
        )
        results["threshold_only"].append({
            "rmse": r["rmse"],
            "cost": r["cost"],
            "ann_cost": r["cost"] / n_years,
            "tolerance": float(tolerance),
        })

    # Time-and-threshold（掃描 threshold）
    for threshold in threshold_values:
        r = run_time_and_threshold(
            rets_stock, rets_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            threshold=float(threshold),
            warmup=warmup,
        )
        results["time_and_threshold"].append({
            "rmse": r["rmse"],
            "cost": r["cost"],
            "ann_cost": r["cost"] / n_years,
            "threshold": float(threshold),
        })

    return results
