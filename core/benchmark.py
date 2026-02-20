"""
benchmark.py

三個再平衡策略的回測實作：
- Bang-Bang Control（門檻再平衡）
- Yearly Rebalance（年度再平衡）
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


def get_metrics(nav_list: list) -> dict:
    """
    計算年化績效指標。
    所有指標均年化，不使用總資金數字。

    Args:
        nav_list: 每日淨值列表，從 1.0 開始

    Returns:
        dict 包含年化報酬率、波動率、Sharpe、最大回撤
    """
    wealth = np.array(nav_list, dtype=np.float64)
    rets = np.diff(wealth) / wealth[:-1]
    n_days = len(rets)

    if n_days == 0:
        return {
            "ann_return": 0.0, "volatility": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
            "ann_return_pct": 0.0, "volatility_pct": 0.0,
        }

    # 年化報酬率（複利公式）
    ann_ret = float((wealth[-1] ** (252.0 / n_days)) - 1.0)
    # 年化波動率
    ann_vol = float(np.std(rets) * np.sqrt(252))
    # Sharpe（無風險利率 2%）
    sharpe = float((ann_ret - 0.02) / ann_vol) if ann_vol > 0 else 0.0
    # 最大回撤
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.min((wealth - peak) / peak))

    return {
        "ann_return": ann_ret,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "ann_return_pct": ann_ret * 100.0,
        "volatility_pct": ann_vol * 100.0,
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


# ── 策略 A：Bang-Bang Control ─────────────────────────────


def run_bangbang(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    drift_tolerance: float = 0.05,
    fee_rate: float = 0.003,
    warmup: int = 30,
) -> dict:
    """
    Bang-Bang 門檻再平衡策略。
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
        "metrics": get_metrics(nav_list),
    }


# ── 策略 B：Yearly Rebalance ─────────────────────────────


def run_yearly(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    warmup: int = 30,
) -> dict:
    """
    年度再平衡策略。
    每年第一個交易日拉回目標權重。
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
            trade_pct = abs(curr_w - target_w)
            cost = trade_pct * total * fee_rate
            total -= cost
            turnover += trade_pct
            trade_count += 1
            current_year = dates[i].year
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
        "metrics": get_metrics(nav_list),
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
    kp: float = 0.5,
    kd: float = 0.5,
    deadband: float = 0.0125,
    warmup: int = 30,
) -> dict:
    """
    Smart Pilot：PD 控制器 + Log-Space 卡爾曼濾波器。

    流程：
    1. 用前 warmup 天初始化 KF（暖機）
    2. 從第 warmup 天開始正式回測
    3. 每天更新 KF，取得速度信號
    4. PD 控制器計算控制量 u
    5. |u| > deadband 才執行交易
    """
    n = len(rets_stock)

    # 初始化 KF（暖機）
    kf_stock = LogKalmanFilter(np.log(prices_stock[0]), q=kf_q)
    kf_bond = LogKalmanFilter(np.log(prices_bond[0]), q=kf_q)
    for i in range(warmup):
        kf_stock.predict()
        kf_stock.update(np.log(prices_stock[i]))
        kf_bond.predict()
        kf_bond.update(np.log(prices_bond[i]))

    pd_ctrl = PDController(kp=kp, kd=kd)

    wealth = 1.0
    p_stock = target_w
    turnover = 0.0
    trade_count = 0
    nav_list = [1.0]
    weights = []
    actions = []

    for i in range(warmup, n):
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
        "metrics": get_metrics(nav_list),
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
    kp: float = 0.5,
    kd: float = 0.5,
    deadband_values: List[float] = None,
    n_points: int = 30,
) -> dict:
    """
    掃描不同閾值，生成三個策略的 Pareto Frontier 數據。

    Smart Pilot：掃描 deadband（可傳入自訂列表或用 n_points 自動產生）
    Bang-Bang：掃描 drift_tolerance 從 0.005 到 0.15
    Yearly：只有一個點（沒有門檻可調，所以只有一個結果）

    Y 軸成本說明：
    - cost：總交易成本（小數）= turnover * fee_rate
    - ann_cost：年化成本 = cost / 回測年數（用於 Pareto 圖 Y 軸）

    Returns:
        {
            "smart_pilot": [{"rmse", "cost", "ann_cost", "deadband"}, ...],
            "bangbang":    [{"rmse", "cost", "ann_cost", "tolerance"}, ...],
            "yearly":      {"rmse", "cost", "ann_cost"},
        }
    """
    n_years = len(dates) / 252.0
    results = {"smart_pilot": [], "bangbang": [], "yearly": None}

    if deadband_values is None:
        deadband_values = np.linspace(0.001, 0.10, n_points).tolist()

    # Smart Pilot
    for deadband in deadband_values:
        r = run_smart_pilot(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=kf_q, kp=kp, kd=kd, deadband=float(deadband)
        )
        results["smart_pilot"].append({
            "rmse": r["rmse"],
            "cost": r["cost"],
            "ann_cost": r["cost"] / n_years,
            "deadband": float(deadband),
        })

    # Bang-Bang
    for tolerance in np.linspace(0.005, 0.15, n_points):
        r = run_bangbang(
            rets_stock, rets_bond, dates,
            target_w=target_w, drift_tolerance=float(tolerance), fee_rate=fee_rate
        )
        results["bangbang"].append({
            "rmse": r["rmse"],
            "cost": r["cost"],
            "ann_cost": r["cost"] / n_years,
            "tolerance": float(tolerance),
        })

    # Yearly（只有一個點）
    r = run_yearly(rets_stock, rets_bond, dates, target_w=target_w, fee_rate=fee_rate)
    results["yearly"] = {
        "rmse": r["rmse"],
        "cost": r["cost"],
        "ann_cost": r["cost"] / n_years,
    }

    return results
