"""
optimizer.py

參數最佳化模組：
1. Grid Search：產生三維熱力圖數據（X=Kp, Y=Kd, 切片=Q）
2. SLSQP：精確找最佳 (Kp, Kd, Q)，固定 deadband

最佳化目標：最小化追蹤誤差 RMSE
約束條件：總交易成本 Cost ≤ cost_limit
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Optional
from joblib import Parallel, delayed

from core.benchmark import run_smart_pilot


# ─────────────────────────────────────────
# Grid Search（產生熱力圖數據）
# ─────────────────────────────────────────


def _single_grid_run(
    kp: float, kd: float, q: float, deadband: float,
    rets_stock, rets_bond, prices_stock, prices_bond,
    dates, target_w, fee_rate
) -> dict:
    """單次回測，供平行化使用。"""
    r = run_smart_pilot(
        rets_stock, rets_bond, prices_stock, prices_bond, dates,
        target_w=target_w, fee_rate=fee_rate,
        kf_q=q, kp=kp, kd=kd, deadband=deadband
    )
    return {
        "kp": kp, "kd": kd, "q": q,
        "rmse": r["rmse"],
        "cost": r["cost"],
        "sharpe": r["metrics"]["sharpe"],
        "ann_return": r["metrics"]["ann_return"],
    }


def run_grid_search(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    deadband: float = 0.0125,
    kp_range: List[float] = None,
    kd_range: List[float] = None,
    q_values: List[float] = None,
    n_jobs: int = -1,
) -> List[dict]:
    """
    Grid Search：掃描 (Kp, Kd) 網格，對每個 Q 切片執行。
    使用 joblib 平行化加速。

    Args:
        kp_range: Kp 網格，預設 [0.1, 0.2, ..., 1.0]
        kd_range: Kd 網格，預設 [0.1, 0.2, ..., 1.0]
        q_values: Q 切片，預設 [0.0001, 0.001, 0.01]
        n_jobs: 平行核心數，-1 表示全用

    Returns:
        List of dict，每個 dict 包含 kp, kd, q, rmse, cost, sharpe, ann_return
        可直接用於畫熱力圖
    """
    if kp_range is None:
        kp_range = np.linspace(0.1, 1.0, 10).tolist()
    if kd_range is None:
        kd_range = np.linspace(0.1, 1.0, 10).tolist()
    if q_values is None:
        q_values = [0.0001, 0.001, 0.01]

    tasks = [
        (kp, kd, q)
        for q in q_values
        for kp in kp_range
        for kd in kd_range
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_grid_run)(
            kp, kd, q, deadband,
            rets_stock, rets_bond, prices_stock, prices_bond,
            dates, target_w, fee_rate
        )
        for kp, kd, q in tasks
    )

    return results


def find_best_from_grid(grid_results: List[dict]) -> dict:
    """
    從 Grid Search 結果找最佳起點（RMSE 最小）。

    Returns:
        {"kp": float, "kd": float, "q": float, "rmse": float, "cost": float}
    """
    best = min(grid_results, key=lambda x: x["rmse"])
    return best


# ─────────────────────────────────────────
# SLSQP 精確最佳化
# ─────────────────────────────────────────


def run_slsqp(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    deadband: float = 0.0125,
    cost_limit: float = None,
    initial_params: dict = None,
) -> dict:
    """
    SLSQP 精確最佳化。
    最小化 RMSE，約束條件為 Cost ≤ cost_limit。

    Args:
        cost_limit: 交易成本上限（小數）。
                    None 表示不限制。
                    建議設為 Yearly Rebalance 的 cost 作為基準。
        initial_params: 起始點，格式 {"kp": float, "kd": float, "q": float}
                        建議從 Grid Search 最佳結果傳入。

    Returns:
        {
            "kp": float, "kd": float, "q": float,
            "rmse": float, "cost": float,
            "success": bool,
            "message": str,
        }
    """
    if initial_params is None:
        initial_params = {"kp": 0.5, "kd": 0.5, "q": 0.001}

    def objective(params):
        kp, kd, log_q = params
        q = np.exp(log_q)  # log scale 防止 Q 變負
        r = run_smart_pilot(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=q, kp=kp, kd=kd, deadband=deadband
        )
        return r["rmse"]

    def cost_constraint(params):
        kp, kd, log_q = params
        q = np.exp(log_q)
        r = run_smart_pilot(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=q, kp=kp, kd=kd, deadband=deadband
        )
        return cost_limit - r["cost"]  # >= 0 表示符合約束

    x0 = [
        initial_params["kp"],
        initial_params["kd"],
        np.log(initial_params["q"]),  # log scale
    ]

    bounds = [
        (0.01, 2.0),    # kp
        (0.01, 2.0),    # kd
        (np.log(1e-5), np.log(1.0)),  # log_q
    ]

    constraints = []
    if cost_limit is not None:
        constraints.append({"type": "ineq", "fun": cost_constraint})

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    opt_kp = float(result.x[0])
    opt_kd = float(result.x[1])
    opt_q = float(np.exp(result.x[2]))

    # 用最佳參數跑一次取得完整結果
    final = run_smart_pilot(
        rets_stock, rets_bond, prices_stock, prices_bond, dates,
        target_w=target_w, fee_rate=fee_rate,
        kf_q=opt_q, kp=opt_kp, kd=opt_kd, deadband=deadband
    )

    return {
        "kp": opt_kp,
        "kd": opt_kd,
        "q": opt_q,
        "rmse": final["rmse"],
        "cost": final["cost"],
        "success": bool(result.success),
        "message": result.message,
    }


# ─────────────────────────────────────────
# 超體積指標（Hypervolume Indicator）
# ─────────────────────────────────────────


def calc_hypervolume(
    frontier_points: List[dict],
    reference_point: dict = None,
) -> float:
    """
    計算 Pareto Frontier 的超體積指標（勒貝格測度）。

    面積越大 → 策略支配的空間越大 → 策略越有價值。
    注意：這裡 RMSE 和 Cost 越小越好，所以超體積是從參考點往左下計算。

    Args:
        frontier_points: [{"rmse": float, "cost": float}, ...]
                         必須已按 rmse 排序（從小到大）
        reference_point: {"rmse": float, "cost": float}
                         右上角參考點，動態設置時傳 None，
                         函數會自動用所有策略中最差點 * 1.1

    Returns:
        hypervolume: float（超體積面積）
    """
    if not frontier_points:
        return 0.0

    points = sorted(frontier_points, key=lambda p: p["rmse"])

    if reference_point is None:
        ref_rmse = max(p["rmse"] for p in points) * 1.1
        ref_cost = max(p["cost"] for p in points) * 1.1
    else:
        ref_rmse = reference_point["rmse"]
        ref_cost = reference_point["cost"]

    # 2D 超體積：沿 RMSE 軸掃描，累加矩形面積
    hv = 0.0
    prev_rmse = 0.0
    for p in points:
        width = p["rmse"] - prev_rmse
        height = ref_cost - p["cost"]
        if height > 0:
            hv += width * height
        prev_rmse = p["rmse"]

    return float(hv)


def compare_hypervolumes(pareto_data: dict) -> dict:
    """
    計算三個策略的超體積並比較。

    Args:
        pareto_data: scan_pareto_frontier() 的回傳值

    Returns:
        {
            "smart_pilot": float,
            "bangbang": float,
            "yearly": float,
            "reference_point": {"rmse": float, "cost": float},
            "winner": str,
        }
    """
    all_points = (
        pareto_data["smart_pilot"] +
        pareto_data["bangbang"] +
        [pareto_data["yearly"]]
    )
    ref_rmse = max(p["rmse"] for p in all_points) * 1.1
    ref_cost = max(p["cost"] for p in all_points) * 1.1
    reference_point = {"rmse": ref_rmse, "cost": ref_cost}

    hv_sp = calc_hypervolume(pareto_data["smart_pilot"], reference_point)
    hv_bb = calc_hypervolume(pareto_data["bangbang"], reference_point)
    hv_yr = calc_hypervolume([pareto_data["yearly"]], reference_point)

    scores = {"smart_pilot": hv_sp, "bangbang": hv_bb, "yearly": hv_yr}
    winner = max(scores, key=scores.get)

    return {
        "smart_pilot": hv_sp,
        "bangbang": hv_bb,
        "yearly": hv_yr,
        "reference_point": reference_point,
        "winner": winner,
    }
