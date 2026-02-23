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
    kp: float, kd: float, q: float,
    rets_stock, rets_bond, prices_stock, prices_bond,
    dates, target_w, fee_rate,
    deadband_values, reference_point,
    kf_r: float = 0.005,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
) -> dict:
    """
    單組 (Kp, Kd, Q) 的完整評估，供平行化使用。

    掃描多個 deadband → 得到 Smart Pilot frontier → 計算超體積。
    同時也掃描 Bang-Bang frontier 作為比較基準（共用同一個參考點）。

    超體積越大 → 這組參數在整個 deadband 範圍內表現越好。
    """
    from core.benchmark import scan_pareto_frontier
    from core.optimizer import calc_hypervolume

    pareto = scan_pareto_frontier(
        rets_stock, rets_bond, prices_stock, prices_bond, dates,
        target_w=target_w, fee_rate=fee_rate,
        kf_q=q, kf_r=kf_r, kp=kp, kd=kd,
        deadband_values=deadband_values,
        warmup=warmup, d_clip=d_clip, output_clip=output_clip,
    )

    hv = calc_hypervolume(pareto["smart_pilot"], reference_point)

    return {
        "kp": kp,
        "kd": kd,
        "q": q,
        "hypervolume": hv,
        "best_rmse": min(p["rmse"] for p in pareto["smart_pilot"]),
        "best_cost": min(p["cost"] for p in pareto["smart_pilot"]),
    }


def run_grid_search(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    kp_range: List[float] = None,
    kd_range: List[float] = None,
    q_values: List[float] = None,
    deadband_values: List[float] = None,
    n_jobs: int = -1,
    kf_r: float = 0.005,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    ref_multiplier: float = 1.1,
) -> List[dict]:
    """
    Grid Search：掃描所有 (Kp, Kd, Q) 組合，每組計算超體積。
    使用 joblib 平行化加速。

    流程：
    1. 對每組 (Kp, Kd, Q)，掃描多個 deadband
    2. 得到 Smart Pilot 的 Pareto frontier
    3. 計算超體積作為這組參數的評分

    結果用於畫 3D scatter：X=Kp, Y=Kd, Z=Q, 顏色=超體積
    顏色越亮 → 超體積越大 → 參數越好

    Args:
        kp_range: Kp 網格，預設 10 個點 [0.1 ~ 1.0]
        kd_range: Kd 網格，預設 10 個點 [0.1 ~ 1.0]
        q_values: Q 候選值，預設 [0.0001, 0.001, 0.01]
        deadband_values: deadband 掃描範圍，預設 15 個點 [0.005 ~ 0.10]
        n_jobs: 平行核心數，-1 表示全用

    Returns:
        List of dict，每個 dict 包含：
        {"kp", "kd", "q", "hypervolume", "best_rmse", "best_cost"}
    """
    if kp_range is None:
        kp_range = np.linspace(0.1, 1.0, 10).tolist()
    if kd_range is None:
        kd_range = np.linspace(0.1, 1.0, 10).tolist()
    if q_values is None:
        q_values = [0.0001, 0.001, 0.01]
    if deadband_values is None:
        deadband_values = np.linspace(0.005, 0.10, 15).tolist()

    # 先算 Bang-Bang frontier 取得動態參考點
    from core.benchmark import run_bangbang
    bb_frontier = []
    for tol in np.linspace(0.005, 0.15, 15):
        r = run_bangbang(rets_stock, rets_bond, dates,
                         target_w=target_w, drift_tolerance=float(tol), fee_rate=fee_rate,
                         warmup=warmup)
        bb_frontier.append({"rmse": r["rmse"], "cost": r["cost"]})

    all_pts = bb_frontier
    ref_rmse = max(p["rmse"] for p in all_pts) * ref_multiplier
    ref_cost = max(p["cost"] for p in all_pts) * ref_multiplier
    reference_point = {"rmse": ref_rmse, "cost": ref_cost}

    tasks = [
        (kp, kd, q)
        for q in q_values
        for kp in kp_range
        for kd in kd_range
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_grid_run)(
            kp, kd, q,
            rets_stock, rets_bond, prices_stock, prices_bond,
            dates, target_w, fee_rate,
            deadband_values, reference_point,
            kf_r, warmup, d_clip, output_clip,
        )
        for kp, kd, q in tasks
    )

    return results


def find_best_from_grid(grid_results: List[dict]) -> dict:
    """
    從 Grid Search 結果找最佳起點（超體積最大）。

    Returns:
        {"kp": float, "kd": float, "q": float, "hypervolume": float}
    """
    best = max(grid_results, key=lambda x: x["hypervolume"])
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
    deadband_values: List[float] = None,
    initial_params: dict = None,
    reference_point: dict = None,
    kf_r: float = 0.005,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
) -> List[dict]:
    """
    SLSQP 精確最佳化。

    流程：
    - 對每個 deadband，固定 deadband，用 SLSQP 找最佳 (Kp, Kd, Q)
    - 目標：最小化 RMSE（追蹤誤差）
    - 每個 deadband 得到一個最佳參數組合和一個 (RMSE, Cost) 點
    - 所有點連起來就是 Smart Pilot 的 Pareto Frontier

    為什麼對每個 deadband 分別最佳化？
    因為 deadband 直接控制交易頻率，不應被最佳化到極端值，
    而是讓使用者選擇可接受的成本區間，再找那個區間內最好的參數。

    Args:
        deadband_values: 要掃描的 deadband 列表，預設 20 個點 [0.005 ~ 0.10]
        initial_params: 起始點 {"kp", "kd", "q"}，建議從 Grid Search 最佳結果傳入
        reference_point: 超體積參考點 {"rmse", "cost"}，None 則動態設置

    Returns:
        List of dict，每個 deadband 對應一個最佳結果：
        [{"deadband", "kp", "kd", "q", "rmse", "cost", "ann_cost", "success", "message"}, ...]
        可直接用於畫 Pareto Frontier
    """
    from core.benchmark import run_smart_pilot as _run_sp

    if deadband_values is None:
        deadband_values = np.linspace(0.005, 0.10, 20).tolist()
    if initial_params is None:
        initial_params = {"kp": 0.5, "kd": 0.5, "q": 0.001}

    bounds = [
        (0.01, 2.0),                          # kp
        (0.01, 2.0),                          # kd
        (np.log(1e-5), np.log(1.0)),          # log_q（log scale 防止負值）
    ]

    all_results = []

    for deadband in deadband_values:
        deadband = float(deadband)

        def objective(params):
            kp, kd, log_q = params
            q = np.exp(log_q)
            r = _run_sp(
                rets_stock, rets_bond, prices_stock, prices_bond, dates,
                target_w=target_w, fee_rate=fee_rate,
                kf_q=q, kf_r=kf_r, kp=kp, kd=kd, deadband=deadband,
                warmup=warmup, d_clip=d_clip, output_clip=output_clip,
            )
            return r["rmse"]  # 最小化追蹤誤差

        x0 = [
            initial_params["kp"],
            initial_params["kd"],
            np.log(initial_params["q"]),
        ]

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        opt_kp = float(result.x[0])
        opt_kd = float(result.x[1])
        opt_q = float(np.exp(result.x[2]))

        final = _run_sp(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=opt_q, kf_r=kf_r, kp=opt_kp, kd=opt_kd, deadband=deadband,
            warmup=warmup, d_clip=d_clip, output_clip=output_clip,
        )

        all_results.append({
            "deadband": deadband,
            "kp": opt_kp,
            "kd": opt_kd,
            "q": opt_q,
            "rmse": final["rmse"],
            "cost": final["cost"],
            "ann_cost": final["cost"] / (len(dates) / 252),
            "success": bool(result.success),
            "message": result.message,
        })

    return all_results


# ─────────────────────────────────────────
# CMA-ES 全域最佳化
# ─────────────────────────────────────────


def run_cma_es(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    deadband_values: List[float] = None,
    kf_r: float = 0.005,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    ref_multiplier: float = 1.1,
    x0: List[float] = None,
    sigma0: float = 0.5,
    maxiter: int = 100,
    popsize: int = 10,
) -> dict:
    """
    CMA-ES 全域最佳化。

    最大化超體積（傳入負值給 CMA-ES 最小化）。
    搜尋最佳 (Kp, Kd, Q) 參數組合。

    Args:
        x0: 起始點 [kp, kd, log_q]，預設 [2.0, 1.5, log(0.001)]
        sigma0: 初始步長
        maxiter: 最大迭代次數
        popsize: 族群大小

    Returns:
        dict 包含最佳參數、超體積、Pareto frontier 等
    """
    import cma
    from core.benchmark import scan_pareto_frontier, run_bangbang

    if deadband_values is None:
        deadband_values = np.linspace(0.005, 0.10, 15).tolist()

    # 先算 Bang-Bang 參考點
    bb_frontier = []
    for tol in np.linspace(0.005, 0.15, 15):
        r = run_bangbang(rets_stock, rets_bond, dates,
                         target_w=target_w, drift_tolerance=float(tol), fee_rate=fee_rate,
                         warmup=warmup)
        bb_frontier.append({"rmse": r["rmse"], "cost": r["cost"]})

    ref_rmse = max(p["rmse"] for p in bb_frontier) * ref_multiplier
    ref_cost = max(p["cost"] for p in bb_frontier) * ref_multiplier
    reference_point = {"rmse": ref_rmse, "cost": ref_cost}

    def objective(params):
        kp, kd, log_q = params
        q = float(np.exp(np.clip(log_q, np.log(1e-5), np.log(1.0))))
        kp = float(np.clip(kp, 0.01, 5.0))
        kd = float(np.clip(kd, 0.01, 5.0))
        pareto = scan_pareto_frontier(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=q, kf_r=kf_r, kp=kp, kd=kd,
            deadband_values=deadband_values,
            warmup=warmup, d_clip=d_clip, output_clip=output_clip,
        )
        hv = calc_hypervolume(pareto["smart_pilot"], reference_point)
        return -hv  # 最大化超體積 = 最小化負超體積

    # 起點預設
    if x0 is None:
        x0 = [2.0, 1.5, np.log(0.001)]

    # bounds
    lower_bounds = [0.01, 0.01, np.log(1e-5)]
    upper_bounds = [5.0, 5.0, np.log(1.0)]

    opts = {
        "maxiter": maxiter,
        "popsize": popsize,
        "bounds": [lower_bounds, upper_bounds],
        "verbose": -9,  # 抑制輸出
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(objective)

    best_params = es.result.xbest
    opt_kp = float(np.clip(best_params[0], 0.01, 5.0))
    opt_kd = float(np.clip(best_params[1], 0.01, 5.0))
    opt_q = float(np.exp(np.clip(best_params[2], np.log(1e-5), np.log(1.0))))
    best_hv = -es.result.fbest

    # 用最佳參數跑一次完整 Pareto
    pareto = scan_pareto_frontier(
        rets_stock, rets_bond, prices_stock, prices_bond, dates,
        target_w=target_w, fee_rate=fee_rate,
        kf_q=opt_q, kf_r=kf_r, kp=opt_kp, kd=opt_kd,
        deadband_values=deadband_values,
        warmup=warmup, d_clip=d_clip, output_clip=output_clip,
    )

    return {
        "kp": opt_kp,
        "kd": opt_kd,
        "q": opt_q,
        "hypervolume": best_hv,
        "hypervolume_pct": best_hv * 10000,
        "pareto": pareto,
        "iterations": es.result.iterations,
        "evaluations": es.result.evaluations,
        "success": True,
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
    for i in range(len(points)):
        current_p = points[i]
        if i < len(points) - 1:
            next_rmse = points[i + 1]["rmse"]
        else:
            next_rmse = ref_rmse
        width = next_rmse - current_p["rmse"]
        height = ref_cost - current_p["cost"]
        if width > 0 and height > 0:
            hv += width * height

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
