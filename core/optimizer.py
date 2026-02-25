"""
optimizer.py

參數最佳化模組：
1. Grid Search：產生三維熱力圖數據（X=Kp, Y=Kd, 切片=Q）
2. Bayesian Opt：使用 optuna TPE 貝氏最佳化尋找最大化超體積的 (Kp, Kd, Q)

最佳化目標：最大化超體積（Hypervolume）
"""

import os
import numpy as np
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
    deadband_values, norm_ref_rmse: float, norm_ref_cost: float,
    kf_r: float = 0.005,
    warmup: int = 30,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    warmup_prices_stock: np.ndarray = None,
    warmup_prices_bond: np.ndarray = None,
) -> dict:
    """
    單組 (Kp, Kd, Q) 的完整評估，供平行化使用。

    掃描多個 deadband → 得到 Smart Pilot frontier → 標準化後計算超體積。
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
        warmup_prices_stock=warmup_prices_stock,
        warmup_prices_bond=warmup_prices_bond,
    )

    norm_pts = [
        {"rmse": p["rmse"] / norm_ref_rmse, "cost": p["ann_cost"] / norm_ref_cost}
        for p in pareto["smart_pilot"]
        if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0
    ]
    hv = calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})

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
    norm_ref_rmse: float = 0.08,
    norm_ref_cost: float = 0.0004,
    warmup_prices_stock: np.ndarray = None,
    warmup_prices_bond: np.ndarray = None,
) -> List[dict]:
    """
    Grid Search：掃描所有 (Kp, Kd, Q) 組合，每組計算標準化超體積。
    使用 joblib 平行化加速。

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
            deadband_values, norm_ref_rmse, norm_ref_cost,
            kf_r, warmup, d_clip, output_clip,
            warmup_prices_stock, warmup_prices_bond,
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


def run_grid_search_with_progress(
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
    norm_ref_rmse: float = 0.08,
    norm_ref_cost: float = 0.0004,
    warmup_prices_stock: np.ndarray = None,
    warmup_prices_bond: np.ndarray = None,
    progress_bar=None,
    status_text=None,
    total_tasks: int = None,
) -> List[dict]:
    """
    帶進度條的 Grid Search。
    由於 joblib 平行化不支援直接回呼 Streamlit，
    改用批次執行（每批 n_jobs 個任務），每批完成後更新進度。
    """
    if kp_range is None:
        kp_range = np.linspace(0.1, 1.0, 10).tolist()
    if kd_range is None:
        kd_range = np.linspace(0.1, 1.0, 10).tolist()
    if q_values is None:
        q_values = [0.0001, 0.001, 0.01]
    if deadband_values is None:
        deadband_values = np.linspace(0.005, 0.10, 15).tolist()

    tasks = [
        (kp, kd, q)
        for q in q_values
        for kp in kp_range
        for kd in kd_range
    ]

    if total_tasks is None:
        total_tasks = len(tasks)

    # 批次大小：每批至少 1，最多 n_jobs * 2（讓進度更新頻繁）
    actual_n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    batch_size = max(1, actual_n_jobs * 2)

    all_results = []
    completed = 0

    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start: batch_start + batch_size]

        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_single_grid_run)(
                kp, kd, q,
                rets_stock, rets_bond, prices_stock, prices_bond,
                dates, target_w, fee_rate,
                deadband_values, norm_ref_rmse, norm_ref_cost,
                kf_r, warmup, d_clip, output_clip,
                warmup_prices_stock, warmup_prices_bond,
            )
            for kp, kd, q in batch
        )
        all_results.extend(batch_results)
        completed += len(batch)

        # 更新進度
        if progress_bar is not None:
            progress_bar.progress(min(completed / total_tasks, 1.0))
        if status_text is not None:
            status_text.text(f"Grid Search 進度：{completed}/{total_tasks}")

    return all_results


# ─────────────────────────────────────────
# Bayesian Optimization（optuna TPE）
# ─────────────────────────────────────────


def run_bayesian_opt(
    rets_stock: np.ndarray,
    rets_bond: np.ndarray,
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    deadband_values: list = None,
    kf_r: float = 0.005,
    warmup: int = 20,
    warmup_prices_stock: np.ndarray = None,
    warmup_prices_bond: np.ndarray = None,
    ref_rmse: float = 0.08,
    ref_cost: float = 0.0004,
    kp_min: float = 0.01, kp_max: float = 5.0,
    kd_min: float = 0.01, kd_max: float = 5.0,
    q_min: float = 1e-5,  q_max: float = 1.0,
    n_trials: int = 50,
    n_jobs: int = 1,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    progress_bar=None,
    status_text=None,
) -> dict:
    """
    使用 optuna TPE 貝氏最佳化尋找最大化超體積的 (Kp, Kd, Q)。

    使用固定標準化參考點（ref_rmse, ref_cost），標準化後參考點為 (1, 1)。
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from core.benchmark import scan_pareto_frontier

    def objective(trial):
        kp    = trial.suggest_float("kp", kp_min, kp_max)
        kd    = trial.suggest_float("kd", kd_min, kd_max)
        log_q = trial.suggest_float("log_q", np.log(q_min), np.log(q_max))
        q     = np.exp(log_q)

        pareto = scan_pareto_frontier(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=target_w, fee_rate=fee_rate,
            kf_q=q, kf_r=kf_r, kp=kp, kd=kd,
            deadband_values=deadband_values,
            warmup=warmup,
            warmup_prices_stock=warmup_prices_stock,
            warmup_prices_bond=warmup_prices_bond,
            d_clip=d_clip, output_clip=output_clip,
        )

        sp_pts = pareto["smart_pilot"]
        if not sp_pts:
            return 0.0

        norm_pts = [
            {"rmse": p["rmse"] / ref_rmse, "cost": p["ann_cost"] / ref_cost}
            for p in sp_pts
        ]
        hv = calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})
        return hv

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    for i in range(n_trials):
        study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=False)
        completed = len(study.trials)
        if progress_bar is not None:
            progress_bar.progress(min(completed / n_trials, 1.0))
        if status_text is not None:
            best_so_far = study.best_value if study.best_value is not None else 0.0
            status_text.text(
                f"貝氏最佳化進度：{completed}/{n_trials}  "
                f"目前最佳 HV={best_so_far:.6f}"
            )
    best_params = study.best_params
    best_kp = best_params["kp"]
    best_kd = best_params["kd"]
    best_q  = np.exp(best_params["log_q"])
    best_hv = study.best_value

    # 用最佳參數跑完整 Pareto frontier
    final_pareto = scan_pareto_frontier(
        rets_stock, rets_bond, prices_stock, prices_bond, dates,
        target_w=target_w, fee_rate=fee_rate,
        kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
        deadband_values=deadband_values,
        warmup=warmup,
        warmup_prices_stock=warmup_prices_stock,
        warmup_prices_bond=warmup_prices_bond,
        d_clip=d_clip, output_clip=output_clip,
    )

    return {
        "kp": best_kp,
        "kd": best_kd,
        "q": best_q,
        "hypervolume": best_hv,
        "hypervolume_x10000": best_hv * 10000,
        "n_trials": n_trials,
        "pareto": final_pareto,
    }


# ─────────────────────────────────────────
# 超體積指標（Hypervolume Indicator）
# ─────────────────────────────────────────


def calc_hypervolume(
    frontier_points: List[dict],
    reference_point: dict,
) -> float:
    """
    計算 Pareto Frontier 的超體積指標（勒貝格測度）。

    面積越大 → 策略支配的空間越大 → 策略越有價值。
    注意：這裡 RMSE 和 Cost 越小越好，所以超體積是從參考點往左下計算。

    Args:
        frontier_points: [{"rmse": float, "cost": float}, ...]
                         必須已按 rmse 排序（從小到大）
        reference_point: {"rmse": float, "cost": float}
                         右上角參考點，使用固定標準化參考點 {"rmse": 1.0, "cost": 1.0}

    Returns:
        hypervolume: float（超體積面積）
    """
    if not frontier_points:
        return 0.0

    points = sorted(frontier_points, key=lambda p: p["rmse"])

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


def compare_hypervolumes(
    pareto_data: dict,
    norm_ref_rmse: float = 0.08,
    norm_ref_cost: float = 0.0004,
) -> dict:
    """
    計算三個策略的標準化超體積並比較。

    Args:
        pareto_data: scan_pareto_frontier() 的回傳值
        norm_ref_rmse: RMSE 標準化參考上限（預設 0.08 = 8%）
        norm_ref_cost: Cost 標準化參考上限（預設 0.0004 = 0.04%/年）

    Returns:
        {
            "smart_pilot": float,
            "threshold_only": float,
            "time_and_threshold": float,
            "winner": str,
        }
    """
    norm_ref = {"rmse": 1.0, "cost": 1.0}

    def _norm_hv(pts):
        norm_pts = [
            {"rmse": p["rmse"] / norm_ref_rmse, "cost": p["ann_cost"] / norm_ref_cost}
            for p in pts
            if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0
        ]
        return calc_hypervolume(norm_pts, norm_ref)

    hv_sp  = _norm_hv(pareto_data["smart_pilot"])
    hv_to  = _norm_hv(pareto_data["threshold_only"])
    hv_tat = _norm_hv(pareto_data["time_and_threshold"])

    scores = {"smart_pilot": hv_sp, "threshold_only": hv_to, "time_and_threshold": hv_tat}
    winner = max(scores, key=scores.get)

    return {
        "smart_pilot": hv_sp,
        "threshold_only": hv_to,
        "time_and_threshold": hv_tat,
        "winner": winner,
    }
