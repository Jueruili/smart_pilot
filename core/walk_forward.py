"""
Walk-Forward Analysis 滾動窗口分析模組
每輪流程：
  IS（樣本內）：is_years 年，用貝氏最佳化找最佳 (Kp, Kd, Q)，最大化標準化 HV
  OOS（樣本外）：oos_years 年，套用 IS 最佳參數，三策略直接回測不再最佳化
  窗口步進：oos_years 年
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from core.benchmark import (
    run_smart_pilot,
    run_threshold_only,
    run_time_and_threshold,
    scan_pareto_frontier,
)
from core.optimizer import calc_hypervolume, run_bayesian_opt


def run_walk_forward(
    prices_stock: np.ndarray,
    prices_bond: np.ndarray,
    dates: list,
    warmup_prices_stock: np.ndarray,
    warmup_prices_bond: np.ndarray,
    target_w: float = 0.6,
    fee_rate: float = 0.003,
    kf_r: float = 0.005,
    d_clip: float = 0.15,
    output_clip: float = 0.2,
    is_years: int = 5,
    oos_years: int = 2,
    step_years: int = 1,
    n_trials: int = 50,
    deadband_values: list = None,
    norm_ref_rmse: float = 0.08,
    norm_ref_cost: float = 0.0004,
    kp_min: float = 0.01, kp_max: float = 5.0,
    kd_min: float = 0.01, kd_max: float = 5.0,
    q_min: float = 1e-5,  q_max: float = 1.0,
    n_jobs: int = 1,
    progress_bar=None,
    status_text=None,
) -> dict:
    """
    執行 Walk-Forward 滾動窗口分析。

    Returns:
        {
            "rounds": list of round result dicts,
            "is_years": int,
            "oos_years": int,
        }
    """
    is_days   = is_years   * 252
    oos_days  = oos_years  * 252
    step_days = step_years * 252
    total_days = len(dates)

    if deadband_values is None:
        deadband_values = np.linspace(0.005, 0.10, 15).tolist()

    # 預先計算總輪數（供進度條使用）
    n_rounds = max(0, (total_days - is_days - oos_days) // step_days + 1)

    rounds = []
    start_idx = 0
    round_num = 1

    while start_idx + is_days + oos_days <= total_days:
        is_end_idx  = start_idx + is_days
        oos_end_idx = is_end_idx + oos_days

        # 切分資料
        is_dates   = dates[start_idx:is_end_idx]
        oos_dates  = dates[is_end_idx:oos_end_idx]

        is_prices_stock  = prices_stock[start_idx:is_end_idx]
        is_prices_bond   = prices_bond[start_idx:is_end_idx]
        oos_prices_stock = prices_stock[is_end_idx:oos_end_idx]
        oos_prices_bond  = prices_bond[is_end_idx:oos_end_idx]

        # 計算報酬率（捨棄第一個元素以對齊）
        is_rets_stock  = np.diff(is_prices_stock)  / is_prices_stock[:-1]
        is_rets_bond   = np.diff(is_prices_bond)   / is_prices_bond[:-1]
        oos_rets_stock = np.diff(oos_prices_stock) / oos_prices_stock[:-1]
        oos_rets_bond  = np.diff(oos_prices_bond)  / oos_prices_bond[:-1]

        # IS 暖機資料：第一輪用傳入的 warmup_prices，之後用前一段資料的最後 20 天
        if round_num == 1:
            is_warmup_stock = warmup_prices_stock
            is_warmup_bond  = warmup_prices_bond
        else:
            pre_start = max(0, start_idx - 20)
            is_warmup_stock = prices_stock[pre_start:start_idx]
            is_warmup_bond  = prices_bond[pre_start:start_idx]

        # OOS 暖機資料：用 IS 最後 20 天
        oos_warmup_stock = is_prices_stock[-20:]
        oos_warmup_bond  = is_prices_bond[-20:]

        print(f"[WalkForward] Round {round_num}: "
              f"IS {is_dates[0].strftime('%Y/%m/%d')}~{is_dates[-1].strftime('%Y/%m/%d')}, "
              f"OOS {oos_dates[0].strftime('%Y/%m/%d')}~{oos_dates[-1].strftime('%Y/%m/%d')}")

        # ── IS：貝氏最佳化 ──
        is_result = run_bayesian_opt(
            is_rets_stock, is_rets_bond,
            is_prices_stock[1:], is_prices_bond[1:],
            is_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            deadband_values=deadband_values,
            kf_r=kf_r,
            warmup=len(is_warmup_stock),
            warmup_prices_stock=is_warmup_stock,
            warmup_prices_bond=is_warmup_bond,
            ref_rmse=norm_ref_rmse,
            ref_cost=norm_ref_cost,
            kp_min=kp_min, kp_max=kp_max,
            kd_min=kd_min, kd_max=kd_max,
            q_min=q_min,   q_max=q_max,
            n_trials=n_trials,
            n_jobs=n_jobs,
            d_clip=d_clip,
            output_clip=output_clip,
        )
        best_kp = is_result["kp"]
        best_kd = is_result["kd"]
        best_q  = is_result["q"]
        is_hv   = is_result["hypervolume"]

        print(f"[WalkForward] Round {round_num} IS best: "
              f"Kp={best_kp:.3f}, Kd={best_kd:.3f}, Q={best_q:.6f}, HV={is_hv:.6f}")

        # ── OOS：三策略回測 ──

        # 找最佳參數（對應 HV 最高的點）
        def _best_param(pts, param_key):
            best_hv, best_val = -1.0, pts[0][param_key]
            for p in pts:
                if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0:
                    hv = calc_hypervolume(
                        [{"rmse": p["rmse"] / norm_ref_rmse, "cost": p["ann_cost"] / norm_ref_cost}],
                        {"rmse": 1.0, "cost": 1.0}
                    )
                    if hv > best_hv:
                        best_hv, best_val = hv, p[param_key]
            return best_val

        # Smart Pilot：掃 deadband 取整條 Pareto，計算 OOS HV
        oos_sp_pareto = scan_pareto_frontier(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:],
            oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband_values=deadband_values,
            warmup=0,
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
            d_clip=d_clip, output_clip=output_clip,
        )
        sp_pts = oos_sp_pareto["smart_pilot"]
        norm_sp_pts = [
            {"rmse": p["rmse"] / norm_ref_rmse, "cost": p["ann_cost"] / norm_ref_cost}
            for p in sp_pts
            if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0
        ]
        oos_sp_hv = calc_hypervolume(norm_sp_pts, {"rmse": 1.0, "cost": 1.0})

        # Smart Pilot：用最佳 deadband 跑單點取績效指標
        best_sp_db = _best_param(sp_pts, "deadband")
        oos_sp = run_smart_pilot(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:], oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband=best_sp_db,
            warmup=0,
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
            d_clip=d_clip, output_clip=output_clip,
        )

        # Threshold-only：掃 Pareto 計算 HV（warmup=0 已由 scan_pareto_frontier 內部處理）
        oos_to_pareto = scan_pareto_frontier(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:],
            oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband_values=deadband_values,
            warmup=0,
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
        )
        to_pts = oos_to_pareto["threshold_only"]
        norm_to_pts = [
            {"rmse": p["rmse"] / norm_ref_rmse, "cost": p["ann_cost"] / norm_ref_cost}
            for p in to_pts
            if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0
        ]
        oos_to_hv = calc_hypervolume(norm_to_pts, {"rmse": 1.0, "cost": 1.0})

        # Threshold-only：用最佳 tolerance 跑單點取績效指標
        best_to_tol = _best_param(to_pts, "tolerance")
        oos_to = run_threshold_only(
            oos_rets_stock, oos_rets_bond, oos_dates[1:],
            target_w=target_w, drift_tolerance=best_to_tol,
            fee_rate=fee_rate, warmup=0,
        )

        # Time-and-threshold：掃 Pareto 計算 HV（warmup=0 已由 scan_pareto_frontier 內部處理）
        oos_tat_pareto = scan_pareto_frontier(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:],
            oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband_values=deadband_values,
            warmup=0,
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
        )
        tat_pts = oos_tat_pareto["time_and_threshold"]
        norm_tat_pts = [
            {"rmse": p["rmse"] / norm_ref_rmse, "cost": p["ann_cost"] / norm_ref_cost}
            for p in tat_pts
            if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0
        ]
        oos_tat_hv = calc_hypervolume(norm_tat_pts, {"rmse": 1.0, "cost": 1.0})

        # Time-and-threshold：用最佳 threshold 跑單點取績效指標
        best_tat_thresh = _best_param(tat_pts, "threshold")
        oos_tat = run_time_and_threshold(
            oos_rets_stock, oos_rets_bond, oos_dates[1:],
            target_w=target_w, threshold=best_tat_thresh,
            fee_rate=fee_rate, warmup=0,
        )

        rounds.append({
            "round":     round_num,
            "is_start":  is_dates[0],
            "is_end":    is_dates[-1],
            "oos_start": oos_dates[0],
            "oos_end":   oos_dates[-1],
            "best_kp":   best_kp,
            "best_kd":   best_kd,
            "best_q":    best_q,
            "is_hv":     is_hv,
            "oos_sp_hv":  oos_sp_hv,
            "oos_to_hv":  oos_to_hv,
            "oos_tat_hv": oos_tat_hv,
            "oos_sp":    oos_sp,
            "oos_to":    oos_to,
            "oos_tat":   oos_tat,
        })

        # 每輪結束後更新進度
        if progress_bar is not None:
            progress_bar.progress((round_num) / n_rounds)
        if status_text is not None:
            status_text.text(
                f"Walk-Forward 進度：{round_num}/{n_rounds} 輪完成"
                f"（IS: {is_dates[0].strftime('%Y/%m/%d')} ~ {is_dates[-1].strftime('%Y/%m/%d')}）"
            )

        start_idx += step_days
        round_num += 1

    return {
        "rounds":     rounds,
        "is_years":   is_years,
        "oos_years":  oos_years,
        "step_years": step_years,
    }
