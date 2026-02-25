"""
Walk-Forward Analysis 滾動窗口分析模組

每輪流程：
  IS（樣本內）：is_years 年，用貝氏最佳化找最佳 (Kp, Kd, Q)，最大化動態 HV
  OOS（樣本外）：oos_years 年，套用 IS 最佳參數，三策略直接回測不再最佳化
  窗口步進：step_years 年（預設 1 年）
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
    norm_ref_cost: float = 0.0008,
    ref_multiplier: float = 1.1,
    kp_min: float = 0.01, kp_max: float = 5.0,
    kd_min: float = 0.01, kd_max: float = 5.0,
    q_min: float = 1e-5,  q_max: float = 1.0,
    n_jobs: int = 1,
) -> dict:
    """
    執行 Walk-Forward 滾動窗口分析。

    Returns:
        {
            "rounds": list of round result dicts,
            "is_years": int,
            "oos_years": int,
            "step_years": int,
        }
    """
    is_days   = is_years  * 252
    oos_days  = oos_years * 252
    step_days = step_years * 252
    total_days = len(dates)

    if deadband_values is None:
        deadband_values = np.linspace(0.005, 0.10, 15).tolist()

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

        # 計算報酬率
        is_rets_stock  = np.diff(is_prices_stock)  / is_prices_stock[:-1]
        is_rets_bond   = np.diff(is_prices_bond)   / is_prices_bond[:-1]
        oos_rets_stock = np.diff(oos_prices_stock) / oos_prices_stock[:-1]
        oos_rets_bond  = np.diff(oos_prices_bond)  / oos_prices_bond[:-1]

        # IS 暖機資料
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

        # ── OOS：三策略 Pareto 掃描（動態參考點）──

        oos_sp_pareto = scan_pareto_frontier(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:],
            oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband_values=deadband_values,
            warmup=len(oos_warmup_stock),
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
            d_clip=d_clip, output_clip=output_clip,
        )

        oos_to_pareto = scan_pareto_frontier(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:],
            oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband_values=deadband_values,
            warmup=len(oos_warmup_stock),
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
        )

        oos_tat_pareto = scan_pareto_frontier(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:],
            oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband_values=deadband_values,
            warmup=len(oos_warmup_stock),
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
        )

        # 動態參考點：三策略所有點合併後取最差 × ref_multiplier
        sp_pts  = oos_sp_pareto["smart_pilot"]
        to_pts  = oos_to_pareto["threshold_only"]
        tat_pts = oos_tat_pareto["time_and_threshold"]

        all_oos_pts = (
            [{"rmse": p["rmse"], "cost": p["ann_cost"]} for p in sp_pts] +
            [{"rmse": p["rmse"], "cost": p["ann_cost"]} for p in to_pts] +
            [{"rmse": p["rmse"], "cost": p["ann_cost"]} for p in tat_pts]
        )
        dyn_ref_rmse = max(p["rmse"] for p in all_oos_pts) * ref_multiplier
        dyn_ref_cost = max(p["cost"] for p in all_oos_pts) * ref_multiplier
        dyn_ref = {"rmse": dyn_ref_rmse, "cost": dyn_ref_cost}

        oos_sp_hv  = calc_hypervolume(
            [{"rmse": p["rmse"], "cost": p["ann_cost"]} for p in sp_pts], dyn_ref)
        oos_to_hv  = calc_hypervolume(
            [{"rmse": p["rmse"], "cost": p["ann_cost"]} for p in to_pts], dyn_ref)
        oos_tat_hv = calc_hypervolume(
            [{"rmse": p["rmse"], "cost": p["ann_cost"]} for p in tat_pts], dyn_ref)

        # 單點績效指標
        mid_db = deadband_values[len(deadband_values) // 2]
        oos_sp = run_smart_pilot(
            oos_rets_stock, oos_rets_bond,
            oos_prices_stock[1:], oos_prices_bond[1:], oos_dates[1:],
            target_w=target_w, fee_rate=fee_rate,
            kf_q=best_q, kf_r=kf_r, kp=best_kp, kd=best_kd,
            deadband=mid_db,
            warmup=len(oos_warmup_stock),
            warmup_prices_stock=oos_warmup_stock,
            warmup_prices_bond=oos_warmup_bond,
            d_clip=d_clip, output_clip=output_clip,
        )

        oos_to = run_threshold_only(
            oos_rets_stock, oos_rets_bond, oos_dates[1:],
            target_w=target_w, drift_tolerance=0.05,
            fee_rate=fee_rate, warmup=1,
        )

        oos_tat = run_time_and_threshold(
            oos_rets_stock, oos_rets_bond, oos_dates[1:],
            target_w=target_w, threshold=0.05,
            fee_rate=fee_rate, warmup=1,
        )

        rounds.append({
            "round":        round_num,
            "is_start":     is_dates[0],
            "is_end":       is_dates[-1],
            "oos_start":    oos_dates[0],
            "oos_end":      oos_dates[-1],
            "best_kp":      best_kp,
            "best_kd":      best_kd,
            "best_q":       best_q,
            "is_hv":        is_hv,
            "oos_sp_hv":    oos_sp_hv,
            "oos_to_hv":    oos_to_hv,
            "oos_tat_hv":   oos_tat_hv,
            "dyn_ref_rmse": dyn_ref_rmse,
            "dyn_ref_cost": dyn_ref_cost,
            "oos_sp":       oos_sp,
            "oos_to":       oos_to,
            "oos_tat":      oos_tat,
        })

        start_idx += step_days
        round_num += 1

    return {
        "rounds":     rounds,
        "is_years":   is_years,
        "oos_years":  oos_years,
        "step_years": step_years,
    }
