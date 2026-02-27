"""
Walk-Forward Monte Carlo 驗證模組
"""
import math
import traceback
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from core.benchmark import run_smart_pilot, run_threshold_only, run_time_and_threshold
from core.optimizer import calc_hypervolume


def generate_block_bootstrap_matrix(
    hist_prices_stock: np.ndarray,
    hist_prices_bond: np.ndarray,
    n_paths: int,
    n_days_simulate: int,
    block_size: int = 20,
    random_seed: int = 42,
) -> dict:
    hist_rets_stock = np.diff(hist_prices_stock) / hist_prices_stock[:-1]
    hist_rets_bond  = np.diff(hist_prices_bond)  / hist_prices_bond[:-1]

    total_blocks = len(hist_rets_stock) - block_size + 1
    if total_blocks <= 0:
        raise ValueError(
            f"歷史資料不足：hist_rets 長度={len(hist_rets_stock)}，"
            f"block_size={block_size}，需要至少 {block_size + 1} 筆歷史價格。"
        )

    n_blocks_needed = math.ceil(n_days_simulate / block_size)
    rng = np.random.default_rng(random_seed)
    block_starts = rng.integers(0, total_blocks, size=(n_paths, n_blocks_needed))

    offsets = np.arange(block_size)
    indices = (block_starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
    indices = indices[:, :n_days_simulate]

    sim_rets_stock = hist_rets_stock[indices]
    sim_rets_bond  = hist_rets_bond[indices]

    p0_stock = hist_prices_stock[-1]
    p0_bond  = hist_prices_bond[-1]
    sim_prices_stock = p0_stock * np.cumprod(1 + sim_rets_stock, axis=1)
    sim_prices_bond  = p0_bond  * np.cumprod(1 + sim_rets_bond,  axis=1)

    return {
        "sim_rets_stock":   sim_rets_stock,
        "sim_rets_bond":    sim_rets_bond,
        "sim_prices_stock": sim_prices_stock,
        "sim_prices_bond":  sim_prices_bond,
        "p0_stock": p0_stock,
        "p0_bond":  p0_bond,
    }


def run_wf_monte_carlo(
    hist_prices_stock: np.ndarray,
    hist_prices_bond: np.ndarray,
    n_paths: int,
    n_days_simulate: int,
    block_size: int,
    avg_kp: float,
    avg_kd: float,
    avg_q: float,
    kf_r: float,
    target_w: float,
    fee_rate: float,
    d_clip: float,
    output_clip: float,
    warmup_prices_stock: np.ndarray,
    warmup_prices_bond: np.ndarray,
    deadband_values: list,
    tol_values: list,
    norm_ref_rmse: float,
    norm_ref_cost: float,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    # Step 1：生成所有路徑
    matrix = generate_block_bootstrap_matrix(
        hist_prices_stock, hist_prices_bond,
        n_paths, n_days_simulate, block_size, random_seed,
    )

    n_years = n_days_simulate / 252.0
    sim_dates = pd.date_range("2019-01-01", periods=n_days_simulate, freq="B").tolist()

    # Step 2：單條路徑回測函式（縮排在 run_wf_monte_carlo 內，能存取閉包變數）
    def _run_one_path(i):
        try:
            rets_s   = matrix["sim_rets_stock"][i]
            rets_b   = matrix["sim_rets_bond"][i]
            prices_s = matrix["sim_prices_stock"][i]
            prices_b = matrix["sim_prices_bond"][i]

            def _hv(r):
                ann_cost = r["cost"] / n_years
                if r["rmse"] / norm_ref_rmse <= 1.0 and ann_cost / norm_ref_cost <= 1.0:
                    norm_pts = [{"rmse": r["rmse"] / norm_ref_rmse,
                                 "cost": ann_cost / norm_ref_cost}]
                else:
                    norm_pts = []
                return calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})

            def _ann_ret(r):
                nav_final = r["nav_list"][-1] if r["nav_list"] else 1.0
                return float(nav_final ** (252.0 / n_days_simulate) - 1)

            def _ann_vol(r):
                return float(r["metrics"]["ann_wealth_vol"])

            # Smart Pilot
            best_sp_hv, best_sp_r = -1.0, None
            for db in deadband_values:
                r = run_smart_pilot(
                    rets_s, rets_b, prices_s, prices_b, sim_dates,
                    target_w=target_w, fee_rate=fee_rate,
                    kf_q=avg_q, kf_r=kf_r,
                    kp=avg_kp, kd=avg_kd,
                    deadband=float(db), warmup=0,
                    warmup_prices_stock=warmup_prices_stock,
                    warmup_prices_bond=warmup_prices_bond,
                    d_clip=d_clip, output_clip=output_clip,
                )
                hv = _hv(r)
                if hv > best_sp_hv:
                    best_sp_hv, best_sp_r = hv, r

            # Threshold-only
            best_to_hv, best_to_r = -1.0, None
            for tol in tol_values:
                r = run_threshold_only(
                    rets_s, rets_b, sim_dates,
                    target_w=target_w, drift_tolerance=float(tol),
                    fee_rate=fee_rate, warmup=0,
                )
                hv = _hv(r)
                if hv > best_to_hv:
                    best_to_hv, best_to_r = hv, r

            # Time-and-threshold
            best_tat_hv, best_tat_r = -1.0, None
            for tol in tol_values:
                r = run_time_and_threshold(
                    rets_s, rets_b, sim_dates,
                    target_w=target_w, fee_rate=fee_rate,
                    threshold=float(tol), warmup=0,
                )
                hv = _hv(r)
                if hv > best_tat_hv:
                    best_tat_hv, best_tat_r = hv, r

            return {
                "sp_hv":      best_sp_hv,
                "sp_ret":     _ann_ret(best_sp_r)  if best_sp_r  else 0.0,
                "sp_vol":     _ann_vol(best_sp_r)  if best_sp_r  else 0.0,
                "sp_trades":  best_sp_r["trade_count"] if best_sp_r  else 0,
                "to_hv":      best_to_hv,
                "to_ret":     _ann_ret(best_to_r)  if best_to_r  else 0.0,
                "to_vol":     _ann_vol(best_to_r)  if best_to_r  else 0.0,
                "to_trades":  best_to_r["trade_count"] if best_to_r  else 0,
                "tat_hv":     best_tat_hv,
                "tat_ret":    _ann_ret(best_tat_r) if best_tat_r else 0.0,
                "tat_vol":    _ann_vol(best_tat_r) if best_tat_r else 0.0,
                "tat_trades": best_tat_r["trade_count"] if best_tat_r else 0,
            }

        except Exception as e:
            print(f"\n=== path {i} FAILED ===")
            print(f"rets_s: len={len(rets_s)}, nan={np.isnan(rets_s).sum()}, inf={np.isinf(rets_s).sum()}")
            print(f"warmup_stock len={len(warmup_prices_stock)}, warmup_bond len={len(warmup_prices_bond)}")
            traceback.print_exc()
            raise

    # Step 3：執行
    path_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_path)(i) for i in range(n_paths)
    )

    # Step 4：整理回傳
    keys = [
        "sp_hv",  "sp_ret",  "sp_vol",  "sp_trades",
        "to_hv",  "to_ret",  "to_vol",  "to_trades",
        "tat_hv", "tat_ret", "tat_vol", "tat_trades",
    ]
    return {k: np.array([r[k] for r in path_results]) for k in keys}