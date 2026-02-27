"""
Smart Pilot - 投資組合再平衡系統

Streamlit 應用程式（v3.0）

四個分頁：
1. Pareto Frontier - 三策略帕雷托前線對比
2. Heatmap - 參數空間熱力圖
3. Rolling Window - 滾動窗口分析
4. Monte Carlo - 蒙地卡羅模擬

作者：Smart Pilot Team
版本：3.0.0
"""

import hashlib
import json
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime
from pathlib import Path

from data.data_loader import DataLoader
from core.benchmark import (
    run_smart_pilot, run_threshold_only, run_time_and_threshold,
    scan_pareto_frontier, get_metrics, calc_rmse
)
from core.optimizer import calc_hypervolume, compare_hypervolumes


# =============================================================================
# 共用工具函數
# =============================================================================
def make_params_hash(params: dict) -> str:
    """把參數 dict 轉成 8 碼 hash，用於比對快取是否有效"""
    params_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def _serialize_rounds(rounds: list) -> list:
    """序列化 rounds（把 datetime.date 轉成字串）"""
    result = []
    for r in rounds:
        row = {k: v for k, v in r.items()}
        for date_key in ["is_start", "is_end", "oos_start", "oos_end"]:
            if date_key in row:
                row[date_key] = row[date_key].strftime("%Y-%m-%d")
        result.append(row)
    return result


def _deserialize_rounds(rounds: list) -> list:
    """反序列化 rounds（把字串轉回 datetime.date）"""
    result = []
    for r in rounds:
        row = {k: v for k, v in r.items()}
        for date_key in ["is_start", "is_end", "oos_start", "oos_end"]:
            if date_key in row:
                row[date_key] = datetime.strptime(row[date_key], "%Y-%m-%d").date()
        result.append(row)
    return result


# =============================================================================
# 設定
# =============================================================================
st.set_page_config(
    page_title="Smart Pilot - 投資組合再平衡系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 快取路徑
GRID_CACHE_PATH = "data/cache/grid_search_results.json"


# =============================================================================
# 快取函數
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: list, start_date: str, end_date: str, warmup_days: int = 20) -> dict:
    """載入並處理資料（含暖機資料）

    Args:
        tickers: 股票代碼列表
        start_date: 回測開始日期
        end_date: 回測結束日期
        warmup_days: 暖機天數

    Returns:
        dict: {
            "backtest_data": DataFrame,
            "warmup_data": DataFrame,
            "actual_warmup_days": int,
        }
    """
    loader = DataLoader()
    result = loader.load_and_process(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        warmup_days=warmup_days,
    )
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def load_grid_cache() -> dict:
    """載入 Grid Search 快取"""
    if os.path.exists(GRID_CACHE_PATH):
        with open(GRID_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =============================================================================
# 側邊欄
# =============================================================================
def render_sidebar() -> dict:
    """渲染側邊欄並返回參數"""

    # =========================================================================
    # session_state 初始化（只在 key 不存在時設預設值）
    # =========================================================================
    defaults = {
        "ticker1": "VTI", "ticker2": "BND",
        "target_w": 0.6, "start_date": date(2000, 1, 1),
        "end_date": date(2024, 1, 1), "fee_rate": 0.003,
        "warmup": 20, "kf_r": 0.005, "d_clip": 0.15, "output_clip": 0.2,
        "norm_ref_rmse_pct": 8.0, "norm_ref_cost_pct": 0.04,
        "bayes_kp_min": 0.01, "bayes_kp_max": 5.0,
        "bayes_kd_min": 0.01, "bayes_kd_max": 5.0,
        "bayes_q_min": 0.00001, "bayes_q_max": 1.0,
        "bayes_n_trials": 50,
        "bayes_db_min": 0.005, "bayes_db_max": 0.10, "bayes_db_points": 15,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # =========================================================================
    # 區塊 1：標的設定
    # =========================================================================
    st.sidebar.header("📈 標的設定")
    with st.sidebar.form("form_target"):
        ticker1 = st.text_input("股票標的", value=st.session_state["ticker1"])
        ticker2 = st.text_input("債券標的", value=st.session_state["ticker2"])
        target_w = st.slider(
            "目標股票比例",
            min_value=0.1, max_value=0.9,
            value=float(st.session_state["target_w"]),
            step=0.05, format="%.2f"
        )
        start_date = st.date_input("回測開始日期", value=st.session_state["start_date"])
        end_date = st.date_input("回測結束日期", value=st.session_state["end_date"])
        fee_rate = st.number_input(
            "手續費率",
            value=float(st.session_state["fee_rate"]),
            step=0.001, format="%.3f"
        )
        submitted_target = st.form_submit_button("更改標的設定", use_container_width=True)
        if submitted_target:
            st.session_state.update({
                "ticker1": ticker1.upper().strip(),
                "ticker2": ticker2.upper().strip(),
                "target_w": target_w,
                "start_date": start_date,
                "end_date": end_date,
                "fee_rate": fee_rate,
            })

    # =========================================================================
    # 區塊 2：卡爾曼濾波器設定
    # =========================================================================
    st.sidebar.header("🔧 卡爾曼濾波器設定")
    with st.sidebar.form("form_kf"):
        warmup = st.number_input(
            "暖機天數",
            min_value=10, max_value=100,
            value=int(st.session_state["warmup"]),
            step=1,
            help="回測起始日前的暖機天數，使用前一年底的歷史資料初始化 KF，建議 20 天"
        )
        kf_r = st.number_input(
            "R 值（觀測雜訊）",
            value=float(st.session_state["kf_r"]),
            min_value=0.0001, step=0.001, format="%.4f",
            help="R 越大越平滑但反應越慢，建議 0.001~0.01"
        )
        d_clip = st.number_input(
            "D Clip",
            value=float(st.session_state["d_clip"]),
            min_value=0.01, step=0.01, format="%.2f",
            help="D 項裁切閾值，預設 0.15"
        )
        output_clip = st.number_input(
            "Output Clip",
            value=float(st.session_state["output_clip"]),
            min_value=0.01, step=0.01, format="%.2f",
            help="總輸出裁切閾值，預設 0.2"
        )
        submitted_kf = st.form_submit_button("更改卡爾曼濾波器設定", use_container_width=True)
        if submitted_kf:
            st.session_state.update({
                "warmup": warmup,
                "kf_r": kf_r,
                "d_clip": d_clip,
                "output_clip": output_clip,
            })

    # =========================================================================
    # 區塊 3：標準化參考點設定
    # =========================================================================
    st.sidebar.header("📐 標準化參考點設定")
    with st.sidebar.form("form_norm"):
        col1, col2 = st.columns(2)
        with col1:
            norm_ref_rmse_pct = st.number_input(
                "參考 RMSE (%)", min_value=0.01,
                value=float(st.session_state["norm_ref_rmse_pct"]),
                step=0.5, format="%.2f",
                help="RMSE 參考上限，例如 8 代表 8%"
            )
        with col2:
            norm_ref_cost_pct = st.number_input(
                "參考 Cost (%/年)", min_value=0.001,
                value=float(st.session_state["norm_ref_cost_pct"]),
                step=0.005, format="%.3f",
                help="Cost 參考上限，例如 0.04 代表 0.04%/年"
            )
        submitted_norm = st.form_submit_button("更改參考點設定", use_container_width=True)
        if submitted_norm:
            st.session_state.update({
                "norm_ref_rmse_pct": norm_ref_rmse_pct,
                "norm_ref_cost_pct": norm_ref_cost_pct,
            })

    # =========================================================================
    # 區塊 4：貝氏最佳化設定 + 執行
    # =========================================================================
    st.sidebar.header("🔍 貝氏最佳化設定")

    # n_jobs 放在 form 外面（立即生效）
    n_cores = os.cpu_count() or 1
    st.sidebar.write(f"你的電腦有 {n_cores} 個核心")
    n_jobs = st.sidebar.slider(
        "使用核心數",
        min_value=1,
        max_value=n_cores,
        value=max(1, n_cores - 1)
    )

    with st.sidebar.form("form_bayes"):
        col1, col2 = st.columns(2)
        with col1:
            bayes_kp_min = st.number_input(
                "Kp 下限", value=float(st.session_state["bayes_kp_min"]),
                min_value=0.001, step=0.01, format="%.3f"
            )
            bayes_kd_min = st.number_input(
                "Kd 下限", value=float(st.session_state["bayes_kd_min"]),
                min_value=0.001, step=0.01, format="%.3f"
            )
        with col2:
            bayes_kp_max = st.number_input(
                "Kp 上限", value=float(st.session_state["bayes_kp_max"]),
                min_value=0.1, step=0.5, format="%.1f"
            )
            bayes_kd_max = st.number_input(
                "Kd 上限", value=float(st.session_state["bayes_kd_max"]),
                min_value=0.1, step=0.5, format="%.1f"
            )

        col1, col2 = st.columns(2)
        with col1:
            bayes_q_min = st.number_input(
                "Q 下限", value=float(st.session_state["bayes_q_min"]),
                min_value=0.000001, format="%.5f"
            )
        with col2:
            bayes_q_max = st.number_input(
                "Q 上限", value=float(st.session_state["bayes_q_max"]),
                min_value=0.00001, format="%.4f"
            )

        st.markdown("**Deadband 掃描範圍**")
        col1, col2, col3 = st.columns(3)
        with col1:
            bayes_db_min = st.number_input(
                "Deadband 最小值", value=float(st.session_state.get("bayes_db_min", 0.005)),
                min_value=0.001, step=0.005, format="%.3f", key="bayes_db_min_input"
            )
        with col2:
            bayes_db_max = st.number_input(
                "Deadband 最大值", value=float(st.session_state.get("bayes_db_max", 0.10)),
                min_value=0.01, step=0.01, format="%.3f", key="bayes_db_max_input"
            )
        with col3:
            bayes_db_points = st.number_input(
                "Deadband 點數", value=int(st.session_state.get("bayes_db_points", 15)),
                min_value=5, step=5, key="bayes_db_points_input"
            )

        bayes_n_trials = st.number_input(
            "試驗次數 (trials)", value=int(st.session_state["bayes_n_trials"]),
            min_value=10, step=10
        )

        submitted_bayes = st.form_submit_button(
            "▶ 執行貝氏最佳化（約 3-8 分鐘）",
            type="primary", use_container_width=True
        )

        if submitted_bayes:
            # 1. 套用貝氏參數到 session_state
            st.session_state.update({
                "bayes_kp_min": bayes_kp_min,
                "bayes_kp_max": bayes_kp_max,
                "bayes_kd_min": bayes_kd_min,
                "bayes_kd_max": bayes_kd_max,
                "bayes_q_min":  bayes_q_min,
                "bayes_q_max":  bayes_q_max,
                "bayes_n_trials": bayes_n_trials,
                "bayes_db_min": bayes_db_min,
                "bayes_db_max": bayes_db_max,
                "bayes_db_points": bayes_db_points,
            })
            # 2. 從 session_state 讀所有參數
            s = st.session_state

            # 3. Hash 比對：參數未變則跳過計算
            bayes_hash_params = {
                "ticker1": s["ticker1"], "ticker2": s["ticker2"],
                "start_date": str(s["start_date"]), "end_date": str(s["end_date"]),
                "target_w": s["target_w"], "fee_rate": s["fee_rate"],
                "kf_r": s["kf_r"], "warmup": s["warmup"],
                "d_clip": s["d_clip"], "output_clip": s["output_clip"],
                "norm_ref_rmse_pct": s["norm_ref_rmse_pct"],
                "norm_ref_cost_pct": s["norm_ref_cost_pct"],
                "kp_min": s["bayes_kp_min"], "kp_max": s["bayes_kp_max"],
                "kd_min": s["bayes_kd_min"], "kd_max": s["bayes_kd_max"],
                "q_min": s["bayes_q_min"],   "q_max": s["bayes_q_max"],
                "n_trials": s["bayes_n_trials"],
            }
            current_hash = make_params_hash(bayes_hash_params)
            bayes_path = Path("data/cache/bayesian_opt_results.json")
            skip = False
            if bayes_path.exists():
                with open(bayes_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("params_hash") == current_hash:
                    st.sidebar.success("✅ 參數未變，使用上次貝氏最佳化結果")
                    skip = True

            if not skip:
                n_trials_int = int(s["bayes_n_trials"])

                # 4. 抓資料
                with st.spinner("載入資料..."):
                    result_data = load_data(
                        tickers=[s["ticker1"], s["ticker2"]],
                        start_date=str(s["start_date"]),
                        end_date=str(s["end_date"]),
                        warmup_days=int(s["warmup"]),
                    )
                    full_backtest   = result_data["backtest_data"]
                    wm_prices_stock = result_data["warmup_data"][s["ticker1"]].values
                    wm_prices_bond  = result_data["warmup_data"][s["ticker2"]].values
                    prices_stock_bt = full_backtest[s["ticker1"]].values
                    prices_bond_bt  = full_backtest[s["ticker2"]].values
                    dates_bt        = full_backtest.index.tolist()
                    rets_stock_bt   = np.diff(prices_stock_bt) / prices_stock_bt[:-1]
                    rets_bond_bt    = np.diff(prices_bond_bt)  / prices_bond_bt[:-1]
                    prices_stock_bt = prices_stock_bt[1:]
                    prices_bond_bt  = prices_bond_bt[1:]
                    dates_bt        = dates_bt[1:]

                # 5. 跑貝氏最佳化
                progress_bar = st.sidebar.progress(0)
                status_text  = st.sidebar.empty()
                status_text.text(f"貝氏最佳化進度：0/{n_trials_int}")

                t0 = time.time()
                from core.optimizer import run_bayesian_opt
                bayes_result = run_bayesian_opt(
                    rets_stock_bt, rets_bond_bt,
                    prices_stock_bt, prices_bond_bt, dates_bt,
                    target_w=s["target_w"],
                    fee_rate=s["fee_rate"],
                    deadband_values=np.linspace(
                        s["bayes_db_min"], s["bayes_db_max"], int(s["bayes_db_points"])
                    ).tolist(),
                    kf_r=s["kf_r"],
                    warmup=int(s["warmup"]),
                    warmup_prices_stock=wm_prices_stock,
                    warmup_prices_bond=wm_prices_bond,
                    ref_rmse=s["norm_ref_rmse_pct"] / 100,
                    ref_cost=s["norm_ref_cost_pct"] / 100,
                    kp_min=s["bayes_kp_min"], kp_max=s["bayes_kp_max"],
                    kd_min=s["bayes_kd_min"], kd_max=s["bayes_kd_max"],
                    q_min=s["bayes_q_min"],   q_max=s["bayes_q_max"],
                    n_trials=n_trials_int,
                    n_jobs=n_jobs,
                    d_clip=s["d_clip"],
                    output_clip=s["output_clip"],
                    progress_bar=progress_bar,
                    status_text=status_text,
                )
                elapsed = time.time() - t0

                # 6. 存 JSON（含 params_hash）
                progress_bar.progress(1.0)
                status_text.text(f"完成！{n_trials_int} 次試驗，耗時 {elapsed:.1f} 秒")

                bayes_path.parent.mkdir(parents=True, exist_ok=True)
                with open(bayes_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "kp": bayes_result["kp"],
                        "kd": bayes_result["kd"],
                        "q":  bayes_result["q"],
                        "hypervolume": bayes_result["hypervolume"],
                        "n_trials": bayes_result["n_trials"],
                        "params_hash": current_hash,
                    }, f, ensure_ascii=False, indent=2)

                # ── 貝氏最佳化三策略對比 ──
                n_years_bt = len(dates_bt) / 252.0
                ref_rmse_bt = s["norm_ref_rmse_pct"] / 100
                ref_cost_bt = s["norm_ref_cost_pct"] / 100
                bayes_db_values = np.linspace(
                    s["bayes_db_min"], s["bayes_db_max"], int(s["bayes_db_points"])
                ).tolist()
                bayes_tol_values = np.linspace(0.005, 0.15, len(bayes_db_values)).tolist()

                pareto_bt = scan_pareto_frontier(
                    rets_stock_bt, rets_bond_bt,
                    prices_stock_bt, prices_bond_bt, dates_bt,
                    target_w=s["target_w"], fee_rate=s["fee_rate"],
                    kf_q=bayes_result["q"], kf_r=s["kf_r"],
                    kp=bayes_result["kp"], kd=bayes_result["kd"],
                    deadband_values=bayes_db_values,
                    warmup=0,
                    warmup_prices_stock=wm_prices_stock,
                    warmup_prices_bond=wm_prices_bond,
                    d_clip=s["d_clip"], output_clip=s["output_clip"],
                )

                def _find_best_point(pts_list):
                    best_hv_p, best_idx_p = -1, 0
                    for i, p in enumerate(pts_list):
                        ann_cost_p = p["ann_cost"]
                        if p["rmse"] / ref_rmse_bt <= 1.0 and ann_cost_p / ref_cost_bt <= 1.0:
                            norm_pts_p = [{"rmse": p["rmse"] / ref_rmse_bt,
                                           "cost": ann_cost_p / ref_cost_bt}]
                        else:
                            norm_pts_p = []
                        hv_p = calc_hypervolume(norm_pts_p, {"rmse": 1.0, "cost": 1.0})
                        if hv_p > best_hv_p:
                            best_hv_p, best_idx_p = hv_p, i
                    return best_idx_p

                sp_idx = _find_best_point(pareto_bt["smart_pilot"])
                best_db_bayes = pareto_bt["smart_pilot"][sp_idx]["deadband"]
                best_sp = run_smart_pilot(
                    rets_stock_bt, rets_bond_bt,
                    prices_stock_bt, prices_bond_bt, dates_bt,
                    target_w=s["target_w"], fee_rate=s["fee_rate"],
                    kf_q=bayes_result["q"], kf_r=s["kf_r"],
                    kp=bayes_result["kp"], kd=bayes_result["kd"],
                    deadband=float(best_db_bayes),
                    warmup=0,
                    warmup_prices_stock=wm_prices_stock,
                    warmup_prices_bond=wm_prices_bond,
                    d_clip=s["d_clip"], output_clip=s["output_clip"],
                )

                to_idx = _find_best_point(pareto_bt["threshold_only"])
                best_tol = pareto_bt["threshold_only"][to_idx]["tolerance"]
                best_to = run_threshold_only(
                    rets_stock_bt, rets_bond_bt, dates_bt,
                    target_w=s["target_w"], drift_tolerance=float(best_tol),
                    fee_rate=s["fee_rate"], warmup=0,
                )

                tat_idx = _find_best_point(pareto_bt["time_and_threshold"])
                best_thresh = pareto_bt["time_and_threshold"][tat_idx]["threshold"]
                best_tat = run_time_and_threshold(
                    rets_stock_bt, rets_bond_bt, dates_bt,
                    target_w=s["target_w"], fee_rate=s["fee_rate"],
                    threshold=float(best_thresh), warmup=0,
                )

                st.session_state["bayes_backtest_results"] = {
                    "sp":  best_sp,
                    "to":  best_to,
                    "tat": best_tat,
                    "best_deadband": best_db_bayes,
                    "best_tol":    best_tol,
                    "best_thresh": best_thresh,
                    "n_years": n_years_bt,
                    "kp": bayes_result["kp"],
                    "kd": bayes_result["kd"],
                    "q":  bayes_result["q"],
                }

                st.sidebar.success(
                    f"貝氏最佳化完成！耗時 {elapsed:.1f} 秒\n"
                    f"最佳：Kp={bayes_result['kp']:.3f}, "
                    f"Kd={bayes_result['kd']:.3f}, "
                    f"Q={bayes_result['q']:.6f}\n"
                    f"HV={bayes_result['hypervolume']:.6f}"
                )

    # =========================================================================
    # 區塊 5：快取管理
    # =========================================================================
    with st.sidebar.expander("🗂️ 快取管理", expanded=False):
        st.write("**股票價格快取（CSV）**")
        cache_dir = Path("data/cache")
        csv_files = list(cache_dir.glob("*.csv")) if cache_dir.exists() else []
        if csv_files:
            csv_options = {
                f.name + f" ({f.stat().st_size // 1024} KB)": f
                for f in csv_files
            }
            selected_csvs = st.multiselect(
                "選擇要刪除的 CSV",
                options=list(csv_options.keys())
            )
            if st.button("刪除勾選的 CSV"):
                for label in selected_csvs:
                    csv_options[label].unlink()
                st.success(f"已刪除 {len(selected_csvs)} 個快取檔案")
                load_data.clear()
        else:
            st.write("無 CSV 快取")

        st.write("**計算結果快取（JSON）**")
        grid_path   = Path("data/cache/grid_search_results.json")
        bayes_path  = Path("data/cache/bayesian_opt_results.json")
        wf_path     = Path("data/cache/walk_forward_results.json")
        pareto_path = Path("data/cache/pareto_results.json")
        mc_path     = Path("data/cache/mc_results.json")

        for label, path in [
            ("grid_search_results.json",    grid_path),
            ("bayesian_opt_results.json",   bayes_path),
            ("walk_forward_results.json",   wf_path),
            ("pareto_results.json",         pareto_path),
            ("mc_results.json",             mc_path),
        ]:
            if path.exists():
                size_kb = path.stat().st_size // 1024
                st.write(f"✅ {label}（{size_kb} KB）")
            else:
                st.write(f"❌ {label}（尚未計算）")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("刪除 Grid Search"):
                if grid_path.exists():
                    grid_path.unlink()
                    load_grid_cache.clear()
                    st.warning("已刪除，需重新執行 Grid Search")
            if st.button("刪除 Walk-Forward"):
                if wf_path.exists():
                    wf_path.unlink()
                    st.warning("已刪除，需重新執行 Walk-Forward")
            if st.button("刪除 MC 快取"):
                if mc_path.exists():
                    mc_path.unlink()
                    st.warning("已刪除，需重新執行 Walk-Forward MC")
        with col2:
            if st.button("刪除貝氏最佳化"):
                if bayes_path.exists():
                    bayes_path.unlink()
                    st.warning("已刪除，需重新執行貝氏最佳化")
            if st.button("刪除 Pareto 快取"):
                if pareto_path.exists():
                    pareto_path.unlink()
                    if "pareto_result" in st.session_state:
                        del st.session_state["pareto_result"]
                    if "pareto_hash" in st.session_state:
                        del st.session_state["pareto_hash"]
                    st.warning("已刪除，需重新執行 Pareto 掃描")

    # =========================================================================
    # 回傳參數（全部從 session_state 讀取）
    # =========================================================================
    s = st.session_state
    return {
        "ticker1":    s["ticker1"],
        "ticker2":    s["ticker2"],
        "target_w":   s["target_w"],
        "start_date": s["start_date"],
        "end_date":   s["end_date"],
        "fee_rate":   s["fee_rate"],
        "kf_r":       s["kf_r"],
        "warmup":     s["warmup"],
        "d_clip":     s["d_clip"],
        "output_clip": s["output_clip"],
        "norm_ref_rmse": s["norm_ref_rmse_pct"] / 100,
        "norm_ref_cost": s["norm_ref_cost_pct"] / 100,
        "bayes_kp_min":  s["bayes_kp_min"], "bayes_kp_max": s["bayes_kp_max"],
        "bayes_kd_min":  s["bayes_kd_min"], "bayes_kd_max": s["bayes_kd_max"],
        "bayes_q_min":   s["bayes_q_min"],  "bayes_q_max":  s["bayes_q_max"],
        "bayes_n_trials": s["bayes_n_trials"],
        "bayes_db_min":    s.get("bayes_db_min", 0.005),
        "bayes_db_max":    s.get("bayes_db_max", 0.10),
        "bayes_db_points": int(s.get("bayes_db_points", 15)),
        "kp": 0.5, "kd": 0.5, "kf_q": 0.001, "deadband": 0.02,
        "deadband_values": np.linspace(
            s.get("bayes_db_min", 0.005),
            s.get("bayes_db_max", 0.10),
            int(s.get("bayes_db_points", 15))
        ).tolist(),
        "n_jobs": n_jobs,
    }


# =============================================================================
# Tab 1: Pareto Frontier
# =============================================================================
def render_tab_pareto(params: dict, data: pd.DataFrame,
                      warmup_prices_stock: np.ndarray,
                      warmup_prices_bond: np.ndarray):
    """渲染 Pareto Frontier 分頁"""
    st.header("Pareto Frontier - 三策略對比")

    st.markdown("""
    比較三種再平衡策略在 **RMSE（追蹤誤差）** vs **Cost（年化交易成本）** 空間的表現。
    越靠左下角的策略越好（低誤差、低成本）。
    """)


    # ── Pareto 參數設定（form）──
    with st.form("form_pareto_params"):
        st.markdown("⚙️ **Pareto 掃描參數設定**")
        col1, col2, col3 = st.columns(3)
        with col1:
            pareto_kp = st.number_input("Kp", value=0.5, min_value=0.01, step=0.1, format="%.2f")
        with col2:
            pareto_kd = st.number_input("Kd", value=0.5, min_value=0.01, step=0.1, format="%.2f")
        with col3:
            pareto_q = st.number_input("Q", value=0.001, min_value=0.000001, format="%.5f")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pareto_db_min = st.number_input("Deadband 最小值", value=0.005, min_value=0.001, step=0.005, format="%.3f")
        with col2:
            pareto_db_max = st.number_input("Deadband 最大值", value=0.10, min_value=0.01, step=0.01, format="%.3f")
        with col3:
            pareto_db_points = st.number_input("Deadband 點數", value=30, min_value=5, step=5)
        
        st.form_submit_button("更改 Pareto 參數", use_container_width=True)

    pareto_db_values = np.linspace(pareto_db_min, pareto_db_max, int(pareto_db_points)).tolist()
    # ── Hash 比對 ──
    pareto_hash_params = {
        "ticker1": params["ticker1"], "ticker2": params["ticker2"],
        "start_date": str(params["start_date"]), "end_date": str(params["end_date"]),
        "target_w": params["target_w"], "fee_rate": params["fee_rate"],
        "kf_r": params["kf_r"], "warmup": params["warmup"],
        "kf_q": pareto_q,
        "kp": pareto_kp,
        "kd": pareto_kd,
        "db_min": pareto_db_min,
        "db_max": pareto_db_max,
        "db_points": int(pareto_db_points),
        "norm_ref_rmse": params["norm_ref_rmse"],
        "norm_ref_cost": params["norm_ref_cost"],
    }
    current_hash = make_params_hash(pareto_hash_params)
    pareto_path = Path("data/cache/pareto_results.json")

    # ── 執行按鈕 ──
    if st.button("▶ 執行 Pareto 掃描", type="primary", key="btn_pareto"):
        if pareto_path.exists():
            with open(pareto_path, encoding="utf-8") as f:
                pareto_cache = json.load(f)
            if pareto_cache.get("params_hash") == current_hash:
                st.success("✅ 參數未變，使用上次 Pareto 掃描結果")
                pareto = pareto_cache["pareto"]
            else:
                pareto = None
        else:
            pareto = None

        if pareto is None:
            # 準備資料
            prices_stock = data[params["ticker1"]].values
            prices_bond = data[params["ticker2"]].values
            dates = data.index.tolist()

            rets_stock = np.zeros(len(prices_stock))
            rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
            rets_bond = np.zeros(len(prices_bond))
            rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

            with st.spinner("掃描 Pareto Frontier..."):
                pareto = scan_pareto_frontier(
                    rets_stock, rets_bond, prices_stock, prices_bond, dates,
                    target_w=params["target_w"],
                    fee_rate=params["fee_rate"],
                    kf_q=pareto_q,
                    kf_r=params["kf_r"],
                    kp=pareto_kp,
                    kd=pareto_kd,
                    deadband_values=pareto_db_values,
                    warmup=params["warmup"],
                    warmup_prices_stock=warmup_prices_stock,
                    warmup_prices_bond=warmup_prices_bond,
                )
            # 存 JSON（含 params_hash）
            pareto_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pareto_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pareto": pareto,
                    "params_hash": current_hash,
                }, f, ensure_ascii=False, indent=2)

        st.session_state["pareto_result"] = pareto
        st.session_state["pareto_hash"] = current_hash

    # ── 從 session_state 或 JSON 讀取結果 ──
    if "pareto_result" not in st.session_state:
        if pareto_path.exists():
            with open(pareto_path, encoding="utf-8") as f:
                pareto_cache = json.load(f)
            st.session_state["pareto_result"] = pareto_cache["pareto"]
            st.session_state["pareto_hash"] = pareto_cache.get("params_hash")
        else:
            st.info("請按「▶ 執行 Pareto 掃描」產生結果")
            return

    pareto = st.session_state["pareto_result"]
    if st.session_state.get("pareto_hash") != current_hash:
        st.warning("⚠️ 目前參數與快取結果不符，如需更新請重新執行")

    # 計算超體積（固定標準化參考點）
    ref_rmse = params["norm_ref_rmse"]
    ref_cost = params["norm_ref_cost"]

    def _norm_hv(pts):
        norm_pts = [
            {"rmse": p["rmse"] / ref_rmse, "cost": p["ann_cost"] / ref_cost}
            for p in pts
            if p["rmse"] / ref_rmse <= 1.0 and p["ann_cost"] / ref_cost <= 1.0
        ]
        return calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})

    hv_sp  = _norm_hv(pareto["smart_pilot"])
    hv_to  = _norm_hv(pareto["threshold_only"])
    hv_tat = _norm_hv(pareto["time_and_threshold"])

    # 決定贏家
    hv_values = {"smart_pilot": hv_sp, "threshold_only": hv_to, "time_and_threshold": hv_tat}
    winner = max(hv_values, key=hv_values.get)
    hv_comparison = {
        "smart_pilot": hv_sp,
        "threshold_only": hv_to,
        "time_and_threshold": hv_tat,
        "winner": winner,
    }

    # 繪製 Pareto 圖
    fig = go.Figure()

    # Smart Pilot
    sp_data = pareto["smart_pilot"]
    fig.add_trace(go.Scatter(
        x=[p["rmse"] * 100 for p in sp_data],
        y=[p["ann_cost"] * 100 for p in sp_data],
        mode="lines+markers",
        name="Smart Pilot",
        line=dict(color="#FFD700", width=2),
        marker=dict(size=6),
        hovertemplate="RMSE: %{x:.2f}%<br>Cost: %{y:.3f}%/年<extra></extra>"
    ))

    # Threshold-only
    to_data = pareto["threshold_only"]
    fig.add_trace(go.Scatter(
        x=[p["rmse"] * 100 for p in to_data],
        y=[p["ann_cost"] * 100 for p in to_data],
        mode="lines+markers",
        name="Threshold-only",
        line=dict(color="gray", width=1.5, dash="dash"),
        marker=dict(size=5),
        hovertemplate="RMSE: %{x:.2f}%<br>Cost: %{y:.3f}%/年<extra></extra>"
    ))

    # Time-and-threshold
    tat_data = pareto["time_and_threshold"]
    fig.add_trace(go.Scatter(
        x=[p["rmse"] * 100 for p in tat_data],
        y=[p["ann_cost"] * 100 for p in tat_data],
        mode="lines+markers",
        name="Time-and-threshold",
        line=dict(color="cyan", width=1.5, dash="dot"),
        marker=dict(size=5),
        hovertemplate="RMSE: %{x:.2f}%<br>Cost: %{y:.3f}%/年<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="追蹤誤差 RMSE (%)",
        yaxis_title="年化交易成本 (%/年)",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 超體積比較
    st.subheader("超體積指標 (Hypervolume)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Smart Pilot",
            f"{hv_sp:.6f}",
            help=f"×10000 = {hv_sp * 10000:.4f}"
        )
    with col2:
        st.metric(
            "Threshold-only",
            f"{hv_to:.6f}",
            help=f"×10000 = {hv_to * 10000:.4f}"
        )
    with col3:
        st.metric(
            "Time-and-threshold",
            f"{hv_tat:.6f}",
            help=f"×10000 = {hv_tat * 10000:.4f}"
        )

    winner = hv_comparison["winner"]
    if winner == "smart_pilot":
        st.success(f"Winner: Smart Pilot (超體積最大，Pareto Frontier 最優)")
    elif winner == "threshold_only":
        st.info(f"Winner: Threshold-only")
    else:
        st.info(f"Winner: Time-and-threshold")

    st.caption(
        f"標準化參考點：RMSE={ref_rmse*100:.2f}%，"
        f"Cost={ref_cost*100:.3f}%/年 → 標準化後固定為 (1, 1)"
    )

    with st.expander("📊 標準化 Pareto 圖", expanded=True):

        norm_ref_rmse = params["norm_ref_rmse"]
        norm_ref_cost = params["norm_ref_cost"]

        # 標準化函式
        def normalize_points(points):
            return [
                {
                    "rmse_norm": p["rmse"] / norm_ref_rmse,
                    "cost_norm": p["ann_cost"] / norm_ref_cost,
                }
                for p in points
                if p["rmse"] / norm_ref_rmse <= 1.0 and p["ann_cost"] / norm_ref_cost <= 1.0
            ]

        sp_norm  = normalize_points(pareto["smart_pilot"])
        to_norm  = normalize_points(pareto["threshold_only"])
        tat_norm = normalize_points(pareto["time_and_threshold"])

        # 標準化空間 HV（參考點固定 (1,1)）
        ref_norm = {"rmse": 1.0, "cost": 1.0}

        def to_hv_input(norm_pts):
            return [{"rmse": p["rmse_norm"], "cost": p["cost_norm"]} for p in norm_pts]

        hv_sp_norm  = calc_hypervolume(to_hv_input(sp_norm),  ref_norm)
        hv_to_norm  = calc_hypervolume(to_hv_input(to_norm),  ref_norm)
        hv_tat_norm = calc_hypervolume(to_hv_input(tat_norm), ref_norm)

        # 繪製標準化圖
        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(
            x=[p["rmse_norm"] for p in sp_norm],
            y=[p["cost_norm"] for p in sp_norm],
            mode="lines+markers",
            name="Smart Pilot",
            line=dict(color="#FFD700", width=2),
            marker=dict(size=6),
            hovertemplate="RMSE: %{x:.3f}<br>Cost: %{y:.4f}<extra></extra>"
        ))
        fig_norm.add_trace(go.Scatter(
            x=[p["rmse_norm"] for p in to_norm],
            y=[p["cost_norm"] for p in to_norm],
            mode="lines+markers",
            name="Threshold-only",
            line=dict(color="gray", width=1.5, dash="dash"),
            marker=dict(size=5),
            hovertemplate="RMSE: %{x:.3f}<br>Cost: %{y:.4f}<extra></extra>"
        ))
        fig_norm.add_trace(go.Scatter(
            x=[p["rmse_norm"] for p in tat_norm],
            y=[p["cost_norm"] for p in tat_norm],
            mode="lines+markers",
            name="Time-and-threshold",
            line=dict(color="cyan", width=1.5, dash="dot"),
            marker=dict(size=5),
            hovertemplate="RMSE: %{x:.3f}<br>Cost: %{y:.4f}<extra></extra>"
        ))
        fig_norm.add_trace(go.Scatter(
            x=[1.0], y=[1.0],
            mode="markers",
            name="參考點 (1,1)",
            marker=dict(size=14, color="red", symbol="x-thin", line=dict(width=3)),
            hovertemplate="參考點 (1.0, 1.0)<extra></extra>"
        ))
        fig_norm.update_layout(
            xaxis_title="標準化追蹤誤差 RMSE（0~1）",
            yaxis_title="標準化年化交易成本（0~1）",
            xaxis=dict(range=[0, 1.15]),
            yaxis=dict(range=[0, 1.15]),
            hovermode="closest",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # 標準化 HV 顯示
        st.subheader("標準化超體積指標")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Smart Pilot",
                f"{hv_sp_norm:.6f}",
                help=f"×10000 = {hv_sp_norm * 10000:.4f}"
            )
        with col2:
            st.metric(
                "Threshold-only",
                f"{hv_to_norm:.6f}",
                help=f"×10000 = {hv_to_norm * 10000:.4f}"
            )
        with col3:
            st.metric(
                "Time-and-threshold",
                f"{hv_tat_norm:.6f}",
                help=f"×10000 = {hv_tat_norm * 10000:.4f}"
            )

        norm_scores = {
            "Smart Pilot": hv_sp_norm,
            "Threshold-only": hv_to_norm,
            "Time-and-threshold": hv_tat_norm,
        }
        norm_winner = max(norm_scores, key=norm_scores.get)

        if norm_winner == "Smart Pilot":
            st.success(f"標準化空間 Winner：Smart Pilot")
        else:
            st.info(f"標準化空間 Winner：{norm_winner}")

        st.caption(
            f"標準化參考點：RMSE={norm_ref_rmse*100:.2f}%，"
            f"Cost={norm_ref_cost*100:.3f}%/年 → 標準化後固定為 (1, 1)"
        )


# =============================================================================
# Tab 2: Heatmap
# =============================================================================
def render_tab_heatmap(params: dict, data: pd.DataFrame,
                       warmup_prices_stock: np.ndarray,
                       warmup_prices_bond: np.ndarray):
    """渲染參數空間熱力圖分頁"""
    st.header("Heatmap - 參數空間")

    # ── Grid Search 參數設定（form，按下 submit 才執行）──
    with st.form("form_grid_search"):
        st.markdown("⚙️ **Grid Search 參數設定**")
        col1, col2, col3 = st.columns(3)
        with col1:
            gs_kp_min = st.number_input("Kp 最小值", value=0.1, step=0.1, format="%.1f")
        with col2:
            gs_kp_max = st.number_input("Kp 最大值", value=1.0, step=0.1, format="%.1f")
        with col3:
            gs_kp_points = st.number_input("Kp 點數", value=10, min_value=3, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            gs_kd_min = st.number_input("Kd 最小值", value=0.1, step=0.1, format="%.1f")
        with col2:
            gs_kd_max = st.number_input("Kd 最大值", value=1.0, step=0.1, format="%.1f")
        with col3:
            gs_kd_points = st.number_input("Kd 點數", value=10, min_value=3, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            gs_q_min = st.number_input("Q 最小值", value=0.00001, min_value=0.000001, format="%.5f")
        with col2:
            gs_q_max = st.number_input("Q 最大值", value=0.1, min_value=0.00001, format="%.4f")
        with col3:
            gs_q_points = st.number_input("Q 點數", value=5, min_value=2, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            gs_db_min = st.number_input("Deadband 最小值", value=0.005, step=0.005, format="%.3f")
        with col2:
            gs_db_max = st.number_input("Deadband 最大值", value=0.10, step=0.005, format="%.3f")
        with col3:
            gs_db_points = st.number_input("Deadband 點數", value=15, min_value=3, step=1)

        submitted_grid = st.form_submit_button(
            "▶ 執行 Grid Search（約 2-5 分鐘）",
            type="primary", use_container_width=True
        )

    # 計算衍生範圍（供 info 顯示及執行使用）
    gs_q_values  = np.logspace(np.log10(gs_q_min), np.log10(gs_q_max), int(gs_q_points)).tolist()
    gs_kp_range  = np.linspace(gs_kp_min, gs_kp_max, int(gs_kp_points)).tolist()
    gs_kd_range  = np.linspace(gs_kd_min, gs_kd_max, int(gs_kd_points)).tolist()
    gs_db_values = np.linspace(gs_db_min, gs_db_max, int(gs_db_points)).tolist()
    gs_total_tasks = len(gs_q_values) * len(gs_kp_range) * len(gs_kd_range)
    n_q = len(gs_q_values)

    st.info(
        f"共 {gs_total_tasks} 組參數（{n_q} 個 Q 值 × {len(gs_kp_range)} Kp × {len(gs_kd_range)} Kd），"
        f"使用 {params['n_jobs']} 核心平行運算。KF 只計算 {n_q} 次（每個 Q 值一次）。"
    )

    if submitted_grid:
        # Hash 比對：參數未變則跳過計算
        grid_hash_params = {
            "ticker1": params["ticker1"], "ticker2": params["ticker2"],
            "start_date": str(params["start_date"]), "end_date": str(params["end_date"]),
            "target_w": params["target_w"], "fee_rate": params["fee_rate"],
            "kf_r": params["kf_r"], "warmup": params["warmup"],
            "norm_ref_rmse": params["norm_ref_rmse"],
            "norm_ref_cost": params["norm_ref_cost"],
            "kp_range": gs_kp_range,
            "kd_range": gs_kd_range,
            "q_values": gs_q_values,
            "deadband_values": gs_db_values,
        }
        current_hash = make_params_hash(grid_hash_params)
        grid_path = Path("data/cache/grid_search_results.json")
        skip_grid = False
        if grid_path.exists():
            with open(grid_path, encoding="utf-8") as f:
                existing_grid = json.load(f)
            if existing_grid.get("params_hash") == current_hash:
                st.success("✅ 參數未變，使用上次 Grid Search 結果")
                load_grid_cache.clear()
                skip_grid = True

        if not skip_grid:
            gs_progress_bar = st.progress(0)
            gs_status_text  = st.empty()

            with st.spinner("載入資料..."):
                result = load_data(
                    tickers=[params["ticker1"], params["ticker2"]],
                    start_date=str(params["start_date"]),
                    end_date=str(params["end_date"]),
                    warmup_days=params["warmup"],
                )
                full_backtest = result["backtest_data"]
                wm_stock = result["warmup_data"][params["ticker1"]].values
                wm_bond  = result["warmup_data"][params["ticker2"]].values
                ps = full_backtest[params["ticker1"]].values
                pb = full_backtest[params["ticker2"]].values
                dt = full_backtest.index.tolist()
                rs = np.diff(ps) / ps[:-1]
                rb = np.diff(pb) / pb[:-1]
                ps, pb, dt = ps[1:], pb[1:], dt[1:]

            gs_status_text.text(f"Grid Search 進度：0/{n_q} 個 Q 值")
            t0 = time.time()
            from core.optimizer import run_grid_search_with_progress, find_best_from_grid
            results = run_grid_search_with_progress(
                rs, rb, ps, pb, dt,
                target_w=params["target_w"],
                fee_rate=params["fee_rate"],
                kp_range=gs_kp_range,
                kd_range=gs_kd_range,
                q_values=gs_q_values,
                deadband_values=gs_db_values,
                n_jobs=params["n_jobs"],
                kf_r=params["kf_r"],
                warmup=params["warmup"],
                norm_ref_rmse=params["norm_ref_rmse"],
                norm_ref_cost=params["norm_ref_cost"],
                warmup_prices_stock=wm_stock,
                warmup_prices_bond=wm_bond,
                progress_bar=gs_progress_bar,
                status_text=gs_status_text,
            )
            best = find_best_from_grid(results)
            elapsed = time.time() - t0
            gs_progress_bar.progress(1.0)
            gs_status_text.text(f"完成！共 {gs_total_tasks} 組，耗時 {elapsed:.1f} 秒")

            grid_path.parent.mkdir(parents=True, exist_ok=True)
            with open(grid_path, "w", encoding="utf-8") as f:
                json.dump({
                    "results": results,
                    "best": best,
                    "metadata": {
                        "q_values":        gs_q_values,
                        "kp_range":        gs_kp_range,
                        "kd_range":        gs_kd_range,
                        "deadband_values": gs_db_values,
                    },
                    "params_hash": current_hash,
                }, f, ensure_ascii=False, indent=2)
            load_grid_cache.clear()
            st.success(
                f"Grid Search 完成！共 {len(results)} 組，耗時 {elapsed:.1f} 秒\n"
                f"最佳：Kp={best['kp']:.2f}, Kd={best['kd']:.2f}, "
                f"Q={best['q']:.5f}, HV={best['hypervolume']:.6f}"
            )

            # ── Grid Search 三策略對比（使用最佳參數回測）──
            n_years = len(dt) / 252.0
            db_values  = gs_db_values
            tol_values = np.linspace(0.005, 0.15, len(gs_db_values)).tolist()
            ref_rmse = params["norm_ref_rmse"]
            ref_cost = params["norm_ref_cost"]

            from core.benchmark import precompute_kf_signals, run_smart_pilot_presignals
            vel_stock, vel_bond = precompute_kf_signals(
                ps, pb,
                kf_q=best["q"], kf_r=params["kf_r"],
                warmup_prices_stock=wm_stock,
                warmup_prices_bond=wm_bond,
            )

            best_hv, best_sp_result, best_deadband = -1.0, None, db_values[0]
            for db in db_values:
                r = run_smart_pilot_presignals(
                    rs, rb, ps, pb, dt,
                    vel_stock=vel_stock, vel_bond=vel_bond,
                    target_w=params["target_w"], fee_rate=params["fee_rate"],
                    kp=best["kp"], kd=best["kd"], deadband=float(db),
                    d_clip=params["d_clip"], output_clip=params["output_clip"],
                )
                ann_cost = r["cost"] / n_years
                if r["rmse"] / ref_rmse <= 1.0 and ann_cost / ref_cost <= 1.0:
                    norm_pts = [{"rmse": r["rmse"] / ref_rmse, "cost": ann_cost / ref_cost}]
                else:
                    norm_pts = []
                hv = calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})
                if hv > best_hv:
                    best_hv, best_sp_result, best_deadband = hv, r, float(db)

            best_hv, best_to_result, best_to_tol = -1.0, None, tol_values[0]
            for tol in tol_values:
                r = run_threshold_only(
                    rs, rb, dt,
                    target_w=params["target_w"], drift_tolerance=float(tol),
                    fee_rate=params["fee_rate"], warmup=0,
                )
                ann_cost = r["cost"] / n_years
                if r["rmse"] / ref_rmse <= 1.0 and ann_cost / ref_cost <= 1.0:
                    norm_pts = [{"rmse": r["rmse"] / ref_rmse, "cost": ann_cost / ref_cost}]
                else:
                    norm_pts = []
                hv = calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})
                if hv > best_hv:
                    best_hv, best_to_result, best_to_tol = hv, r, float(tol)

            best_hv, best_tat_result, best_tat_thresh = -1.0, None, tol_values[0]
            for tol in tol_values:
                r = run_time_and_threshold(
                    rs, rb, dt,
                    target_w=params["target_w"], fee_rate=params["fee_rate"],
                    threshold=float(tol), warmup=0,
                )
                ann_cost = r["cost"] / n_years
                if r["rmse"] / ref_rmse <= 1.0 and ann_cost / ref_cost <= 1.0:
                    norm_pts = [{"rmse": r["rmse"] / ref_rmse, "cost": ann_cost / ref_cost}]
                else:
                    norm_pts = []
                hv = calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})
                if hv > best_hv:
                    best_hv, best_tat_result, best_tat_thresh = hv, r, float(tol)

            st.session_state["gs_backtest_results"] = {
                "sp":  best_sp_result,
                "to":  best_to_result,
                "tat": best_tat_result,
                "best_deadband": best_deadband,
                "n_years": n_years,
                "kp": best["kp"], "kd": best["kd"], "q": best["q"],
            }

    grid_cache = load_grid_cache()

    if grid_cache is None:
        st.warning("找不到 Grid Search 快取檔案。")
        st.info("請展開上方「⚙️ Grid Search 參數設定」執行 Grid Search 產生快取。")
        return

    st.markdown("""
    Grid Search 結果視覺化。每個點代表一組 (Kp, Kd, Q) 參數，
    顏色代表超體積（Hypervolume），顏色越亮表示該參數組合在整個 deadband 範圍內表現越好。
    """)

    results = grid_cache["results"]
    best = grid_cache["best"]
    metadata = grid_cache["metadata"]
    q_values = metadata["q_values"]

    # 選擇 Q 值切片
    selected_q = st.selectbox(
        "選擇 Q 值切片",
        options=q_values,
        index=min(1, len(q_values) - 1),
        format_func=lambda x: f"Q = {x}"
    )

    # 篩選該 Q 值的結果
    q_results = [r for r in results if abs(r["q"] - selected_q) < 1e-8]

    if not q_results:
        st.warning(f"沒有 Q = {selected_q} 的資料")
        return

    # 建立熱力圖資料
    kp_vals = sorted(set(r["kp"] for r in q_results))
    kd_vals = sorted(set(r["kd"] for r in q_results))

    heatmap_data = np.zeros((len(kd_vals), len(kp_vals)))
    for r in q_results:
        i = kd_vals.index(r["kd"])
        j = kp_vals.index(r["kp"])
        heatmap_data[i, j] = r["hypervolume"]

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{kp:.2f}" for kp in kp_vals],
        y=[f"{kd:.2f}" for kd in kd_vals],
        colorscale="Viridis",
        colorbar=dict(title="Hypervolume"),
        hovertemplate="Kp: %{x}<br>Kd: %{y}<br>HV: %{z:.6f}<extra></extra>"
    ))

    # 標記最佳點
    if abs(best["q"] - selected_q) < 1e-8:
        fig.add_trace(go.Scatter(
            x=[f"{best['kp']:.2f}"],
            y=[f"{best['kd']:.2f}"],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Best",
            hovertemplate=f"Best: Kp={best['kp']:.2f}, Kd={best['kd']:.2f}<extra></extra>"
        ))

    fig.update_layout(
        xaxis_title="Kp",
        yaxis_title="Kd",
        title=f"Hypervolume Heatmap (Q = {selected_q})",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 最佳參數摘要
    st.subheader("Grid Search 最佳參數")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kp", f"{best['kp']:.3f}")
    with col2:
        st.metric("Kd", f"{best['kd']:.3f}")
    with col3:
        st.metric("Q", f"{best['q']:.6f}")
    with col4:
        st.metric("Hypervolume", f"{best['hypervolume']:.6f}")

    # 貝氏最佳化結果
    bayes_path = Path("data/cache/bayesian_opt_results.json")
    if bayes_path.exists():
        with open(bayes_path, "r", encoding="utf-8") as f:
            bayes_cache = json.load(f)
        st.markdown("---")
        st.subheader("貝氏最佳化結果")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Kp", f"{bayes_cache['kp']:.3f}")
        with col2:
            st.metric("Kd", f"{bayes_cache['kd']:.3f}")
        with col3:
            st.metric("Q",  f"{bayes_cache['q']:.6f}")
        with col4:
            st.metric("Hypervolume", f"{bayes_cache['hypervolume']:.6f}")
        st.caption(f"試驗次數：{bayes_cache.get('n_trials', 'N/A')}")
    else:
        st.info("尚未執行貝氏最佳化，請在側邊欄執行。")

    # =========================================================================
    # 單點快速測試
    # =========================================================================
    st.markdown("---")
    with st.expander("🧪 單點快速測試", expanded=False):
        st.markdown("輸入一組參數，快速計算這組參數的超體積和回測指標。")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            test_kp = st.number_input(
                "Kp", min_value=0.01, max_value=2.0,
                value=0.5, step=0.1, format="%.2f", key="test_kp"
            )
        with col2:
            test_kd = st.number_input(
                "Kd", min_value=0.01, max_value=2.0,
                value=0.5, step=0.1, format="%.2f", key="test_kd"
            )
        with col3:
            test_q = st.number_input(
                "Q", min_value=0.00001, max_value=1.0,
                value=0.001, step=0.0001, format="%.5f", key="test_q"
            )
        with col4:
            test_deadband = st.number_input(
                "Deadband", min_value=0.001, max_value=0.10,
                value=0.0125, step=0.005, format="%.4f", key="test_deadband"
            )

        if st.button("▶ 執行單點測試", key="btn_single_test"):
            with st.spinner("計算中..."):
                # 準備資料
                prices_stock = data[params["ticker1"]].values
                prices_bond = data[params["ticker2"]].values
                dates = data.index.tolist()
                rets_stock = np.diff(prices_stock) / prices_stock[:-1]
                rets_bond = np.diff(prices_bond) / prices_bond[:-1]
                prices_stock = prices_stock[1:]
                prices_bond = prices_bond[1:]
                dates = dates[1:]

                # 掃描 deadband 計算 Pareto frontier
                pareto = scan_pareto_frontier(
                    rets_stock, rets_bond, prices_stock, prices_bond, dates,
                    target_w=params["target_w"],
                    fee_rate=params["fee_rate"],
                    kf_q=test_q,
                    kf_r=params["kf_r"],
                    kp=test_kp,
                    kd=test_kd,
                    n_points=15,
                    warmup=params["warmup"],
                    warmup_prices_stock=warmup_prices_stock,
                    warmup_prices_bond=warmup_prices_bond,
                )

                # 計算標準化超體積（固定參考點）
                ref_rmse = params["norm_ref_rmse"]
                ref_cost = params["norm_ref_cost"]
                norm_pts = [
                    {"rmse": p["rmse"] / ref_rmse, "cost": p["ann_cost"] / ref_cost}
                    for p in pareto["smart_pilot"]
                    if p["rmse"] / ref_rmse <= 1.0 and p["ann_cost"] / ref_cost <= 1.0
                ]
                hv = calc_hypervolume(norm_pts, {"rmse": 1.0, "cost": 1.0})

                # 用指定 deadband 跑完整回測
                result = run_smart_pilot(
                    rets_stock, rets_bond, prices_stock, prices_bond, dates,
                    target_w=params["target_w"],
                    fee_rate=params["fee_rate"],
                    kf_q=test_q,
                    kf_r=params["kf_r"],
                    kp=test_kp,
                    kd=test_kd,
                    deadband=test_deadband,
                    warmup=params["warmup"],
                    warmup_prices_stock=warmup_prices_stock,
                    warmup_prices_bond=warmup_prices_bond,
                )
                n_years = len(dates) / 252.0

            # 顯示結果
            st.markdown("**測試結果：**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("超體積", f"{hv:.6f}")
            with col2:
                st.metric("RMSE", f"{result['rmse'] * 100:.3f}%")
            with col3:
                st.metric("年化成本", f"{result['cost'] / n_years * 100:.4f}%")
            with col4:
                st.metric("交易次數", f"{result['trade_count']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("年化報酬", f"{result['metrics']['ann_return_pct']:.2f}%")
            with col2:
                st.metric("Sharpe", f"{result['metrics']['sharpe']:.2f}")
            with col3:
                st.metric("最大回撤", f"{result['metrics']['max_drawdown'] * 100:.2f}%")

            st.caption(
                f"標準化參考點：RMSE={ref_rmse*100:.2f}%，Cost={ref_cost*100:.3f}%/年"
            )

            # 和 Grid Search 最佳結果比較
            grid_cache = load_grid_cache()
            if grid_cache is not None:
                best = grid_cache["best"]
                st.markdown("**與 Grid Search 最佳結果比較：**")
                delta_hv = hv - best["hypervolume"]
                if delta_hv >= 0:
                    st.success(
                        f"這組參數的超體積比 Grid Search 最佳結果高 {delta_hv:.6f}"
                    )
                else:
                    st.info(
                        f"Grid Search 最佳：Kp={best['kp']:.2f}, "
                        f"Kd={best['kd']:.2f}, Q={best['q']:.5f}, "
                        f"HV={best['hypervolume']:.6f}（差距 {abs(delta_hv):.6f}）"
                    )
            else:
                st.caption("尚未執行 Grid Search，無法比較最佳結果")

    # =========================================================================
    # Helper（放函數最底部，供下方 expander 共用）
    # =========================================================================
    def _make_row(name, r, n_years):
        m = r["metrics"]
        return {
            "策略":    name,
            "交易次數": str(res["trade_count"]),
            "總周轉率": f"{r['turnover']:.3f}",
            "RMSE":    f"{r['rmse']*100:.2f}%",
            "年化報酬": f"{m['ann_return']*100:.2f}%",
            "Sharpe":  f"{m['sharpe']:.2f}",
            "MDD":     f"{m['max_drawdown']*100:.2f}%",
            "年化波動": f"{m['ann_wealth_vol']*100:.2f}%",
        }

    st.markdown("---")
    with st.expander("▶ Grid Search 最佳參數三策略對比", expanded=False):
        if "gs_backtest_results" in st.session_state:
            gs = st.session_state["gs_backtest_results"]
            st.caption(
                f"最佳參數：Kp={gs['kp']:.3f}, Kd={gs['kd']:.3f}, "
                f"Q={gs['q']:.6f}, Deadband={gs['best_deadband']:.4f}"
            )
            rows = [
                _make_row("Smart Pilot",        gs["sp"],  gs["n_years"]),
                _make_row("Threshold-only",     gs["to"],  gs["n_years"]),
                _make_row("Time-and-threshold", gs["tat"], gs["n_years"]),
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("請先執行 Grid Search")

    with st.expander("▶ 貝氏最佳化最佳參數三策略對比", expanded=False):
        if "bayes_backtest_results" in st.session_state:
            br = st.session_state["bayes_backtest_results"]
            st.caption(
                f"最佳參數：Kp={br['kp']:.3f}, Kd={br['kd']:.3f}, Q={br['q']:.6f} | "
                f"SP Deadband={br['best_deadband']:.4f}, "
                f"TO Tolerance={br['best_tol']:.4f}, "
                f"TAT Threshold={br['best_thresh']:.4f}"
            )
            rows = [
                _make_row("Smart Pilot",        br["sp"],  br["n_years"]),
                _make_row("Threshold-only",     br["to"],  br["n_years"]),
                _make_row("Time-and-threshold", br["tat"], br["n_years"]),
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("請先執行貝氏最佳化")


# =============================================================================
# Tab 3: Rolling Window
# =============================================================================
def render_tab_walking_forward(params: dict, data: pd.DataFrame,
                               warmup_prices_stock: np.ndarray,
                               warmup_prices_bond: np.ndarray):
    """Walk-Forward 滾動窗口分析"""
    st.header("Walk-Forward Analysis")

    st.markdown("""
    每一輪包含：
    - **IS（樣本內）**：貝氏最佳化尋找最佳 (Kp, Kd, Q)，最大化標準化超體積
    - **OOS（樣本外）**：套用 IS 最佳參數，三策略直接回測，不再最佳化
    - **窗口步進**：可自訂步進年數（預設 1 年）
    - **OOS HV 參考點**：與 IS 相同的固定參考點，IS/OOS HV 直接可比
    """)

    prices_stock = data[params["ticker1"]].values
    prices_bond  = data[params["ticker2"]].values
    dates        = data.index.tolist()
    total_days   = len(dates)

    # ── Walk-Forward 參數設定（form，按下才執行）──
    with st.form("form_wf"):
        col1, col2, col3 = st.columns(3)
        with col1:
            is_years = st.number_input(
                "IS 年數（樣本內）", min_value=2, max_value=10, value=5, step=1
            )
        with col2:
            oos_years = st.number_input(
                "OOS 年數（樣本外）", min_value=1, max_value=5, value=2, step=1
            )
        with col3:
            step_years = st.number_input(
                "步進年數", min_value=1, max_value=5, value=1, step=1,
                help="每輪窗口向後移動幾年，預設 1 年（最密集）"
            )
        wf_n_trials = st.number_input(
            "每輪貝氏試驗次數", value=int(params["bayes_n_trials"]),
            min_value=10, step=10,
            help="Walk-Forward 每一輪 IS 最佳化的試驗次數"
        )
        st.markdown("**Deadband 掃描範圍**")
        col1, col2, col3 = st.columns(3)
        with col1:
            wf_db_min = st.number_input(
                "Deadband 最小值", value=0.005, min_value=0.001, step=0.005, format="%.3f"
            )
        with col2:
            wf_db_max = st.number_input(
                "Deadband 最大值", value=0.10, min_value=0.01, step=0.01, format="%.3f"
            )
        with col3:
            wf_db_points = st.number_input(
                "Deadband 點數", value=15, min_value=5, step=5
            )
        submitted_wf = st.form_submit_button(
            "▶ 執行 Walk-Forward 分析", type="primary", use_container_width=True
        )
    run_wf = submitted_wf

    st.info(
        f"OOS HV 參考點沿用側邊欄設定："
        f"RMSE={params['norm_ref_rmse']*100:.2f}%，"
        f"Cost={params['norm_ref_cost']*100:.3f}%/年"
    )

    # 預估輪數
    is_days   = int(is_years)  * 252
    oos_days  = int(oos_years) * 252
    step_days = int(step_years) * 252
    n_rounds_est = max(0, (total_days - is_days - oos_days) // step_days + 1)
    st.info(
        f"預估輪數：約 {n_rounds_est} 輪 | "
        f"每輪 {int(wf_n_trials)} 次貝氏試驗 | "
        f"步進 {int(step_years)} 年 | "
        f"資料總長：{total_days} 天（{total_days/252:.1f} 年）"
    )

    if n_rounds_est == 0:
        st.warning("資料長度不足以完成一輪 IS+OOS，請縮短 IS/OOS 年數或延長回測區間")
        return

    # Hash 比對
    wf_hash_params = {
        "ticker1": params["ticker1"], "ticker2": params["ticker2"],
        "start_date": str(params["start_date"]), "end_date": str(params["end_date"]),
        "target_w": params["target_w"], "fee_rate": params["fee_rate"],
        "kf_r": params["kf_r"], "warmup": params["warmup"],
        "d_clip": params["d_clip"], "output_clip": params["output_clip"],
        "norm_ref_rmse": params["norm_ref_rmse"],
        "norm_ref_cost": params["norm_ref_cost"],
        "is_years": int(is_years), "oos_years": int(oos_years),
        "step_years": int(step_years), "n_trials": int(wf_n_trials),
        "wf_db_min": float(wf_db_min), "wf_db_max": float(wf_db_max),
        "wf_db_points": int(wf_db_points),
        "kp_min": params["bayes_kp_min"], "kp_max": params["bayes_kp_max"],
        "kd_min": params["bayes_kd_min"], "kd_max": params["bayes_kd_max"],
        "q_min": params["bayes_q_min"],   "q_max": params["bayes_q_max"],
    }
    current_hash = make_params_hash(wf_hash_params)
    wf_path = Path("data/cache/walk_forward_results.json")

    rounds = None
    if run_wf:
        if wf_path.exists():
            with open(wf_path, encoding="utf-8") as f:
                wf_cache = json.load(f)
            if wf_cache.get("params_hash") == current_hash:
                st.success("✅ 參數未變，使用上次 Walk-Forward 結果")
                rounds = _deserialize_rounds(wf_cache["rounds"])
            else:
                rounds = None  # 強制重跑
        if rounds is None:
            wf_progress_bar = st.progress(0)
            wf_status_text = st.empty()
            with st.spinner("執行中，請耐心等候..."):
                from core.walk_forward import run_walk_forward
                wf_result = run_walk_forward(
                    prices_stock, prices_bond, dates,
                    warmup_prices_stock=warmup_prices_stock,
                    warmup_prices_bond=warmup_prices_bond,
                    target_w=params["target_w"],
                    fee_rate=params["fee_rate"],
                    kf_r=params["kf_r"],
                    d_clip=params["d_clip"],
                    output_clip=params["output_clip"],
                    is_years=int(is_years),
                    oos_years=int(oos_years),
                    step_years=int(step_years),
                    n_trials=int(wf_n_trials),
                    deadband_values=np.linspace(
                        wf_db_min, wf_db_max, int(wf_db_points)
                    ).tolist(),
                    norm_ref_rmse=params["norm_ref_rmse"],
                    norm_ref_cost=params["norm_ref_cost"],
                    kp_min=params["bayes_kp_min"],
                    kp_max=params["bayes_kp_max"],
                    kd_min=params["bayes_kd_min"],
                    kd_max=params["bayes_kd_max"],
                    q_min=params["bayes_q_min"],
                    q_max=params["bayes_q_max"],
                    n_jobs=params["n_jobs"],
                    progress_bar=wf_progress_bar,
                    status_text=wf_status_text,
                )
            rounds = wf_result["rounds"]
            if not rounds:
                st.warning("未能完成任何一輪，請調整參數")
                return
            # 存 JSON（含 params_hash）
            wf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wf_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rounds": _serialize_rounds(rounds),
                    "params_hash": current_hash,
                    "is_years": int(is_years),
                    "oos_years": int(oos_years),
                    "step_years": int(step_years),
                }, f, ensure_ascii=False, indent=2)
            st.success(f"Walk-Forward 完成！共 {len(rounds)} 輪")
    elif wf_path.exists():
        with open(wf_path, encoding="utf-8") as f:
            wf_cache = json.load(f)
        rounds = _deserialize_rounds(wf_cache["rounds"])
        if wf_cache.get("params_hash") != current_hash:
            st.warning("⚠️ 目前參數與快取結果不符，如需更新請重新執行")
        else:
            st.success(f"顯示上次 Walk-Forward 結果，共 {len(rounds)} 輪")
    else:
        st.info("請調整好參數後，按下「▶ 執行 Walk-Forward 分析」")
        return

    if not rounds:
        st.warning("未能完成任何一輪，請調整參數")
        return

    # ── 圖一：Walk-Forward 時間軸（Gantt Chart）──
    st.subheader("圖一：Walk-Forward 時間軸")
    fig_gantt = go.Figure()
    base_date = rounds[0]["is_start"]
    for r in rounds:
        is_start_days  = (r["is_start"]  - base_date).days
        is_len_days    = (r["is_end"]    - r["is_start"]).days
        oos_start_days = (r["oos_start"] - base_date).days
        oos_len_days   = (r["oos_end"]   - r["oos_start"]).days
        label = f"Round {r['round']}"
        fig_gantt.add_trace(go.Bar(
            name="IS（樣本內）", x=[is_len_days], y=[label],
            base=[is_start_days], orientation="h", marker_color="#4A90D9",
            showlegend=(r["round"] == 1), legendgroup="IS",
            hovertemplate=(
                f"Round {r['round']} IS<br>"
                f"{r['is_start'].strftime('%Y/%m/%d')} ~ "
                f"{r['is_end'].strftime('%Y/%m/%d')}<extra></extra>"
            ),
        ))
        fig_gantt.add_trace(go.Bar(
            name="OOS（樣本外）", x=[oos_len_days], y=[label],
            base=[oos_start_days], orientation="h", marker_color="#F5A623",
            showlegend=(r["round"] == 1), legendgroup="OOS",
            hovertemplate=(
                f"Round {r['round']} OOS<br>"
                f"{r['oos_start'].strftime('%Y/%m/%d')} ~ "
                f"{r['oos_end'].strftime('%Y/%m/%d')}<extra></extra>"
            ),
        ))
    fig_gantt.update_layout(
        barmode="overlay", xaxis_title="距第一輪起始天數",
        yaxis_title="滾動輪次", hovermode="closest",
        height=max(300, len(rounds) * 60),
    )
    st.plotly_chart(fig_gantt, use_container_width=True)

    # ── 圖二：OOS 績效大對決（雙 Y 軸）──
    st.subheader("圖二：OOS 績效大對決")
    oos_labels = [
        f"{r['oos_start'].strftime('%Y/%m')}~{r['oos_end'].strftime('%Y/%m')}"
        for r in rounds
    ]
    fig_perf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_perf.add_trace(go.Bar(
        name="Smart Pilot HV", x=oos_labels,
        y=[r["oos_sp_hv"] for r in rounds],
        marker_color="#FFD700", offsetgroup=0,
    ), secondary_y=False)
    fig_perf.add_trace(go.Bar(
        name="Threshold-only HV", x=oos_labels,
        y=[r["oos_to_hv"] for r in rounds],
        marker_color="gray", offsetgroup=1,
    ), secondary_y=False)
    fig_perf.add_trace(go.Bar(
        name="Time-and-threshold HV", x=oos_labels,
        y=[r["oos_tat_hv"] for r in rounds],
        marker_color="#00CED1", offsetgroup=2,
    ), secondary_y=False)
    fig_perf.add_trace(go.Scatter(
        name="Smart Pilot 波動率", x=oos_labels,
        y=[r["oos_sp"]["metrics"]["ann_wealth_vol"] * 100 for r in rounds],
        mode="lines+markers",
        line=dict(color="#FFD700", dash="dot", width=2),
        marker=dict(symbol="circle", size=8),
    ), secondary_y=True)
    fig_perf.add_trace(go.Scatter(
        name="Threshold-only 波動率", x=oos_labels,
        y=[r["oos_to"]["metrics"]["ann_wealth_vol"] * 100 for r in rounds],
        mode="lines+markers",
        line=dict(color="gray", dash="dot", width=2),
        marker=dict(symbol="square", size=8),
    ), secondary_y=True)
    fig_perf.add_trace(go.Scatter(
        name="Time-and-threshold 波動率", x=oos_labels,
        y=[r["oos_tat"]["metrics"]["ann_wealth_vol"] * 100 for r in rounds],
        mode="lines+markers",
        line=dict(color="#00CED1", dash="dot", width=2),
        marker=dict(symbol="diamond", size=8),
    ), secondary_y=True)
    fig_perf.update_yaxes(title_text="OOS 超體積（固定參考點）", secondary_y=False)
    fig_perf.update_yaxes(title_text="OOS 年化波動率 (%)", secondary_y=True)
    fig_perf.update_layout(barmode="group", hovermode="x unified", height=500)
    st.plotly_chart(fig_perf, use_container_width=True)
    st.caption(
        f"OOS HV 使用固定參考點（RMSE={params['norm_ref_rmse']*100:.2f}%，"
        f"Cost={params['norm_ref_cost']*100:.3f}%/年），與 IS HV 同一尺度，衰退比值有意義。"
    )

    # ── 圖三：參數穩定性追蹤 ──
    st.subheader("圖三：最佳化參數穩定性追蹤")
    round_labels = [f"Round {r['round']}" for r in rounds]
    fig_params = go.Figure()
    fig_params.add_trace(go.Scatter(
        name="Kp", x=round_labels, y=[r["best_kp"] for r in rounds],
        mode="lines+markers", line=dict(color="#1f77b4", width=2), marker=dict(size=8),
    ))
    fig_params.add_trace(go.Scatter(
        name="Kd", x=round_labels, y=[r["best_kd"] for r in rounds],
        mode="lines+markers", line=dict(color="#ff7f0e", width=2), marker=dict(size=8),
    ))
    fig_params.add_trace(go.Scatter(
        name="Q×100", x=round_labels, y=[r["best_q"] * 100 for r in rounds],
        mode="lines+markers",
        line=dict(color="#2ca02c", width=2, dash="dash"), marker=dict(size=8),
    ))
    fig_params.update_layout(
        xaxis_title="滾動輪次", yaxis_title="參數數值",
        hovermode="x unified", height=400,
    )
    st.plotly_chart(fig_params, use_container_width=True)

    # ── 表格：每個 Round 三策略詳細指標 ──
    st.subheader("各輪次三策略詳細指標")

    table_rows = []
    for r in rounds:
        round_label = f"Round {r['round']} OOS ({r['oos_start'].strftime('%Y/%m')}~{r['oos_end'].strftime('%Y/%m')})"
        for strategy_key, strategy_name in [
            ("oos_sp",  "Smart Pilot"),
            ("oos_to",  "Threshold-only"),
            ("oos_tat", "Time-and-threshold"),
        ]:
            res = r[strategy_key]
            m = res["metrics"]
            hv_map = {
                "oos_sp":  r["oos_sp_hv"],
                "oos_to":  r["oos_to_hv"],
                "oos_tat": r["oos_tat_hv"],
            }
            table_rows.append({
                "Round":   round_label,
                "策略":    strategy_name,
                "HV":      f"{hv_map[strategy_key]:.6f}",
                "交易次數": str(res["trade_count"]),
                "總周轉率": f"{res['turnover']:.3f}",
                "RMSE":    f"{res['rmse']*100:.2f}%",
                "年化報酬": f"{m['ann_return']*100:.2f}%",
                "Sharpe":  f"{m['sharpe']:.2f}",
                "MDD":     f"{m['max_drawdown']*100:.2f}%",
                "年化波動": f"{m['ann_wealth_vol']*100:.2f}%",
            })

    # 平均列
    for strategy_key, strategy_name in [
        ("oos_sp",  "Smart Pilot"),
        ("oos_to",  "Threshold-only"),
        ("oos_tat", "Time-and-threshold"),
    ]:
        avg_trades = np.mean([r[strategy_key]["trade_count"]               for r in rounds])
        avg_to_val = np.mean([r[strategy_key]["turnover"]                  for r in rounds])
        avg_rmse   = np.mean([r[strategy_key]["rmse"]                      for r in rounds])
        avg_ret    = np.mean([r[strategy_key]["metrics"]["ann_return"]      for r in rounds])
        avg_sharpe = np.mean([r[strategy_key]["metrics"]["sharpe"]          for r in rounds])
        avg_mdd    = np.mean([r[strategy_key]["metrics"]["max_drawdown"]    for r in rounds])
        avg_vol    = np.mean([r[strategy_key]["metrics"]["ann_wealth_vol"]  for r in rounds])
        hv_key_map = {"oos_sp": "oos_sp_hv", "oos_to": "oos_to_hv", "oos_tat": "oos_tat_hv"}
        avg_hv = np.mean([r[hv_key_map[strategy_key]] for r in rounds])
        table_rows.append({
            "Round":   "**平均**",
            "策略":    strategy_name,
            "HV":      f"{avg_hv:.6f}",
            "交易次數": f"{avg_trades:.1f}",
            "總周轉率": f"{avg_to_val:.3f}",
            "RMSE":    f"{avg_rmse*100:.2f}%",
            "年化報酬": f"{avg_ret*100:.2f}%",
            "Sharpe":  f"{avg_sharpe:.2f}",
            "MDD":     f"{avg_mdd*100:.2f}%",
            "年化波動": f"{avg_vol*100:.2f}%",
        })

    detail_df = pd.DataFrame(table_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # ── 表格二：過度擬合檢驗 ──
    st.subheader("表格二：過度擬合檢驗（Smart Pilot IS vs OOS）")
    overfit_df = pd.DataFrame({
        "輪次": [
            f"Round {r['round']} ({r['is_start'].strftime('%Y')}~{r['oos_end'].strftime('%Y')})"
            for r in rounds
        ],
        "IS 區間":  [f"{r['is_start'].strftime('%Y/%m/%d')}~{r['is_end'].strftime('%Y/%m/%d')}"   for r in rounds],
        "OOS 區間": [f"{r['oos_start'].strftime('%Y/%m/%d')}~{r['oos_end'].strftime('%Y/%m/%d')}" for r in rounds],
        "IS HV":   [f"{r['is_hv']:.6f}"     for r in rounds],
        "OOS HV":  [f"{r['oos_sp_hv']:.6f}" for r in rounds],
        "衰退比值(OOS/IS)": [
            f"{r['oos_sp_hv']/r['is_hv']:.3f}" if r["is_hv"] > 0 else "N/A"
            for r in rounds
        ],
        "最佳 Kp": [f"{r['best_kp']:.3f}" for r in rounds],
        "最佳 Kd": [f"{r['best_kd']:.3f}" for r in rounds],
        "最佳 Q":  [f"{r['best_q']:.6f}"  for r in rounds],
    })
    st.dataframe(overfit_df, use_container_width=True, hide_index=True)
    st.caption(
        "衰退比值接近 1.0 → 無過擬合；遠小於 1.0（如 < 0.5）→ 可能過擬合。\n"
        "IS HV 與 OOS HV 使用相同參考點，衰退比值可直接判讀。"
    )


# =============================================================================
# Tab 4: Monte Carlo
# =============================================================================
def render_tab_monte_carlo(params: dict, data: pd.DataFrame,
                           warmup_prices_stock: np.ndarray, warmup_prices_bond: np.ndarray):
    st.header("Walk-Forward MC - 矩陣化蒙地卡羅模擬")
    st.markdown("""
    以 Walk-Forward 各窗口的**平均最佳參數 (Kp, Kd, Q)**，
    對未來期間用區塊重抽樣（Block Bootstrap）產生虛擬路徑，
    評估三策略的績效分布與 Smart Pilot 的相對優勢。
    """)
    # ── 前置檢查：Walk-Forward 快取（改為軟性依賴）──
    wf_path = Path("data/cache/walk_forward_results.json")
    wf_available = wf_path.exists()

    # WF 預設值（有快取就從快取讀，沒有就給 fallback）
    if wf_available:
        with open(wf_path, encoding="utf-8") as f:
            wf_cache = json.load(f)
        rounds = _deserialize_rounds(wf_cache["rounds"])
        avg_kp_auto = float(np.mean([r["best_kp"] for r in rounds]))
        avg_kd_auto = float(np.mean([r["best_kd"] for r in rounds]))
        avg_q_auto  = float(np.mean([r["best_q"]  for r in rounds]))
        last_is_end = rounds[-1]["is_end"]
        wf_sim_start_auto = date(last_is_end.year + 1, 1, 1)
        wf_hist_end_auto  = last_is_end
        st.success(
            f"✅ Walk-Forward 快取已載入｜平均參數：Kp={avg_kp_auto:.3f}, "
            f"Kd={avg_kd_auto:.3f}, Q={avg_q_auto:.6f}（{len(rounds)} 個窗口）"
        )
    else:
        avg_kp_auto = 0.5
        avg_kd_auto = 0.5
        avg_q_auto  = 0.001
        wf_hist_end_auto  = params["end_date"]
        wf_sim_start_auto = date(params["end_date"].year + 1, 1, 1)
        st.warning(
            "⚠️ 尚未執行 Walk-Forward，使用手動設定的參數。"
            "可在 Walk-Forward 分頁執行後，本頁將自動帶入平均最佳參數。"
        )

    # ── 日期覆蓋區塊 ──
    with st.expander("📅 重抽樣來源區間 & 模擬期間（可覆蓋）", expanded=not wf_available):
        with st.form("form_mc_dates"):
            st.markdown("**重抽樣來源區間**（歷史報酬率來源，用於 Block Bootstrap）")
            col1, col2 = st.columns(2)
            with col1:
                hist_start_override = st.date_input(
                    "重抽樣來源：開始日",
                    value=params["start_date"],
                    help="歷史報酬率的起始日，預設為 sidebar 的回測開始日"
                )
            with col2:
                hist_end_override = st.date_input(
                    "重抽樣來源：結束日",
                    value=wf_hist_end_auto,
                    help="歷史報酬率的截止日，預設為最後一個 IS 結束日（如有 WF 快取）"
                )
            st.markdown("**虛擬股價路徑模擬期間**")
            col1, col2 = st.columns(2)
            with col1:
                sim_start_override = st.date_input(
                    "模擬期間：開始日",
                    value=wf_sim_start_auto,
                    help="虛擬路徑的起始日，預設為最後一個 IS 結束的隔年年初"
                )
            with col2:
                sim_end_override = st.date_input(
                    "模擬期間：結束日",
                    value=date(datetime.today().year, 1, 1),
                    help="虛擬路徑的終止日"
                )
            apply_dates = st.form_submit_button("套用日期設定", use_container_width=True)
        if apply_dates:
            st.session_state["mc_hist_start"] = hist_start_override
            st.session_state["mc_hist_end"]   = hist_end_override
            st.session_state["mc_sim_start"]  = sim_start_override
            st.session_state["mc_sim_end"]    = sim_end_override
            st.success(
                f"已套用：重抽樣 {hist_start_override}~{hist_end_override}｜"
                f"模擬 {sim_start_override}~{sim_end_override}"
            )

    # 決定實際使用的日期
    mc_hist_start = st.session_state.get("mc_hist_start", params["start_date"])
    mc_hist_end   = st.session_state.get("mc_hist_end",   wf_hist_end_auto)
    mc_sim_start  = st.session_state.get("mc_sim_start",  wf_sim_start_auto)
    mc_sim_end    = st.session_state.get("mc_sim_end",    date(datetime.today().year, 1, 1))

    # 計算模擬天數
    n_days_simulate = len(pd.bdate_range(mc_sim_start, mc_sim_end))
    if n_days_simulate <= 0:
        st.error("模擬結束日必須晚於開始日，請重新設定")
        return

    # 切出重抽樣來源價格
    hist_start_idx = data.index.searchsorted(pd.Timestamp(mc_hist_start))
    hist_end_idx   = data.index.searchsorted(pd.Timestamp(mc_hist_end))
    hist_prices_stock = data[params["ticker1"]].values[hist_start_idx:hist_end_idx]
    hist_prices_bond  = data[params["ticker2"]].values[hist_start_idx:hist_end_idx]

    if len(hist_prices_stock) < 30:
        st.error(f"重抽樣來源資料不足（僅 {len(hist_prices_stock)} 天），請擴大來源區間")
        return

    st.info(
        f"重抽樣來源：{mc_hist_start} ~ {mc_hist_end}（{len(hist_prices_stock)} 個交易日）｜"
        f"模擬期間：{mc_sim_start} ~ {mc_sim_end}（{n_days_simulate} 個交易日）"
    )
    # ── 手動覆蓋參數（可選）──
    with st.expander("⚙️ 手動覆蓋 Kp / Kd / Q（選填，留空則使用 Walk-Forward 平均值）",
                     expanded=False):
        with st.form("form_mc_override"):
            col1, col2, col3 = st.columns(3)
            with col1:
                override_kp = st.number_input(
                    "Kp（覆蓋）", min_value=0.0, max_value=10.0,
                    value=avg_kp_auto, step=0.01, format="%.4f",
                    help=f"Walk-Forward 平均值：{avg_kp_auto:.4f}"
                )
            with col2:
                override_kd = st.number_input(
                    "Kd（覆蓋）", min_value=0.0, max_value=10.0,
                    value=avg_kd_auto, step=0.01, format="%.4f",
                    help=f"Walk-Forward 平均值：{avg_kd_auto:.4f}"
                )
            with col3:
                override_q = st.number_input(
                    "Q（覆蓋）", min_value=0.0, max_value=1.0,
                    value=avg_q_auto, step=0.00001, format="%.6f",
                    help=f"Walk-Forward 平均值：{avg_q_auto:.6f}"
                )
            use_override = st.form_submit_button("套用覆蓋參數")
        if use_override:
            st.session_state["mc_override_kp"] = override_kp
            st.session_state["mc_override_kd"] = override_kd
            st.session_state["mc_override_q"]  = override_q
            st.success(f"已套用覆蓋參數：Kp={override_kp:.4f}, Kd={override_kd:.4f}, Q={override_q:.6f}")
    # 決定實際使用的參數（有覆蓋就用覆蓋，否則用平均）
    avg_kp = st.session_state.get("mc_override_kp", avg_kp_auto)
    avg_kd = st.session_state.get("mc_override_kd", avg_kd_auto)
    avg_q  = st.session_state.get("mc_override_q",  avg_q_auto)
    if (avg_kp != avg_kp_auto or avg_kd != avg_kd_auto or avg_q != avg_q_auto):
        st.warning(
            f"⚠️ 目前使用覆蓋參數：Kp={avg_kp:.4f}, Kd={avg_kd:.4f}, Q={avg_q:.6f}，"
            f"非 Walk-Forward 平均值"
        )
    # ── 主設定 form ──
    with st.form("form_mc"):
        col1, col2 = st.columns(2)
        with col1:
            n_paths = st.number_input(
                "模擬路徑數", min_value=500, max_value=10000, value=5000, step=500
            )
        with col2:
            block_size = st.number_input(
                "區塊大小（天）", min_value=5, max_value=63, value=20, step=1,
                help="20=約一個月，63=約一季"
            )
        col1, col2 = st.columns(2)
        with col1:
            random_seed = st.number_input(
                "隨機種子", min_value=0, max_value=9999, value=42
            )
        st.markdown("**Smart Pilot Deadband 掃描範圍**")
        col1, col2, col3 = st.columns(3)
        with col1:
            mc_db_min = st.number_input(
                "Deadband 最小值", value=0.005, min_value=0.001, step=0.005, format="%.3f"
            )
        with col2:
            mc_db_max = st.number_input(
                "Deadband 最大值", value=0.10, min_value=0.01, step=0.01, format="%.3f"
            )
        with col3:
            mc_db_points = st.number_input(
                "Deadband 點數", value=10, min_value=3, step=1
            )
        st.markdown("**TO / TAT Tolerance 掃描範圍**")
        col1, col2, col3 = st.columns(3)
        with col1:
            mc_tol_min = st.number_input(
                "Tolerance 最小值", value=0.005, min_value=0.001, step=0.005, format="%.3f"
            )
        with col2:
            mc_tol_max = st.number_input(
                "Tolerance 最大值", value=0.15, min_value=0.01, step=0.01, format="%.3f"
            )
        with col3:
            mc_tol_points = st.number_input(
                "Tolerance 點數", value=10, min_value=3, step=1
            )
        submitted_mc = st.form_submit_button(
            "▶ 執行 Walk-Forward MC", type="primary", use_container_width=True
        )
    # ── Hash 比對 ──
    mc_hash_params = {
        "ticker1": params["ticker1"], "ticker2": params["ticker2"],
        "start_date": str(params["start_date"]), "end_date": str(params["end_date"]),
        "target_w": params["target_w"], "fee_rate": params["fee_rate"],
        "kf_r": params["kf_r"], "d_clip": params["d_clip"], "output_clip": params["output_clip"],
        "norm_ref_rmse": params["norm_ref_rmse"], "norm_ref_cost": params["norm_ref_cost"],
        "avg_kp": round(avg_kp, 6), "avg_kd": round(avg_kd, 6), "avg_q": round(avg_q, 8),
        "n_paths": int(n_paths), "block_size": int(block_size),
        "n_days_simulate": n_days_simulate, "random_seed": int(random_seed),
        "mc_db_min": mc_db_min, "mc_db_max": mc_db_max, "mc_db_points": int(mc_db_points),
        "mc_tol_min": mc_tol_min, "mc_tol_max": mc_tol_max, "mc_tol_points": int(mc_tol_points),
        "mc_hist_start": str(mc_hist_start),
        "mc_hist_end":   str(mc_hist_end),
        "mc_sim_start":  str(mc_sim_start),
        "mc_sim_end":    str(mc_sim_end),
    }
    current_hash = make_params_hash(mc_hash_params)
    mc_cache_path = Path("data/cache/mc_results.json")
    mc_result = None
    if submitted_mc:
        if mc_cache_path.exists():
            with open(mc_cache_path, encoding="utf-8") as f:
                mc_cache = json.load(f)
            if mc_cache.get("params_hash") == current_hash:
                st.success("✅ 參數未變，使用上次 MC 結果")
                mc_result = {k: np.array(v) for k, v in mc_cache["data"].items()}
        if mc_result is None:
            deadband_values = np.linspace(mc_db_min, mc_db_max, int(mc_db_points)).tolist()
            tol_values      = np.linspace(mc_tol_min, mc_tol_max, int(mc_tol_points)).tolist()
            mc_progress = st.progress(0)
            mc_status   = st.empty()
            mc_status.text(
                f"矩陣化生成 {int(n_paths)} 條路徑（{mc_sim_start}~{mc_sim_end}）並平行回測中..."
            )
            t0 = time.time()
            # ── DEBUG ──
            print(f"warmup_stock len={len(warmup_prices_stock)}, warmup_bond len={len(warmup_prices_bond)}")
            print(f"hist_stock len={len(hist_prices_stock)}, hist_bond len={len(hist_prices_bond)}")
            print(f"NaN in hist_stock: {np.isnan(hist_prices_stock).sum()}")
            print(f"NaN in hist_bond:  {np.isnan(hist_prices_bond).sum()}")
            print(f"n_days_simulate={n_days_simulate}, block_size={int(block_size)}")
            from validation.monte_carlo import run_wf_monte_carlo
            mc_result = run_wf_monte_carlo(
                hist_prices_stock=hist_prices_stock,
                hist_prices_bond=hist_prices_bond,
                n_paths=int(n_paths),
                n_days_simulate=n_days_simulate,
                block_size=int(block_size),
                avg_kp=avg_kp, avg_kd=avg_kd, avg_q=avg_q,
                kf_r=params["kf_r"],
                target_w=params["target_w"],
                fee_rate=params["fee_rate"],
                d_clip=params["d_clip"],
                output_clip=params["output_clip"],
                warmup_prices_stock=warmup_prices_stock,
                warmup_prices_bond=warmup_prices_bond,
                deadband_values=deadband_values,
                tol_values=tol_values,
                norm_ref_rmse=params["norm_ref_rmse"],
                norm_ref_cost=params["norm_ref_cost"],
                random_seed=int(random_seed),
                n_jobs=params["n_jobs"],
            )
            elapsed = time.time() - t0
            mc_progress.progress(1.0)
            mc_status.text(f"完成！{int(n_paths)} 條路徑，耗時 {elapsed:.1f} 秒")
            mc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mc_cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "data": {k: v.tolist() for k, v in mc_result.items()},
                    "params_hash": current_hash,
                }, f, ensure_ascii=False, indent=2)
    elif mc_cache_path.exists():
        with open(mc_cache_path, encoding="utf-8") as f:
            mc_cache = json.load(f)
        mc_result = {k: np.array(v) for k, v in mc_cache["data"].items()}
        if mc_cache.get("params_hash") != current_hash:
            st.warning("⚠️ 目前參數與快取結果不符，如需更新請重新執行")
    else:
        st.info("請設定好參數後按「▶ 執行 Walk-Forward MC」")
        return
    if mc_result is None:
        return

    # ── 取出陣列 ──
    sp_hv,  sp_ret,  sp_vol,  sp_trades  = (mc_result["sp_hv"],  mc_result["sp_ret"],
                                              mc_result["sp_vol"],  mc_result["sp_trades"])
    to_hv,  to_ret,  to_vol,  to_trades  = (mc_result["to_hv"],  mc_result["to_ret"],
                                              mc_result["to_vol"],  mc_result["to_trades"])
    tat_hv, tat_ret, tat_vol, tat_trades = (mc_result["tat_hv"], mc_result["tat_ret"],
                                              mc_result["tat_vol"], mc_result["tat_trades"])
    excess_sp_vs_to  = sp_ret - to_ret
    excess_sp_vs_tat = sp_ret - tat_ret
    sp_dominates = float(np.mean((sp_ret > to_ret) & (sp_ret > tat_ret)))

    # ── 摘要表格 ──
    def pct_range(arr, lo=5, hi=95):
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    summary_rows = []
    for name, hvs, vols, rets in [
        ("Smart Pilot",        sp_hv,  sp_vol,  sp_ret),
        ("Threshold-only",     to_hv,  to_vol,  to_ret),
        ("Time-and-threshold", tat_hv, tat_vol, tat_ret),
    ]:
        hv_lo,  hv_hi  = pct_range(hvs)
        vol_lo, vol_hi = pct_range(vols)
        summary_rows.append({
            "策略":           name,
            "HV 中位數":      f"{np.median(hvs):.4f}",
            "HV 5th~95th":   f"{hv_lo:.4f} ~ {hv_hi:.4f}",
            "年化波動 中位數": f"{np.median(vols)*100:.2f}%",
            "波動 5th~95th":  f"{vol_lo*100:.2f}% ~ {vol_hi*100:.2f}%",
            "賺錢機率":       f"{np.mean(rets > 0)*100:.1f}%",
        })
    st.subheader("摘要統計")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.metric("Smart Pilot 相對勝率（同時贏過兩對手）", f"{sp_dominates*100:.1f}%")

    # ── 圖一：HV 分布 ──
    st.markdown("---")
    fig1 = go.Figure()
    for arr, name, color in [
        (sp_hv,  "Smart Pilot",        "#FFD700"),
        (to_hv,  "Threshold-only",     "#808080"),
        (tat_hv, "Time-and-threshold", "#00CED1"),
    ]:
        fig1.add_trace(go.Histogram(x=arr, name=name, opacity=0.7,
                                    marker_color=color, nbinsx=60,
                                    hovertemplate="HV: %{x:.4f}<br>Count: %{y}<extra>" + name + "</extra>"))
        fig1.add_vline(x=float(np.median(arr)), line_dash="dash", line_color=color,
                       annotation_text=f"{name}: {np.median(arr):.4f}",
                       annotation_position="top")
    fig1.update_layout(
        title="圖一：HV 分布（三策略，5000 條路徑）",
        xaxis_title="Hypervolume", yaxis_title="路徑數",
        barmode="overlay", height=450, template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── 圖二：年化波動率分布 ──
    fig2 = go.Figure()
    for arr, name, color in [
        (sp_vol,  "Smart Pilot",        "#FFD700"),
        (to_vol,  "Threshold-only",     "#808080"),
        (tat_vol, "Time-and-threshold", "#00CED1"),
    ]:
        fig2.add_trace(go.Histogram(x=arr*100, name=name, opacity=0.7,
                                    marker_color=color, nbinsx=60,
                                    hovertemplate="波動率: %{x:.2f}%<br>Count: %{y}<extra>" + name + "</extra>"))
        fig2.add_vline(x=float(np.median(arr)*100), line_dash="dash", line_color=color,
                       annotation_text=f"{name}: {np.median(arr)*100:.2f}%",
                       annotation_position="top")
    fig2.update_layout(
        title="圖二：年化波動率分布（三策略，5000 條路徑）",
        xaxis_title="年化波動率 (%)", yaxis_title="路徑數",
        barmode="overlay", height=450, template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 圖三：超額報酬分布 ──
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=[
        "SP − Threshold-only 年化超額報酬",
        "SP − Time-and-threshold 年化超額報酬",
    ])
    for col_idx, (excess, color, label) in enumerate([
        (excess_sp_vs_to,  "#FF6B6B", "SP − TO"),
        (excess_sp_vs_tat, "#A78BFA", "SP − TAT"),
    ], start=1):
        pos_ratio  = float(np.mean(excess > 0))
        median_val = float(np.median(excess))
        xref = "x domain"  if col_idx == 1 else "x2 domain"
        yref = "y domain"  if col_idx == 1 else "y2 domain"
        fig3.add_trace(go.Histogram(
            x=excess*100, marker_color=color, opacity=0.8, nbinsx=60, name=label,
            hovertemplate="超額報酬: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ), row=1, col=col_idx)
        fig3.add_vline(x=0, line_dash="solid", line_color="white",
                       line_width=1.5, row=1, col=col_idx)
        fig3.add_vline(x=median_val*100, line_dash="dash", line_color=color,
                       annotation_text=f"中位數: {median_val*100:.3f}%",
                       annotation_position="top", row=1, col=col_idx)
        fig3.add_annotation(
            x=0.05, y=0.95, xref=xref, yref=yref,
            text=f"SP 勝率: {pos_ratio*100:.1f}%",
            showarrow=False, font=dict(color=color, size=13),
        )
    fig3.update_xaxes(title_text="年化超額報酬 (%)", row=1, col=1)
    fig3.update_xaxes(title_text="年化超額報酬 (%)", row=1, col=2)
    fig3.update_yaxes(title_text="路徑數", row=1, col=1)
    fig3.update_layout(
        title="圖三：Smart Pilot 超額報酬分布",
        height=450, showlegend=False, template="plotly_dark",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── 路徑明細表 ──
    n_paths_actual = len(sp_hv)
    st.markdown("---")
    st.subheader(f"{n_paths_actual} 條路徑明細（點擊欄位標題可排序）")
    detail_df = pd.DataFrame({
        "路徑":           np.arange(1, n_paths_actual + 1),
        "SP_HV":         np.round(sp_hv,   4),
        "SP_年化報酬%":  np.round(sp_ret  * 100, 2),
        "SP_年化波動%":  np.round(sp_vol  * 100, 2),
        "SP_交易次數":   sp_trades.astype(int),
        "TO_HV":         np.round(to_hv,   4),
        "TO_年化報酬%":  np.round(to_ret  * 100, 2),
        "TO_年化波動%":  np.round(to_vol  * 100, 2),
        "TO_交易次數":   to_trades.astype(int),
        "TAT_HV":        np.round(tat_hv,  4),
        "TAT_年化報酬%": np.round(tat_ret * 100, 2),
        "TAT_年化波動%": np.round(tat_vol * 100, 2),
        "TAT_交易次數":  tat_trades.astype(int),
        "SP超額_vs_TO%":  np.round(excess_sp_vs_to  * 100, 2),
        "SP超額_vs_TAT%": np.round(excess_sp_vs_tat * 100, 2),
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)


# =============================================================================
# 主程式
# =============================================================================
def main():
    """主程式入口"""
    # 側邊欄
    params = render_sidebar()

    # 標題
    st.title("Smart Pilot 投資組合再平衡系統")
    st.markdown("""
    基於 **PD 控制器 + Log-Space 卡爾曼濾波** 的智慧再平衡系統 (v3.0)。
    """)

    # 載入資料（含暖機資料）
    try:
        result = load_data(
            tickers=[params["ticker1"], params["ticker2"]],
            start_date=str(params["start_date"]),
            end_date=str(params["end_date"]),
            warmup_days=params["warmup"],
        )
        data = result["backtest_data"][[params["ticker1"], params["ticker2"]]]
        warmup_prices_stock = result["warmup_data"][params["ticker1"]].values
        warmup_prices_bond = result["warmup_data"][params["ticker2"]].values

        # 警告：暖機資料不足
        if result["actual_warmup_days"] < params["warmup"]:
            st.warning(
                f"警告：只取得 {result['actual_warmup_days']} 天暖機資料，"
                f"少於設定的 {params['warmup']} 天，"
                "建議將回測起始日往後移或縮短暖機天數"
            )
    except Exception as e:
        st.error(f"資料載入失敗：{str(e)}")
        return

    # 四個分頁
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pareto Frontier",
        "Heatmap",
        "Walk-Forward",
        "Monte Carlo"
    ])

    with tab1:
        render_tab_pareto(params, data, warmup_prices_stock, warmup_prices_bond)

    with tab2:
        render_tab_heatmap(params, data, warmup_prices_stock, warmup_prices_bond)

    with tab3:
        render_tab_walking_forward(params, data, warmup_prices_stock, warmup_prices_bond)

    with tab4:
        render_tab_monte_carlo(params, data, warmup_prices_stock, warmup_prices_bond)


if __name__ == "__main__":
    main()
