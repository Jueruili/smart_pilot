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

import json
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
from pathlib import Path

from data.data_loader import DataLoader
from core.benchmark import (
    run_smart_pilot, run_threshold_only, run_time_and_threshold,
    scan_pareto_frontier, get_metrics, calc_rmse
)
from core.optimizer import calc_hypervolume, compare_hypervolumes


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
    # 區塊 1：標的設定
    # =========================================================================
    st.sidebar.header("📈 標的設定")

    ticker1 = st.sidebar.text_input("股票標的", value="VTI").upper().strip()
    ticker2 = st.sidebar.text_input("債券標的", value="BND").upper().strip()

    target_w = st.sidebar.slider(
        "目標股票比例",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        format="%.2f"
    )

    start_date = st.sidebar.date_input(
        "回測開始日期",
        value=date(2013, 1, 1)
    )

    end_date = st.sidebar.date_input(
        "回測結束日期",
        value=date(2024, 1, 1)
    )

    fee_rate = st.sidebar.number_input(
        "手續費率",
        value=0.003,
        step=0.001,
        format="%.3f"
    )

    # =========================================================================
    # 區塊 2：卡爾曼濾波器設定
    # =========================================================================
    st.sidebar.header("🔧 卡爾曼濾波器設定")

    warmup = st.sidebar.number_input(
        "暖機天數",
        min_value=10,
        max_value=100,
        value=20,
        step=1,
        help="回測起始日前的暖機天數，使用前一年底的歷史資料初始化 KF，建議 20 天"
    )

    kf_r = st.sidebar.number_input(
        "R 值（觀測雜訊）",
        value=0.005,
        min_value=0.0001,
        step=0.001,
        format="%.4f",
        help="R 越大越平滑但反應越慢，建議 0.001~0.01"
    )

    d_clip = st.sidebar.number_input(
        "D Clip",
        value=0.15,
        min_value=0.01,
        step=0.01,
        format="%.2f",
        help="D 項裁切閾值，預設 0.15"
    )

    output_clip = st.sidebar.number_input(
        "Output Clip",
        value=0.2,
        min_value=0.01,
        step=0.01,
        format="%.2f",
        help="總輸出裁切閾值，預設 0.2"
    )

    # =========================================================================
    # 區塊 3：貝氏最佳化設定
    # =========================================================================
    st.sidebar.header("🔍 貝氏最佳化設定")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        bayes_kp_min = st.number_input("Kp 下限", value=0.01, min_value=0.001,
                                        step=0.01, format="%.3f", key="bayes_kp_min")
        bayes_kd_min = st.number_input("Kd 下限", value=0.01, min_value=0.001,
                                        step=0.01, format="%.3f", key="bayes_kd_min")
    with col2:
        bayes_kp_max = st.number_input("Kp 上限", value=5.0, min_value=0.1,
                                        step=0.5, format="%.1f", key="bayes_kp_max")
        bayes_kd_max = st.number_input("Kd 上限", value=5.0, min_value=0.1,
                                        step=0.5, format="%.1f", key="bayes_kd_max")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        bayes_q_min = st.number_input("Q 下限", value=0.00001, min_value=0.000001,
                                       format="%.5f", key="bayes_q_min")
    with col2:
        bayes_q_max = st.number_input("Q 上限", value=1.0, min_value=0.00001,
                                       format="%.4f", key="bayes_q_max")

    bayes_n_trials = st.sidebar.number_input(
        "試驗次數 (trials)", value=50, min_value=10, step=10, key="bayes_n_trials"
    )

    # =========================================================================
    # 區塊 4：執行計算（核心數設定）
    # =========================================================================
    st.sidebar.header("🚀 執行計算")

    n_cores = os.cpu_count() or 1
    st.sidebar.write(f"你的電腦有 {n_cores} 個核心")
    n_jobs = st.sidebar.slider(
        "使用核心數",
        min_value=1,
        max_value=n_cores,
        value=max(1, n_cores - 1)
    )

    if st.sidebar.button("▶ 執行貝氏最佳化（約 3-8 分鐘）", type="primary"):
        with st.spinner("載入資料..."):
            result_data = load_data(
                tickers=[ticker1, ticker2],
                start_date=str(start_date),
                end_date=str(end_date),
                warmup_days=warmup,
            )
            full_backtest  = result_data["backtest_data"]
            wm_prices_stock = result_data["warmup_data"][ticker1].values
            wm_prices_bond  = result_data["warmup_data"][ticker2].values
            prices_stock_bt = full_backtest[ticker1].values
            prices_bond_bt  = full_backtest[ticker2].values
            dates_bt        = full_backtest.index.tolist()
            rets_stock_bt   = np.diff(prices_stock_bt) / prices_stock_bt[:-1]
            rets_bond_bt    = np.diff(prices_bond_bt)  / prices_bond_bt[:-1]
            prices_stock_bt = prices_stock_bt[1:]
            prices_bond_bt  = prices_bond_bt[1:]
            dates_bt        = dates_bt[1:]

        n_trials_int = int(bayes_n_trials)
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        status_text.text(f"貝氏最佳化進度：0/{n_trials_int}")

        t0 = time.time()
        from core.optimizer import run_bayesian_opt
        bayes_result = run_bayesian_opt(
            rets_stock_bt, rets_bond_bt,
            prices_stock_bt, prices_bond_bt, dates_bt,
            target_w=target_w,
            fee_rate=fee_rate,
            deadband_values=np.linspace(0.005, 0.10, 15).tolist(),
            kf_r=kf_r,
            warmup=warmup,
            warmup_prices_stock=wm_prices_stock,
            warmup_prices_bond=wm_prices_bond,
            ref_rmse=0.08,
            ref_cost=0.0004,
            kp_min=bayes_kp_min, kp_max=bayes_kp_max,
            kd_min=bayes_kd_min, kd_max=bayes_kd_max,
            q_min=bayes_q_min,   q_max=bayes_q_max,
            n_trials=n_trials_int,
            n_jobs=n_jobs,
            d_clip=d_clip,
            output_clip=output_clip,
            progress_bar=progress_bar,
            status_text=status_text,
        )
        elapsed = time.time() - t0

        progress_bar.progress(1.0)
        status_text.text(f"完成！{n_trials_int} 次試驗，耗時 {elapsed:.1f} 秒")

        cache_path = Path("data/cache/bayesian_opt_results.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "kp": bayes_result["kp"],
                "kd": bayes_result["kd"],
                "q":  bayes_result["q"],
                "hypervolume": bayes_result["hypervolume"],
                "hypervolume_x10000": bayes_result["hypervolume_x10000"],
                "n_trials": bayes_result["n_trials"],
            }, f, ensure_ascii=False, indent=2)

        st.sidebar.success(
            f"貝氏最佳化完成！耗時 {elapsed:.1f} 秒\n"
            f"最佳：Kp={bayes_result['kp']:.3f}, "
            f"Kd={bayes_result['kd']:.3f}, "
            f"Q={bayes_result['q']:.6f}\n"
            f"HV={bayes_result['hypervolume']:.6f}"
            f"（×10000 = {bayes_result['hypervolume_x10000']:.4f}）"
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
        grid_path = Path("data/cache/grid_search_results.json")
        if grid_path.exists():
            size_kb = grid_path.stat().st_size // 1024
            st.write(f"✅ grid_search_results.json（{size_kb} KB）")
        else:
            st.write("❌ grid_search_results.json（尚未計算）")

        bayes_path = Path("data/cache/bayesian_opt_results.json")
        if bayes_path.exists():
            size_kb = bayes_path.stat().st_size // 1024
            st.write(f"✅ bayesian_opt_results.json（{size_kb} KB）")
        else:
            st.write("❌ bayesian_opt_results.json（尚未計算）")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("刪除 Grid Search"):
                if grid_path.exists():
                    grid_path.unlink()
                    load_grid_cache.clear()
                    st.warning("已刪除，需重新執行 Grid Search")
        with col2:
            if st.button("刪除貝氏最佳化"):
                if bayes_path.exists():
                    bayes_path.unlink()
                    st.warning("已刪除，需重新執行貝氏最佳化")

    # =========================================================================
    # 回傳參數
    # =========================================================================
    return {
        "ticker1": ticker1,
        "ticker2": ticker2,
        "target_w": target_w,
        "start_date": start_date,
        "end_date": end_date,
        "fee_rate": fee_rate,
        "kp": 0.5,
        "kd": 0.5,
        "kf_q": 0.001,
        "kf_r": kf_r,
        "deadband": 0.02,
        "warmup": warmup,
        "deadband_values": np.linspace(0.005, 0.10, 15).tolist(),
        "n_jobs": n_jobs,
        "d_clip": d_clip,
        "output_clip": output_clip,
        "bayes_kp_min": bayes_kp_min,
        "bayes_kp_max": bayes_kp_max,
        "bayes_kd_min": bayes_kd_min,
        "bayes_kd_max": bayes_kd_max,
        "bayes_q_min":  bayes_q_min,
        "bayes_q_max":  bayes_q_max,
        "bayes_n_trials": bayes_n_trials,
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

    # 準備資料
    prices_stock = data[params["ticker1"]].values
    prices_bond = data[params["ticker2"]].values
    dates = data.index.tolist()
    n_years = len(dates) / 252.0

    rets_stock = np.zeros(len(prices_stock))
    rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
    rets_bond = np.zeros(len(prices_bond))
    rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

    # 執行 Pareto Scan
    with st.spinner("掃描 Pareto Frontier..."):
        pareto = scan_pareto_frontier(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=params["target_w"],
            fee_rate=params["fee_rate"],
            kf_q=params["kf_q"],
            kf_r=params["kf_r"],
            kp=params["kp"],
            kd=params["kd"],
            n_points=30,
            warmup=params["warmup"],
            warmup_prices_stock=warmup_prices_stock,
            warmup_prices_bond=warmup_prices_bond,
        )

    # 計算超體積（使用 ref_multiplier）
    tat_pts = [{"rmse": p["rmse"], "cost": p["ann_cost"]}
               for p in pareto["time_and_threshold"]]
    all_pts = pareto["smart_pilot"] + pareto["threshold_only"] + tat_pts
    ref_rmse = max(p["rmse"] for p in all_pts) * 1.1
    ref_cost = max(p["cost"] for p in all_pts) * 1.1
    reference_point = {"rmse": ref_rmse, "cost": ref_cost}

    hv_sp = calc_hypervolume(pareto["smart_pilot"], reference_point)
    hv_to = calc_hypervolume(pareto["threshold_only"], reference_point)
    hv_tat = calc_hypervolume(tat_pts, reference_point)

    # 決定贏家
    hv_values = {"smart_pilot": hv_sp, "threshold_only": hv_to, "time_and_threshold": hv_tat}
    winner = max(hv_values, key=hv_values.get)
    hv_comparison = {
        "smart_pilot": hv_sp,
        "threshold_only": hv_to,
        "time_and_threshold": hv_tat,
        "winner": winner,
        "reference_point": reference_point,
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
        f"參考點倍數：1.1x | "
        f"參考點：RMSE={ref_rmse:.5f}，Cost={ref_cost:.6f}"
    )

    with st.expander("📊 標準化 Pareto 圖", expanded=True):

        col1, col2 = st.columns(2)
        with col1:
            norm_ref_rmse_pct = st.number_input(
                "參考 RMSE (%)",
                min_value=0.01, value=8.0, step=0.5, format="%.2f",
                help="x 軸最大值，單位 %，例如 8 代表 8%",
                key="pareto_norm_ref_rmse"
            )
        with col2:
            norm_ref_cost_pct = st.number_input(
                "參考 Cost (%/年)",
                min_value=0.001, value=0.04, step=0.005, format="%.3f",
                help="y 軸最大值，單位 %/年，例如 0.04 代表 0.04%/年",
                key="pareto_norm_ref_cost"
            )
        norm_ref_rmse = norm_ref_rmse_pct / 100
        norm_ref_cost = norm_ref_cost_pct / 100

        # 標準化函式
        def normalize_points(points):
            return [
                {
                    "rmse_norm": p["rmse"] / norm_ref_rmse,
                    "cost_norm": p["ann_cost"] / norm_ref_cost,
                }
                for p in points
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

    # ── Grid Search 參數設定 ──
    with st.expander("⚙️ Grid Search 參數設定", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            gs_kp_min = st.number_input("Kp 最小值", value=0.1, step=0.1, format="%.1f", key="gs_kp_min")
        with col2:
            gs_kp_max = st.number_input("Kp 最大值", value=1.0, step=0.1, format="%.1f", key="gs_kp_max")
        with col3:
            gs_kp_points = st.number_input("Kp 點數", value=10, min_value=3, step=1, key="gs_kp_points")

        col1, col2, col3 = st.columns(3)
        with col1:
            gs_kd_min = st.number_input("Kd 最小值", value=0.1, step=0.1, format="%.1f", key="gs_kd_min")
        with col2:
            gs_kd_max = st.number_input("Kd 最大值", value=1.0, step=0.1, format="%.1f", key="gs_kd_max")
        with col3:
            gs_kd_points = st.number_input("Kd 點數", value=10, min_value=3, step=1, key="gs_kd_points")

        col1, col2, col3 = st.columns(3)
        with col1:
            gs_q_min = st.number_input("Q 最小值", value=0.00001, min_value=0.000001, format="%.5f", key="gs_q_min")
        with col2:
            gs_q_max = st.number_input("Q 最大值", value=0.1, min_value=0.00001, format="%.4f", key="gs_q_max")
        with col3:
            gs_q_points = st.number_input("Q 點數", value=5, min_value=2, step=1, key="gs_q_points")
        gs_q_values = np.logspace(np.log10(gs_q_min), np.log10(gs_q_max), int(gs_q_points)).tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            gs_db_min = st.number_input("Deadband 最小值", value=0.005, step=0.005, format="%.3f", key="gs_db_min")
        with col2:
            gs_db_max = st.number_input("Deadband 最大值", value=0.10, step=0.005, format="%.3f", key="gs_db_max")
        with col3:
            gs_db_points = st.number_input("Deadband 點數", value=15, min_value=3, step=1, key="gs_db_points")

        gs_kp_range  = np.linspace(gs_kp_min, gs_kp_max, int(gs_kp_points)).tolist()
        gs_kd_range  = np.linspace(gs_kd_min, gs_kd_max, int(gs_kd_points)).tolist()
        gs_db_values = np.linspace(gs_db_min, gs_db_max, int(gs_db_points)).tolist()
        gs_total_tasks = len(gs_q_values) * len(gs_kp_range) * len(gs_kd_range)

        st.info(f"共 {gs_total_tasks} 組參數，使用 {params['n_jobs']} 核心平行運算")
        gs_progress_bar = st.progress(0)
        gs_status_text  = st.empty()

        if st.button("▶ 執行 Grid Search（約 2-5 分鐘）", type="primary", key="btn_grid_search"):
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

            gs_status_text.text(f"Grid Search 進度：0/{gs_total_tasks}")
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
                ref_multiplier=1.1,
                warmup_prices_stock=wm_stock,
                warmup_prices_bond=wm_bond,
                progress_bar=gs_progress_bar,
                status_text=gs_status_text,
                total_tasks=gs_total_tasks,
            )
            best = find_best_from_grid(results)
            elapsed = time.time() - t0
            gs_progress_bar.progress(1.0)
            gs_status_text.text(f"完成！共 {gs_total_tasks} 組，耗時 {elapsed:.1f} 秒")

            cache_path = Path("data/cache/grid_search_results.json")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "results": results,
                    "best": best,
                    "metadata": {
                        "q_values":        gs_q_values,
                        "kp_range":        gs_kp_range,
                        "kd_range":        gs_kd_range,
                        "deadband_values": gs_db_values,
                    }
                }, f, ensure_ascii=False, indent=2)
            load_grid_cache.clear()
            st.success(
                f"Grid Search 完成！共 {len(results)} 組，耗時 {elapsed:.1f} 秒\n"
                f"最佳：Kp={best['kp']:.2f}, Kd={best['kd']:.2f}, "
                f"Q={best['q']:.5f}, HV={best['hypervolume']*10000:.4f}"
            )

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
        heatmap_data[i, j] = r["hypervolume"] * 10000

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"{kp:.2f}" for kp in kp_vals],
        y=[f"{kd:.2f}" for kd in kd_vals],
        colorscale="Viridis",
        colorbar=dict(title="Hypervolume (%%)"),
        hovertemplate="Kp: %{x}<br>Kd: %{y}<br>HV: %{z:.4f} %%<extra></extra>"
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
        st.metric("Hypervolume", f"{best['hypervolume'] * 10000:.4f} %%")

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
            st.metric(
                "Hypervolume",
                f"{bayes_cache['hypervolume']:.6f}",
                help=f"×10000 = {bayes_cache['hypervolume_x10000']:.4f}"
            )
        st.caption(f"試驗次數：{bayes_cache.get('n_trials', 'N/A')}")
    else:
        st.info("尚未執行貝氏最佳化，請在側邊欄執行。")

    # =========================================================================
    # 單點快速測試
    # =========================================================================
    st.markdown("---")
    with st.expander("🧪 單點快速測試", expanded=False):
        st.markdown("輸入一組參數，快速計算這組參數的超體積和回測指標。")
        st.markdown("目前參考點倍數：**1.1x**")

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

                # 計算參考點（用 Threshold-only 最差點 × ref_multiplier）
                all_pts = pareto["smart_pilot"] + pareto["threshold_only"]
                ref_rmse = max(p["rmse"] for p in all_pts) * 1.1
                ref_cost = max(p["ann_cost"] for p in all_pts) * 1.1
                reference_point = {"rmse": ref_rmse, "cost": ref_cost}
                hv = calc_hypervolume(pareto["smart_pilot"], reference_point)

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
                st.metric("超體積", f"{hv * 10000:.4f} %%")
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

            # 顯示使用的參考點數值
            st.caption(
                f"參考點：RMSE={ref_rmse:.5f}，Cost={ref_cost:.6f} "
                f"（Threshold-only 最差點 × 1.1）"
            )

            # 和 Grid Search 最佳結果比較
            grid_cache = load_grid_cache()
            if grid_cache is not None:
                best = grid_cache["best"]
                st.markdown("**與 Grid Search 最佳結果比較：**")
                delta_hv = hv - best["hypervolume"]
                if delta_hv >= 0:
                    st.success(
                        f"這組參數的超體積比 Grid Search 最佳結果高 {delta_hv * 10000:.4f} %%"
                    )
                else:
                    st.info(
                        f"Grid Search 最佳：Kp={best['kp']:.2f}, "
                        f"Kd={best['kd']:.2f}, Q={best['q']:.5f}, "
                        f"HV={best['hypervolume'] * 10000:.4f} %%（差距 {abs(delta_hv) * 10000:.4f} %%）"
                    )
            else:
                st.caption("尚未執行 Grid Search，無法比較最佳結果")


# =============================================================================
# Tab 3: Rolling Window
# =============================================================================
def render_tab_rolling(params: dict, data: pd.DataFrame,
                       warmup_prices_stock: np.ndarray,
                       warmup_prices_bond: np.ndarray):
    """Walk-Forward 滾動窗口分析"""
    st.header("Walk-Forward Analysis - 滾動窗口分析")

    st.markdown("""
    每一輪包含：
    - **IS（樣本內）**：貝氏最佳化尋找最佳 (Kp, Kd, Q)，最大化標準化超體積
    - **OOS（樣本外）**：套用 IS 最佳參數，三策略直接回測，不再最佳化
    - **窗口步進**：可自訂步進年數（預設 1 年）
    - **OOS HV 參考點**：與 IS 相同的固定參考點，IS/OOS HV 直接可比
    """)

    # ── Walk-Forward 參數設定 ──
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

    st.markdown("---")
    st.markdown("**OOS HV 參考點設定**（與 IS 共用同一組，HV 才能直接比較）")
    col1, col2 = st.columns(2)
    with col1:
        wf_ref_rmse_pct = st.number_input(
            "參考 RMSE (%)",
            min_value=0.01, value=8.0, step=0.5, format="%.2f",
            help="x 軸最大值，單位 %，例如 8 代表 8%",
            key="wf_ref_rmse"
        )
    with col2:
        wf_ref_cost_pct = st.number_input(
            "參考 Cost (%/年)",
            min_value=0.001, value=0.08, step=0.01, format="%.3f",
            help="y 軸最大值，單位 %/年，例如 0.08 代表 0.08%/年",
            key="wf_ref_cost"
        )
    wf_norm_ref_rmse = wf_ref_rmse_pct / 100
    wf_norm_ref_cost = wf_ref_cost_pct / 100

    prices_stock = data[params["ticker1"]].values
    prices_bond  = data[params["ticker2"]].values
    dates        = data.index.tolist()
    total_days   = len(dates)

    # 預估輪數
    is_days   = int(is_years)  * 252
    oos_days  = int(oos_years) * 252
    step_days = int(step_years) * 252
    n_rounds_est = max(0, (total_days - is_days - oos_days) // step_days + 1)
    st.info(
        f"預估輪數：約 {n_rounds_est} 輪 | "
        f"每輪 {int(params['bayes_n_trials'])} 次貝氏試驗 | "
        f"步進 {int(step_years)} 年 | "
        f"資料總長：{total_days} 天（{total_days/252:.1f} 年）"
    )

    if n_rounds_est == 0:
        st.warning("資料長度不足以完成一輪 IS+OOS，請縮短 IS/OOS 年數或延長回測區間")
        return

    if st.button("▶ 執行 Walk-Forward 分析", type="primary"):
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
                n_trials=int(params["bayes_n_trials"]),
                deadband_values=params["deadband_values"],
                norm_ref_rmse=wf_norm_ref_rmse,
                norm_ref_cost=wf_norm_ref_cost,
                kp_min=params["bayes_kp_min"],
                kp_max=params["bayes_kp_max"],
                kd_min=params["bayes_kd_min"],
                kd_max=params["bayes_kd_max"],
                q_min=params["bayes_q_min"],
                q_max=params["bayes_q_max"],
                n_jobs=params["n_jobs"],
            )

        rounds = wf_result["rounds"]
        if not rounds:
            st.warning("未能完成任何一輪，請調整參數")
            return

        st.success(f"Walk-Forward 完成！共 {len(rounds)} 輪")

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
            f"OOS HV 使用固定參考點（RMSE={wf_ref_rmse_pct:.1f}%，"
            f"Cost={wf_ref_cost_pct:.3f}%/年），與 IS HV 同一尺度，衰退比值有意義。"
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

        # ── 表格一：OOS 綜合績效指標比較 ──
        st.subheader("表格一：OOS 綜合績效指標比較（所有輪次平均）")

        def collect_metrics(key):
            rows = []
            for r in rounds:
                m = r[key]["metrics"]
                rows.append({
                    "交易次數":   r[key]["trade_count"],
                    "總周轉率":   r[key]["turnover"],
                    "RMSE":      r[key]["rmse"],
                    "年化報酬率": m["ann_return"],
                    "夏普值":     m["sharpe"],
                    "最大回撤":   m["max_drawdown"],
                    "年化波動率": m["ann_wealth_vol"],
                })
            return pd.DataFrame(rows).mean()

        avg_sp  = collect_metrics("oos_sp")
        avg_to  = collect_metrics("oos_to")
        avg_tat = collect_metrics("oos_tat")

        summary_df = pd.DataFrame({
            "策略": ["Smart Pilot", "Threshold-only", "Time-and-threshold"],
            "交易次數(均)":  [f"{avg_sp['交易次數']:.1f}",       f"{avg_to['交易次數']:.1f}",       f"{avg_tat['交易次數']:.1f}"],
            "總周轉率(均)":  [f"{avg_sp['總周轉率']:.3f}",       f"{avg_to['總周轉率']:.3f}",       f"{avg_tat['總周轉率']:.3f}"],
            "RMSE(均)":     [f"{avg_sp['RMSE']*100:.2f}%",      f"{avg_to['RMSE']*100:.2f}%",      f"{avg_tat['RMSE']*100:.2f}%"],
            "年化報酬(均)":  [f"{avg_sp['年化報酬率']*100:.2f}%",f"{avg_to['年化報酬率']*100:.2f}%",f"{avg_tat['年化報酬率']*100:.2f}%"],
            "Sharpe(均)":   [f"{avg_sp['夏普值']:.2f}",         f"{avg_to['夏普值']:.2f}",         f"{avg_tat['夏普值']:.2f}"],
            "MDD(均)":      [f"{avg_sp['最大回撤']*100:.2f}%",  f"{avg_to['最大回撤']*100:.2f}%",  f"{avg_tat['最大回撤']*100:.2f}%"],
            "年化波動率(均)":[f"{avg_sp['年化波動率']*100:.2f}%",f"{avg_to['年化波動率']*100:.2f}%",f"{avg_tat['年化波動率']*100:.2f}%"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

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
    """渲染蒙地卡羅模擬分頁"""
    st.header("Monte Carlo Simulation")

    st.markdown("""
    使用 Bootstrap 方法隨機重組歷史報酬率，模擬多次投資路徑，
    評估策略在不同市場情境下的表現分布。
    """)

    # 模擬設定
    col1, col2 = st.columns(2)
    with col1:
        n_simulations = st.slider(
            "模擬次數",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )
    with col2:
        random_seed = st.number_input(
            "隨機種子",
            min_value=0,
            max_value=9999,
            value=42,
            step=1
        )

    # 準備資料
    prices_stock = data[params["ticker1"]].values
    prices_bond = data[params["ticker2"]].values
    dates = data.index.tolist()
    n_days = len(dates)

    rets_stock = np.zeros(n_days)
    rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
    rets_bond = np.zeros(n_days)
    rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

    warmup = params["warmup"]

    if st.button("執行蒙地卡羅模擬", type="primary"):
        with st.spinner(f"執行 {n_simulations} 次模擬..."):
            np.random.seed(random_seed)

            # Bootstrap 模擬
            final_returns = []

            for _ in range(n_simulations):
                # 隨機重組報酬率
                indices = np.random.choice(
                    np.arange(warmup, n_days),
                    size=n_days - warmup,
                    replace=True
                )

                sim_rets_stock = rets_stock[indices]
                sim_rets_bond = rets_bond[indices]
                sim_prices_stock = prices_stock[0] * np.cumprod(1 + sim_rets_stock)
                sim_prices_bond = prices_bond[0] * np.cumprod(1 + sim_rets_bond)
                sim_dates = dates[warmup:]

                # 執行回測
                # 注意：KF 暖機用原始的 warmup_prices，每次模擬都用同一份初始化 KF
                result = run_smart_pilot(
                    np.concatenate([[0], sim_rets_stock]),
                    np.concatenate([[0], sim_rets_bond]),
                    np.concatenate([[prices_stock[0]], sim_prices_stock]),
                    np.concatenate([[prices_bond[0]], sim_prices_bond]),
                    sim_dates,
                    target_w=params["target_w"],
                    fee_rate=params["fee_rate"],
                    kf_q=params["kf_q"],
                    kf_r=params["kf_r"],
                    kp=params["kp"],
                    kd=params["kd"],
                    deadband=params["deadband"],
                    warmup=0,
                    warmup_prices_stock=warmup_prices_stock,
                    warmup_prices_bond=warmup_prices_bond,
                )

                final_nav = result["nav_list"][-1]
                final_returns.append(final_nav - 1.0)

            final_returns = np.array(final_returns)

        # 統計指標
        mean_return = float(np.mean(final_returns))
        median_return = float(np.median(final_returns))
        std_return = float(np.std(final_returns))
        prob_profit = float(np.mean(final_returns > 0))
        var_95 = float(np.percentile(final_returns, 5))
        var_99 = float(np.percentile(final_returns, 1))
        ruin_risk = float(np.mean(final_returns < -0.5))

        # 顯示關鍵指標
        st.subheader("關鍵統計指標")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("賺錢機率", f"{prob_profit:.1%}")
        with col2:
            st.metric("平均報酬", f"{mean_return:.2%}")
        with col3:
            st.metric("95% VaR", f"{var_95:.2%}")
        with col4:
            st.metric("破產風險", f"{ruin_risk:.2%}")

        # 報酬分布直方圖
        st.subheader("報酬率分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=final_returns * 100,
            nbinsx=100,
            name="報酬分布",
            marker_color="#1f77b4"
        ))

        # 添加參考線
        fig.add_shape(
            type="line",
            x0=0, x1=0, y0=0, y1=1, yref="paper",
            line=dict(color="black", width=2)
        )
        fig.add_annotation(
            x=0, y=1.02, yref="paper",
            text="盈虧分界", showarrow=False
        )

        fig.add_shape(
            type="line",
            x0=mean_return * 100, x1=mean_return * 100, y0=0, y1=1, yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=mean_return * 100, y=1.05, yref="paper",
            text=f"平均 {mean_return:.1%}", showarrow=False, font=dict(color="red")
        )

        fig.add_shape(
            type="line",
            x0=var_95 * 100, x1=var_95 * 100, y0=0, y1=1, yref="paper",
            line=dict(color="blue", width=2, dash="dot")
        )
        fig.add_annotation(
            x=var_95 * 100, y=0.95, yref="paper",
            text=f"95% VaR", showarrow=False, font=dict(color="blue")
        )

        fig.update_layout(
            xaxis_title="最終報酬率 (%)",
            yaxis_title="出現次數",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # 詳細統計表
        with st.expander("詳細統計"):
            stats_df = pd.DataFrame([
                {"統計量": "平均報酬率", "數值": f"{mean_return:.2%}"},
                {"統計量": "中位數報酬率", "數值": f"{median_return:.2%}"},
                {"統計量": "標準差", "數值": f"{std_return:.2%}"},
                {"統計量": "偏態", "數值": f"{float(pd.Series(final_returns).skew()):.2f}"},
                {"統計量": "峰態", "數值": f"{float(pd.Series(final_returns).kurtosis()):.2f}"},
                {"統計量": "5th 百分位", "數值": f"{float(np.percentile(final_returns, 5)):.2%}"},
                {"統計量": "25th 百分位", "數值": f"{float(np.percentile(final_returns, 25)):.2%}"},
                {"統計量": "75th 百分位", "數值": f"{float(np.percentile(final_returns, 75)):.2%}"},
                {"統計量": "95th 百分位", "數值": f"{float(np.percentile(final_returns, 95)):.2%}"},
                {"統計量": "99% VaR", "數值": f"{var_99:.2%}"},
                {"統計量": "破產風險 (>50% 虧損)", "數值": f"{ruin_risk:.2%}"},
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # 診斷提示
        st.subheader("診斷提示")

        if prob_profit >= 0.9:
            st.success(f"賺錢機率 {prob_profit:.1%}，策略在各種市場情境下都具有穩健的獲利能力。")
        elif prob_profit >= 0.8:
            st.info(f"賺錢機率 {prob_profit:.1%}，策略整體穩健。")
        elif prob_profit >= 0.6:
            st.warning(f"賺錢機率 {prob_profit:.1%}，偏低。建議檢查參數設定。")
        else:
            st.error(f"賺錢機率 {prob_profit:.1%}，策略存在問題。")

        if var_95 > -0.1:
            st.success(f"95% VaR 為 {var_95:.1%}，下行風險控制良好。")
        elif var_95 > -0.2:
            st.info(f"95% VaR 為 {var_95:.1%}，下行風險在可接受範圍。")
        else:
            st.warning(f"95% VaR 為 {var_95:.1%}，下行風險偏高。建議加大 Kd 或 deadband。")


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
        "Rolling Window",
        "Monte Carlo"
    ])

    with tab1:
        render_tab_pareto(params, data, warmup_prices_stock, warmup_prices_bond)

    with tab2:
        render_tab_heatmap(params, data, warmup_prices_stock, warmup_prices_bond)

    with tab3:
        render_tab_rolling(params, data, warmup_prices_stock, warmup_prices_bond)

    with tab4:
        render_tab_monte_carlo(params, data, warmup_prices_stock, warmup_prices_bond)


if __name__ == "__main__":
    main()
