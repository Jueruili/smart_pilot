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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.data_loader import DataLoader
from core.benchmark import (
    run_smart_pilot, run_bangbang, run_yearly,
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
SLSQP_CACHE_PATH = "data/cache/slsqp_results.json"

# 預設參數
DEFAULT_STOCK_TICKER = "VTI"
DEFAULT_BOND_TICKER = "BND"
DEFAULT_START_DATE = "2013-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_TARGET_W = 0.6
DEFAULT_FEE_RATE = 0.003

# 風險偏好對應的參數（v3.0 PD + KF）
RISK_PROFILES = {
    "保守": {"kf_q": 0.0001, "kp": 0.3, "kd": 0.1, "deadband": 0.05},
    "穩健": {"kf_q": 0.001, "kp": 0.5, "kd": 0.3, "deadband": 0.025},
    "積極": {"kf_q": 0.01, "kp": 0.8, "kd": 0.5, "deadband": 0.01},
}


# =============================================================================
# 快取函數
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """載入並處理資料"""
    loader = DataLoader()
    data = loader.load_and_process(tickers, start_date, end_date)
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_grid_cache() -> dict:
    """載入 Grid Search 快取"""
    if os.path.exists(GRID_CACHE_PATH):
        with open(GRID_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_slsqp_cache() -> dict:
    """載入 SLSQP 快取"""
    if os.path.exists(SLSQP_CACHE_PATH):
        with open(SLSQP_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =============================================================================
# 側邊欄
# =============================================================================
def render_sidebar() -> dict:
    """渲染側邊欄並返回參數"""
    st.sidebar.header("參數設定")

    # 資產配置
    st.sidebar.subheader("資產配置")
    target_ratio = st.sidebar.slider(
        "目標股票比例",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        format="%d%%",
        help="股票在投資組合中的目標比例"
    )

    # 風險偏好
    st.sidebar.subheader("風險偏好")
    risk_profile = st.sidebar.selectbox(
        "選擇風險偏好",
        options=list(RISK_PROFILES.keys()),
        index=1,
        help="風險偏好決定 KF + PD 參數"
    )
    default_params = RISK_PROFILES[risk_profile]

    # KF + PD 參數（可手動調整）
    st.sidebar.subheader("KF + PD 參數")

    kf_q = st.sidebar.select_slider(
        "KF Q (過程雜訊)",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        value=default_params["kf_q"],
        help="卡爾曼濾波器的過程雜訊協方差"
    )

    kp = st.sidebar.number_input(
        "Kp (比例增益)",
        min_value=0.0,
        max_value=2.0,
        value=default_params["kp"],
        step=0.1,
        format="%.2f",
        help="PD 控制器的比例增益"
    )

    kd = st.sidebar.number_input(
        "Kd (微分增益)",
        min_value=0.0,
        max_value=2.0,
        value=default_params["kd"],
        step=0.1,
        format="%.2f",
        help="PD 控制器的微分增益（使用 KF 速度）"
    )

    deadband = st.sidebar.number_input(
        "Deadband (死區閾值)",
        min_value=0.001,
        max_value=0.10,
        value=default_params["deadband"],
        step=0.005,
        format="%.3f",
        help="控制量低於此閾值時不執行交易"
    )

    # 進階設定
    st.sidebar.subheader("進階設定")
    fee_rate = st.sidebar.slider(
        "手續費率",
        min_value=0.0,
        max_value=0.01,
        value=DEFAULT_FEE_RATE,
        step=0.0005,
        format="%.4f",
        help="每筆交易的手續費率"
    )

    # 系統資訊
    st.sidebar.markdown("---")
    with st.sidebar.expander("系統資訊"):
        st.write("**版本:** 3.0.0")
        st.write("**架構:** PD + Log-KF")
        st.write("**資料來源:** Yahoo Finance")
        st.write(f"**股票標的:** {DEFAULT_STOCK_TICKER}")
        st.write(f"**債券標的:** {DEFAULT_BOND_TICKER}")
        st.write(f"**資料期間:** {DEFAULT_START_DATE} ~ {DEFAULT_END_DATE}")

    return {
        "target_w": target_ratio / 100,
        "risk_profile": risk_profile,
        "kf_q": kf_q,
        "kp": kp,
        "kd": kd,
        "deadband": deadband,
        "fee_rate": fee_rate,
        "stock_ticker": DEFAULT_STOCK_TICKER,
        "bond_ticker": DEFAULT_BOND_TICKER,
    }


# =============================================================================
# Tab 1: Pareto Frontier
# =============================================================================
def render_tab_pareto(params: dict, data: pd.DataFrame):
    """渲染 Pareto Frontier 分頁"""
    st.header("Pareto Frontier - 三策略對比")

    st.markdown("""
    比較三種再平衡策略在 **RMSE（追蹤誤差）** vs **Cost（年化交易成本）** 空間的表現。
    越靠左下角的策略越好（低誤差、低成本）。
    """)

    # 準備資料
    prices_stock = data[params["stock_ticker"]].values
    prices_bond = data[params["bond_ticker"]].values
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
            kp=params["kp"],
            kd=params["kd"],
            n_points=30,
        )

    # 計算超體積
    hv_comparison = compare_hypervolumes(pareto)

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

    # Bang-Bang
    bb_data = pareto["bangbang"]
    fig.add_trace(go.Scatter(
        x=[p["rmse"] * 100 for p in bb_data],
        y=[p["ann_cost"] * 100 for p in bb_data],
        mode="lines+markers",
        name="Bang-Bang",
        line=dict(color="gray", width=1.5, dash="dash"),
        marker=dict(size=5),
        hovertemplate="RMSE: %{x:.2f}%<br>Cost: %{y:.3f}%/年<extra></extra>"
    ))

    # Yearly
    yr_data = pareto["yearly"]
    fig.add_trace(go.Scatter(
        x=[yr_data["rmse"] * 100],
        y=[yr_data["ann_cost"] * 100],
        mode="markers",
        name="Yearly",
        marker=dict(size=12, color="cyan", symbol="star"),
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
        st.metric("Smart Pilot", f"{hv_comparison['smart_pilot']:.6f}")
    with col2:
        st.metric("Bang-Bang", f"{hv_comparison['bangbang']:.6f}")
    with col3:
        st.metric("Yearly", f"{hv_comparison['yearly']:.6f}")

    winner = hv_comparison["winner"]
    if winner == "smart_pilot":
        st.success(f"Winner: Smart Pilot (超體積最大，Pareto Frontier 最優)")
    elif winner == "bangbang":
        st.info(f"Winner: Bang-Bang")
    else:
        st.info(f"Winner: Yearly")

    # 當前參數回測結果
    st.markdown("---")
    st.subheader("當前參數回測結果")

    with st.spinner("執行回測..."):
        sp_result = run_smart_pilot(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=params["target_w"],
            fee_rate=params["fee_rate"],
            kf_q=params["kf_q"],
            kp=params["kp"],
            kd=params["kd"],
            deadband=params["deadband"],
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("年化報酬", f"{sp_result['metrics']['ann_return_pct']:.2f}%")
    with col2:
        st.metric("Sharpe", f"{sp_result['metrics']['sharpe']:.2f}")
    with col3:
        st.metric("RMSE", f"{sp_result['rmse'] * 100:.2f}%")
    with col4:
        st.metric("交易次數", f"{sp_result['trade_count']}")

    # 淨值曲線
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(
        y=sp_result["nav_list"],
        mode="lines",
        name="Smart Pilot NAV",
        line=dict(color="#FFD700", width=2)
    ))
    fig_nav.update_layout(
        xaxis_title="交易日",
        yaxis_title="淨值",
        hovermode="x unified"
    )
    st.plotly_chart(fig_nav, use_container_width=True)


# =============================================================================
# Tab 2: Heatmap
# =============================================================================
def render_tab_heatmap(params: dict):
    """渲染參數空間熱力圖分頁"""
    st.header("Heatmap - 參數空間")

    grid_cache = load_grid_cache()

    if grid_cache is None:
        st.warning("找不到 Grid Search 快取檔案。")
        st.info("請執行 `python scripts/run_grid_search.py` 產生快取。")
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
        index=1,
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

    # SLSQP 結果
    slsqp_cache = load_slsqp_cache()
    if slsqp_cache is not None:
        st.markdown("---")
        st.subheader("SLSQP 精確最佳化結果")

        slsqp_results = slsqp_cache["results"]

        # 表格
        df = pd.DataFrame(slsqp_results)
        df["deadband"] = df["deadband"].apply(lambda x: f"{x:.4f}")
        df["kp"] = df["kp"].apply(lambda x: f"{x:.3f}")
        df["kd"] = df["kd"].apply(lambda x: f"{x:.3f}")
        df["q"] = df["q"].apply(lambda x: f"{x:.6f}")
        df["rmse"] = df["rmse"].apply(lambda x: f"{x * 100:.3f}%")
        df["ann_cost"] = df["ann_cost"].apply(lambda x: f"{x * 100:.4f}%")

        st.dataframe(
            df[["deadband", "kp", "kd", "q", "rmse", "ann_cost"]],
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# Tab 3: Rolling Window
# =============================================================================
def render_tab_rolling(params: dict, data: pd.DataFrame):
    """渲染滾動窗口分析分頁"""
    st.header("Rolling Window Analysis")

    st.markdown("""
    使用滾動窗口分析策略的穩定性。每個窗口獨立計算 RMSE 和 Sharpe，
    觀察策略在不同時間段的表現是否一致。
    """)

    # 窗口設定
    col1, col2 = st.columns(2)
    with col1:
        window_years = st.slider(
            "窗口大小（年）",
            min_value=1,
            max_value=5,
            value=2,
            step=1
        )
    with col2:
        step_months = st.slider(
            "步進（月）",
            min_value=1,
            max_value=12,
            value=6,
            step=1
        )

    window_days = window_years * 252
    step_days = step_months * 21

    prices_stock = data[params["stock_ticker"]].values
    prices_bond = data[params["bond_ticker"]].values
    dates = data.index.tolist()

    # 計算日報酬率
    rets_stock = np.zeros(len(prices_stock))
    rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
    rets_bond = np.zeros(len(prices_bond))
    rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

    # 滾動窗口分析
    rolling_results = []

    with st.spinner("執行滾動窗口分析..."):
        start_idx = 0
        while start_idx + window_days <= len(dates):
            end_idx = start_idx + window_days

            # 切片資料
            w_rets_stock = rets_stock[start_idx:end_idx]
            w_rets_bond = rets_bond[start_idx:end_idx]
            w_prices_stock = prices_stock[start_idx:end_idx]
            w_prices_bond = prices_bond[start_idx:end_idx]
            w_dates = dates[start_idx:end_idx]

            # 執行回測
            result = run_smart_pilot(
                w_rets_stock, w_rets_bond, w_prices_stock, w_prices_bond, w_dates,
                target_w=params["target_w"],
                fee_rate=params["fee_rate"],
                kf_q=params["kf_q"],
                kp=params["kp"],
                kd=params["kd"],
                deadband=params["deadband"],
            )

            rolling_results.append({
                "start_date": w_dates[0].strftime("%Y-%m-%d"),
                "end_date": w_dates[-1].strftime("%Y-%m-%d"),
                "rmse": result["rmse"],
                "sharpe": result["metrics"]["sharpe"],
                "ann_return": result["metrics"]["ann_return"],
                "trade_count": result["trade_count"],
            })

            start_idx += step_days

    if not rolling_results:
        st.warning("資料不足以進行滾動窗口分析")
        return

    # 繪製結果
    df_rolling = pd.DataFrame(rolling_results)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("RMSE", "Sharpe Ratio"))

    fig.add_trace(go.Scatter(
        x=df_rolling["start_date"],
        y=df_rolling["rmse"] * 100,
        mode="lines+markers",
        name="RMSE (%)",
        line=dict(color="#1f77b4")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_rolling["start_date"],
        y=df_rolling["sharpe"],
        mode="lines+markers",
        name="Sharpe",
        line=dict(color="#ff7f0e")
    ), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 統計摘要
    st.subheader("統計摘要")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均 RMSE", f"{df_rolling['rmse'].mean() * 100:.2f}%")
    with col2:
        st.metric("RMSE 標準差", f"{df_rolling['rmse'].std() * 100:.2f}%")
    with col3:
        st.metric("平均 Sharpe", f"{df_rolling['sharpe'].mean():.2f}")
    with col4:
        st.metric("Sharpe 標準差", f"{df_rolling['sharpe'].std():.2f}")

    # 詳細結果表格
    with st.expander("詳細結果"):
        df_display = df_rolling.copy()
        df_display["rmse"] = df_display["rmse"].apply(lambda x: f"{x * 100:.2f}%")
        df_display["sharpe"] = df_display["sharpe"].apply(lambda x: f"{x:.2f}")
        df_display["ann_return"] = df_display["ann_return"].apply(lambda x: f"{x * 100:.2f}%")
        st.dataframe(df_display, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 4: Monte Carlo
# =============================================================================
def render_tab_monte_carlo(params: dict, data: pd.DataFrame):
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
    prices_stock = data[params["stock_ticker"]].values
    prices_bond = data[params["bond_ticker"]].values
    dates = data.index.tolist()
    n_days = len(dates)

    rets_stock = np.zeros(n_days)
    rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
    rets_bond = np.zeros(n_days)
    rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

    if st.button("執行蒙地卡羅模擬", type="primary"):
        with st.spinner(f"執行 {n_simulations} 次模擬..."):
            np.random.seed(random_seed)

            # Bootstrap 模擬
            final_returns = []
            warmup = 30

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
                result = run_smart_pilot(
                    np.concatenate([[0], sim_rets_stock]),
                    np.concatenate([[0], sim_rets_bond]),
                    np.concatenate([[prices_stock[0]], sim_prices_stock]),
                    np.concatenate([[prices_bond[0]], sim_prices_bond]),
                    sim_dates,
                    target_w=params["target_w"],
                    fee_rate=params["fee_rate"],
                    kf_q=params["kf_q"],
                    kp=params["kp"],
                    kd=params["kd"],
                    deadband=params["deadband"],
                    warmup=1,
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

    # 載入資料
    try:
        data = load_data(
            tickers=[params["stock_ticker"], params["bond_ticker"]],
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE
        )
        data = data[[params["stock_ticker"], params["bond_ticker"]]]
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
        render_tab_pareto(params, data)

    with tab2:
        render_tab_heatmap(params)

    with tab3:
        render_tab_rolling(params, data)

    with tab4:
        render_tab_monte_carlo(params, data)


if __name__ == "__main__":
    main()
