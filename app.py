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
CMA_ES_CACHE_PATH = "data/cache/cma_es_results.json"


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


@st.cache_data(ttl=3600, show_spinner=False)
def load_cma_es_cache() -> dict:
    """載入 CMA-ES 快取"""
    if os.path.exists(CMA_ES_CACHE_PATH):
        with open(CMA_ES_CACHE_PATH, "r", encoding="utf-8") as f:
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
    # 區塊 2：Grid Search 參數範圍
    # =========================================================================
    st.sidebar.header("⚙️ Grid Search 參數範圍")

    # Kp 設定
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        kp_min = st.number_input("Kp 最小值", value=0.1, step=0.1, format="%.1f")
    with col2:
        kp_max = st.number_input("Kp 最大值", value=1.0, step=0.1, format="%.1f")
    with col3:
        kp_points = st.number_input("Kp 點數", value=10, min_value=3, step=1)

    # Kd 設定
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        kd_min = st.number_input("Kd 最小值", value=0.1, step=0.1, format="%.1f")
    with col2:
        kd_max = st.number_input("Kd 最大值", value=1.0, step=0.1, format="%.1f")
    with col3:
        kd_points = st.number_input("Kd 點數", value=10, min_value=3, step=1)

    # Q 候選值（log scale）
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        q_min = st.number_input("Q 最小值", value=0.00001,
                                 min_value=0.000001, format="%.5f")
    with col2:
        q_max = st.number_input("Q 最大值", value=0.1,
                                 min_value=0.00001, format="%.4f")
    with col3:
        q_points = st.number_input("Q 點數", value=5, min_value=2, step=1)
    q_values = np.logspace(
        np.log10(q_min), np.log10(q_max), int(q_points)
    ).tolist()

    # Deadband 設定
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        db_min = st.number_input("Deadband 最小值", value=0.005, step=0.005, format="%.3f")
    with col2:
        db_max = st.number_input("Deadband 最大值", value=0.10, step=0.005, format="%.3f")
    with col3:
        db_points = st.number_input("Deadband 點數", value=15, min_value=3, step=1)

    # =========================================================================
    # 區塊 3：卡爾曼濾波器設定
    # =========================================================================
    st.sidebar.header("🔧 卡爾曼濾波器設定")

    warmup = st.sidebar.number_input(
        "暖機天數",
        min_value=10,
        max_value=100,
        value=30,
        step=1,
        help="KF 初始化所需天數，建議 30 天"
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
    # 區塊 3.5：參考點設定
    # =========================================================================
    st.sidebar.header("📐 參考點設定")
    st.sidebar.markdown(
        "超體積參考點 = Bang-Bang 最差點 × 倍數\n"
        "倍數越大 → 面積越大，倍數越小 → 比較更嚴格"
    )
    ref_multiplier = st.sidebar.number_input(
        "參考點倍數",
        min_value=0.5,
        max_value=3.0,
        value=1.1,
        step=0.1,
        format="%.1f",
        help="預設 1.1，調小（如 0.8）讓比較更嚴格，調大讓差距更明顯"
    )

    # =========================================================================
    # 區塊 3.6：CMA-ES 設定
    # =========================================================================
    st.sidebar.header("🔍 CMA-ES 設定")

    # Kp 上下限
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cma_kp_min = st.number_input("Kp 下限", value=kp_min,
                                      min_value=0.001, format="%.3f")
    with col2:
        cma_kp_max = st.number_input("Kp 上限", value=2.0,
                                      min_value=0.1, format="%.1f")

    # Kd 上下限
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cma_kd_min = st.number_input("Kd 下限", value=kd_min,
                                      min_value=0.001, format="%.3f")
    with col2:
        cma_kd_max = st.number_input("Kd 上限", value=2.0,
                                      min_value=0.1, format="%.1f")

    # Q 上下限
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cma_q_min = st.number_input("Q 下限", value=q_min,
                                     min_value=0.000001, format="%.5f")
    with col2:
        cma_q_max = st.number_input("Q 上限", value=q_max,
                                     min_value=0.00001, format="%.4f")

    # 搜索設定
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cma_maxiter = st.number_input("最多幾輪", value=100,
                                       min_value=10, step=10)
    with col2:
        cma_popsize = st.number_input("每輪幾個點", value=10,
                                       min_value=5, step=5)

    # =========================================================================
    # 區塊 4：執行計算
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

    # 執行 Grid Search 按鈕
    if st.sidebar.button("▶ 執行 Grid Search（約 2-5 分鐘）", type="primary"):
        # 1. 先載入資料
        with st.spinner("載入資料..."):
            data = load_data(
                tickers=[ticker1, ticker2],
                start_date=str(start_date),
                end_date=str(end_date)
            )
            prices_stock = data[ticker1].values
            prices_bond = data[ticker2].values
            dates = data.index.tolist()
            rets_stock = np.diff(prices_stock) / prices_stock[:-1]
            rets_bond = np.diff(prices_bond) / prices_bond[:-1]
            prices_stock = prices_stock[1:]
            prices_bond = prices_bond[1:]
            dates = dates[1:]

        # 2. 執行 Grid Search
        t0 = time.time()
        with st.spinner("正在執行 Grid Search，請稍候..."):
            from core.optimizer import run_grid_search, find_best_from_grid
            results = run_grid_search(
                rets_stock, rets_bond, prices_stock, prices_bond, dates,
                target_w=target_w,
                fee_rate=fee_rate,
                kp_range=np.linspace(kp_min, kp_max, int(kp_points)).tolist(),
                kd_range=np.linspace(kd_min, kd_max, int(kd_points)).tolist(),
                q_values=q_values,
                deadband_values=np.linspace(db_min, db_max, int(db_points)).tolist(),
                n_jobs=n_jobs,
                kf_r=kf_r,
                warmup=warmup,
                ref_multiplier=ref_multiplier,
            )
            best = find_best_from_grid(results)
        elapsed = time.time() - t0

        # 3. 儲存 JSON
        cache_path = Path("data/cache/grid_search_results.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "best": best,
                "metadata": {
                    "q_values": q_values,
                    "kp_range": np.linspace(kp_min, kp_max, int(kp_points)).tolist(),
                    "kd_range": np.linspace(kd_min, kd_max, int(kd_points)).tolist(),
                    "deadband_values": np.linspace(db_min, db_max, int(db_points)).tolist(),
                }
            }, f, ensure_ascii=False, indent=2)

        # 4. 清除舊快取讓 load_grid_cache() 重新讀取
        load_grid_cache.clear()

        # 5. 顯示結果
        st.sidebar.success(
            f"Grid Search 完成！共 {len(results)} 組，耗時 {elapsed:.1f} 秒\n"
            f"最佳參數：Kp={best['kp']:.2f}, Kd={best['kd']:.2f}, "
            f"Q={best['q']:.5f}, HV={best['hypervolume'] * 10000:.4f} %%"
        )

    # 執行 CMA-ES 全域最佳化按鈕（雙起點）
    if st.sidebar.button("▶ 執行 CMA-ES 全域最佳化（約 5-10 分鐘）", type="primary"):
        # 1. 載入資料
        with st.spinner("載入資料..."):
            data = load_data(
                tickers=[ticker1, ticker2],
                start_date=str(start_date),
                end_date=str(end_date)
            )
            prices_stock = data[ticker1].values
            prices_bond = data[ticker2].values
            dates = data.index.tolist()
            rets_stock = np.diff(prices_stock) / prices_stock[:-1]
            rets_bond = np.diff(prices_bond) / prices_bond[:-1]
            prices_stock = prices_stock[1:]
            prices_bond = prices_bond[1:]
            dates = dates[1:]

        # 2. 執行 CMA-ES 雙起點
        t0 = time.time()
        from core.optimizer import run_cma_es

        with st.spinner("CMA-ES 第一輪搜索（從左下角出發）..."):
            result1 = run_cma_es(
                rets_stock, rets_bond, prices_stock, prices_bond, dates,
                target_w=target_w,
                fee_rate=fee_rate,
                deadband_values=np.linspace(db_min, db_max, int(db_points)).tolist(),
                kf_r=kf_r,
                warmup=warmup,
                ref_multiplier=ref_multiplier,
                kp_min=cma_kp_min,
                kp_max=cma_kp_max,
                kd_min=cma_kd_min,
                kd_max=cma_kd_max,
                q_min=cma_q_min,
                q_max=cma_q_max,
                x0=[cma_kp_min, cma_kd_min, np.log(cma_q_min)],
                maxiter=int(cma_maxiter),
                popsize=int(cma_popsize),
            )

        with st.spinner("CMA-ES 第二輪搜索（從右上角出發）..."):
            result2 = run_cma_es(
                rets_stock, rets_bond, prices_stock, prices_bond, dates,
                target_w=target_w,
                fee_rate=fee_rate,
                deadband_values=np.linspace(db_min, db_max, int(db_points)).tolist(),
                kf_r=kf_r,
                warmup=warmup,
                ref_multiplier=ref_multiplier,
                kp_min=cma_kp_min,
                kp_max=cma_kp_max,
                kd_min=cma_kd_min,
                kd_max=cma_kd_max,
                q_min=cma_q_min,
                q_max=cma_q_max,
                x0=[cma_kp_max, cma_kd_max, np.log(cma_q_max)],
                maxiter=int(cma_maxiter),
                popsize=int(cma_popsize),
            )

        elapsed = time.time() - t0

        # 取超體積較大的結果
        best_result = result1 if result1["hypervolume"] > result2["hypervolume"] else result2

        # 3. 儲存 JSON
        cache_path = Path("data/cache/cma_es_results.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "best": {
                    "kp": best_result["kp"],
                    "kd": best_result["kd"],
                    "q": best_result["q"],
                    "hypervolume": best_result["hypervolume"],
                    "hypervolume_pct": best_result["hypervolume_pct"],
                    "iterations": best_result["iterations"],
                    "evaluations": best_result["evaluations"],
                },
                "result1": {
                    "kp": result1["kp"], "kd": result1["kd"], "q": result1["q"],
                    "hypervolume": result1["hypervolume"],
                    "hypervolume_pct": result1["hypervolume_pct"],
                },
                "result2": {
                    "kp": result2["kp"], "kd": result2["kd"], "q": result2["q"],
                    "hypervolume": result2["hypervolume"],
                    "hypervolume_pct": result2["hypervolume_pct"],
                },
            }, f, ensure_ascii=False, indent=2)

        # 4. 清除快取
        load_cma_es_cache.clear()
        st.sidebar.success(
            f"CMA-ES 完成！耗時 {elapsed:.1f} 秒\n"
            f"第一輪：HV={result1['hypervolume_pct']:.4f} %%\n"
            f"第二輪：HV={result2['hypervolume_pct']:.4f} %%\n"
            f"最佳：Kp={best_result['kp']:.3f}, "
            f"Kd={best_result['kd']:.3f}, "
            f"Q={best_result['q']:.6f}, "
            f"HV={best_result['hypervolume_pct']:.4f} %%"
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

        cma_path = Path("data/cache/cma_es_results.json")
        if cma_path.exists():
            size_kb = cma_path.stat().st_size // 1024
            st.write(f"✅ cma_es_results.json（{size_kb} KB）")
        else:
            st.write("❌ cma_es_results.json（尚未計算）")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("刪除 Grid Search"):
                if grid_path.exists():
                    grid_path.unlink()
                    load_grid_cache.clear()
                    st.warning("已刪除，需重新執行 Grid Search")
        with col2:
            if st.button("刪除 CMA-ES"):
                if cma_path.exists():
                    cma_path.unlink()
                    load_cma_es_cache.clear()
                    st.warning("已刪除，需重新執行 CMA-ES")

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
        "kp": kp_min,
        "kd": kd_min,
        "kf_q": q_values[0] if q_values else 0.001,
        "kf_r": kf_r,
        "deadband": db_min,
        "warmup": warmup,
        "kp_range": np.linspace(kp_min, kp_max, int(kp_points)).tolist(),
        "kd_range": np.linspace(kd_min, kd_max, int(kd_points)).tolist(),
        "q_values": q_values,
        "deadband_values": np.linspace(db_min, db_max, int(db_points)).tolist(),
        "n_jobs": n_jobs,
        "d_clip": d_clip,
        "output_clip": output_clip,
        "ref_multiplier": ref_multiplier,
        "cma_kp_min": cma_kp_min,
        "cma_kp_max": cma_kp_max,
        "cma_kd_min": cma_kd_min,
        "cma_kd_max": cma_kd_max,
        "cma_q_min": cma_q_min,
        "cma_q_max": cma_q_max,
        "cma_maxiter": cma_maxiter,
        "cma_popsize": cma_popsize,
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
        )

    # 計算超體積（使用 ref_multiplier）
    all_pts = pareto["smart_pilot"] + pareto["bangbang"] + [pareto["yearly"]]
    ref_rmse = max(p["rmse"] for p in all_pts) * params["ref_multiplier"]
    ref_cost = max(p["ann_cost"] for p in all_pts) * params["ref_multiplier"]
    reference_point = {"rmse": ref_rmse, "cost": ref_cost}

    hv_sp = calc_hypervolume(pareto["smart_pilot"], reference_point)
    hv_bb = calc_hypervolume(pareto["bangbang"], reference_point)
    hv_yr = calc_hypervolume([pareto["yearly"]], reference_point)

    # 決定贏家
    hv_values = {"smart_pilot": hv_sp, "bangbang": hv_bb, "yearly": hv_yr}
    winner = max(hv_values, key=hv_values.get)
    hv_comparison = {
        "smart_pilot": hv_sp,
        "bangbang": hv_bb,
        "yearly": hv_yr,
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
        st.metric("Smart Pilot", f"{hv_comparison['smart_pilot'] * 10000:.4f} %%")
    with col2:
        st.metric("Bang-Bang", f"{hv_comparison['bangbang'] * 10000:.4f} %%")
    with col3:
        st.metric("Yearly", f"{hv_comparison['yearly'] * 10000:.4f} %%")

    winner = hv_comparison["winner"]
    if winner == "smart_pilot":
        st.success(f"Winner: Smart Pilot (超體積最大，Pareto Frontier 最優)")
    elif winner == "bangbang":
        st.info(f"Winner: Bang-Bang")
    else:
        st.info(f"Winner: Yearly")

    st.caption(
        f"參考點倍數：{params['ref_multiplier']}x | "
        f"參考點：RMSE={ref_rmse:.5f}，Cost={ref_cost:.6f}"
    )

    # 當前參數回測結果
    st.markdown("---")
    st.subheader("當前參數回測結果")

    with st.spinner("執行回測..."):
        sp_result = run_smart_pilot(
            rets_stock, rets_bond, prices_stock, prices_bond, dates,
            target_w=params["target_w"],
            fee_rate=params["fee_rate"],
            kf_q=params["kf_q"],
            kf_r=params["kf_r"],
            kp=params["kp"],
            kd=params["kd"],
            deadband=params["deadband"],
            warmup=params["warmup"],
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

    col1, col2 = st.columns(2)
    with col1:
        st.metric("年化週轉率", f"{sp_result['metrics']['ann_turnover'] * 100:.2f}%")
    with col2:
        st.metric("年化資產波動", f"{sp_result['metrics']['ann_wealth_vol'] * 100:.2f}%")

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
def render_tab_heatmap(params: dict, data: pd.DataFrame):
    """渲染參數空間熱力圖分頁"""
    st.header("Heatmap - 參數空間")

    grid_cache = load_grid_cache()

    if grid_cache is None:
        st.warning("找不到 Grid Search 快取檔案。")
        st.info("請在側邊欄執行 Grid Search 產生快取。")
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

    # CMA-ES 結果
    cma_cache = load_cma_es_cache()
    if cma_cache is not None:
        st.markdown("---")
        st.subheader("CMA-ES 全域最佳化結果")

        # 支援新格式（best/result1/result2）和舊格式（直接存 kp/kd/q）
        if "best" in cma_cache:
            cma_best = cma_cache["best"]
            r1 = cma_cache.get("result1", {})
            r2 = cma_cache.get("result2", {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Kp", f"{cma_best['kp']:.3f}")
            with col2:
                st.metric("Kd", f"{cma_best['kd']:.3f}")
            with col3:
                st.metric("Q", f"{cma_best['q']:.6f}")
            with col4:
                st.metric("Hypervolume", f"{cma_best['hypervolume'] * 10000:.4f} %%")

            # 顯示雙起點結果比較
            if r1 and r2:
                st.caption(
                    f"第一輪（左下角出發）：HV={r1['hypervolume_pct']:.4f} %% | "
                    f"第二輪（右上角出發）：HV={r2['hypervolume_pct']:.4f} %%"
                )
        else:
            # 舊格式相容
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Kp", f"{cma_cache['kp']:.3f}")
            with col2:
                st.metric("Kd", f"{cma_cache['kd']:.3f}")
            with col3:
                st.metric("Q", f"{cma_cache['q']:.6f}")
            with col4:
                st.metric("Hypervolume", f"{cma_cache['hypervolume'] * 10000:.4f} %%")
    else:
        st.info("尚未計算 CMA-ES，請在側邊欄執行 CMA-ES 全域最佳化。")

    # =========================================================================
    # 單點快速測試
    # =========================================================================
    st.markdown("---")
    with st.expander("🧪 單點快速測試", expanded=False):
        st.markdown("輸入一組參數，快速計算這組參數的超體積和回測指標。")
        st.markdown(
            f"目前參考點倍數：**{params['ref_multiplier']}x** "
            "（可在左側側邊欄「參考點設定」調整）"
        )

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
                )

                # 計算參考點（用 Bang-Bang 最差點 × ref_multiplier）
                all_pts = pareto["smart_pilot"] + pareto["bangbang"]
                ref_rmse = max(p["rmse"] for p in all_pts) * params["ref_multiplier"]
                ref_cost = max(p["ann_cost"] for p in all_pts) * params["ref_multiplier"]
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
                f"（Bang-Bang 最差點 × {params['ref_multiplier']}）"
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

    prices_stock = data[params["ticker1"]].values
    prices_bond = data[params["ticker2"]].values
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
                kf_r=params["kf_r"],
                kp=params["kp"],
                kd=params["kd"],
                deadband=params["deadband"],
                warmup=params["warmup"],
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
            tickers=[params["ticker1"], params["ticker2"]],
            start_date=str(params["start_date"]),
            end_date=str(params["end_date"])
        )
        data = data[[params["ticker1"], params["ticker2"]]]
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
        render_tab_heatmap(params, data)

    with tab3:
        render_tab_rolling(params, data)

    with tab4:
        render_tab_monte_carlo(params, data)


if __name__ == "__main__":
    main()
