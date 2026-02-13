"""
Smart Pilot - 投資組合再平衡系統

Streamlit 單頁應用程式（升級版）

功能：
- 參數設定（初始資金、目標比例、PID 參數可編輯）
- 執行回測與績效分析
- 策略對比（Smart Pilot vs Threshold vs Yearly）
- 蒙地卡羅模擬（含診斷提示）
- 樣本外測試（含過擬合檢測）

作者：Smart Pilot Team
版本：2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback

from data.data_loader import DataLoader
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator
from core.benchmark import run_threshold_rebalance, run_yearly_rebalance, calculate_tracking_error
from validation.monte_carlo import MonteCarloSimulator
from validation.out_of_sample import OutOfSampleValidator


# =============================================================================
# 頁面設定
# =============================================================================
st.set_page_config(
    page_title="Smart Pilot - 投資組合再平衡系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 風險偏好對應的 PID 參數（新版）
# =============================================================================
RISK_PROFILES = {
    "保守": {"kp": 0.15, "ki": 0.02, "kd": 0.05, "deadband": 0.03},
    "穩健": {"kp": 0.3, "ki": 0.05, "kd": 0.1, "deadband": 0.02},
    "積極": {"kp": 0.5, "ki": 0.1, "kd": 0.2, "deadband": 0.01},
}

# 預設標的
DEFAULT_STOCK_TICKER = "SPY"
DEFAULT_BOND_TICKER = "TLT"


# =============================================================================
# 快取函數
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """載入並處理資料（使用快取避免重複下載）"""
    loader = DataLoader()
    data = loader.load_and_process(tickers, start_date, end_date)
    return data


# =============================================================================
# 側邊欄
# =============================================================================
def render_sidebar() -> dict:
    """渲染側邊欄並返回使用者輸入的參數"""
    st.sidebar.header(" 參數設定")

    # 初始資金
    st.sidebar.subheader(" 初始資金")
    initial_cash = st.sidebar.number_input(
        "初始資金 (USD)",
        min_value=10_000,
        max_value=100_000_000,
        value=1_000_000,
        step=100_000,
        format="%d",
        help="投入的初始資金金額"
    )

    # 目標股票比例
    st.sidebar.subheader(" 資產配置")
    target_ratio = st.sidebar.slider(
        "目標股票比例",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        format="%d%%",
        help="股票在投資組合中的目標比例，其餘為債券"
    )

    # 標的設定
    st.sidebar.subheader(" 投資標的")
    stock_ticker = st.sidebar.text_input(
        "股票標的", value="VTI",
        help="股票型 ETF 代碼（如 SPY, QQQ, VTI）"
    ).upper().strip()
    bond_ticker = st.sidebar.text_input(
        "債券標的", value="BND",
        help="債券型 ETF 代碼（如 TLT, BND, AGG）"
    ).upper().strip()
    
    # 風險偏好
    st.sidebar.subheader(" 風險偏好")
    risk_profile = st.sidebar.selectbox(
        "選擇風險偏好",
        options=list(RISK_PROFILES.keys()),
        index=1,  # 預設選「穩健」
        help="風險偏好決定 PID 控制器的初始參數"
    )

    # 取得預設參數
    default_params = RISK_PROFILES[risk_profile]

    # PID 參數（可手動調整）
    st.sidebar.subheader(" PID 參數（可手動調整）")

    kp = st.sidebar.number_input(
        "Kp (比例增益)",
        min_value=0.0,
        max_value=2.0,
        value=default_params["kp"],
        step=0.05,
        format="%.2f",
        help="控制對誤差的即時反應強度"
    )

    ki = st.sidebar.number_input(
        "Ki (積分增益)",
        min_value=0.0,
        max_value=0.5,
        value=default_params["ki"],
        step=0.01,
        format="%.2f",
        help="控制累積誤差的修正力度"
    )

    kd = st.sidebar.number_input(
        "Kd (微分增益)",
        min_value=0.0,
        max_value=1.0,
        value=default_params["kd"],
        step=0.05,
        format="%.2f",
        help="控制對誤差變化的預測反應"
    )

    # 進階設定
    st.sidebar.subheader(" 進階設定")

    deadband = st.sidebar.number_input(
        "死區閾值",
        min_value=0.0,
        max_value=0.10,
        value=default_params["deadband"],
        step=0.005,
        format="%.3f",
        help="PID 調整量低於此閾值時不執行交易"
    )

    commission_rate = st.sidebar.slider(
        "手續費率",
        min_value=0.0,
        max_value=0.01,
        value=0.001,
        step=0.0005,
        format="%.4f",
        help="每筆交易的手續費率"
    )

    # 分隔線
    st.sidebar.markdown("---")

    # 標的設定
    stock_ticker = DEFAULT_STOCK_TICKER
    bond_ticker = DEFAULT_BOND_TICKER

    # 系統資訊
    with st.sidebar.expander("ℹ️ 系統資訊"):
        st.write("**版本:** 2.0.0")
        st.write("**資料來源:** Yahoo Finance")
        st.write(f"**股票標的:** {stock_ticker}")
        st.write(f"**債券標的:** {bond_ticker}")
        st.write("**資料期間:** 2013-01-01 ~ 2025-12-31")

    return {
        "initial_cash": initial_cash,
        "target_ratio": target_ratio / 100,
        "risk_profile": risk_profile,
        "pid_params": {"kp": kp, "ki": ki, "kd": kd},
        "deadband": deadband,
        "commission_rate": commission_rate,
        "stock_ticker": stock_ticker,
        "bond_ticker": bond_ticker,
    }


# =============================================================================
# 主畫面
# =============================================================================
def render_main_content(params: dict):
    """渲染主畫面內容"""
    # 標題
    st.title("📊 Smart Pilot 投資組合再平衡系統")

    # 說明文字
    st.markdown("""
    **Smart Pilot** 是基於 **增量型 PID 控制理論** 的投資組合再平衡系統。

    系統使用 PID 控制器智慧計算再平衡調整量，在維持目標資產配置的同時，
    避免過度交易並降低交易成本。

    ---

    **運作原理：**
    1. 每日計算實際股票比例與目標比例的誤差
    2. PID 控制器根據誤差計算調整量
    3. 若調整量超過死區閾值，則執行交易
    4. 記錄交易並累計手續費

    **使用方式：** 從左側設定參數後，點擊下方「開始回測」按鈕。

    ---
    """)

    # 顯示當前設定摘要
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("初始資金", f"${params['initial_cash']:,.0f}")
    with col2:
        st.metric("目標股票比例", f"{params['target_ratio']:.0%}")
    with col3:
        st.metric("風險偏好", params['risk_profile'])
    with col4:
        st.metric("死區閾值", f"{params['deadband']:.1%}")

    # PID 參數顯示
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Kp", f"{params['pid_params']['kp']:.2f}")
    with col6:
        st.metric("Ki", f"{params['pid_params']['ki']:.2f}")
    with col7:
        st.metric("Kd", f"{params['pid_params']['kd']:.2f}")

    st.markdown("---")

    # 開始回測按鈕
    if st.button(" 開始回測", type="primary", use_container_width=True):
        run_backtest(params)


def run_backtest(params: dict):
    """執行回測並顯示結果"""
    try:
        with st.spinner("回測進行中..."):
            # 載入資料
            st.text("📥 載入資料中...")
            tickers = [params["stock_ticker"], params["bond_ticker"]]
            data = load_data(
                tickers=tickers,
                start_date="2013-01-01",
                end_date="2025-12-31"
            )
            # 強制排序欄位
            data = data[[params["stock_ticker"], params["bond_ticker"]]]

            # 執行 Smart Pilot 回測
            st.text("⚙️ 執行 Smart Pilot 回測中...")
            engine = BacktestEngine(
                initial_cash=params["initial_cash"],
                target_ratio=params["target_ratio"],
                pid_params=params["pid_params"],
                deadband=params["deadband"],
                commission_rate=params["commission_rate"]
            )
            result = engine.run(data)

            # 計算績效指標
            st.text("📊 計算績效指標中...")
            calculator = MetricsCalculator(risk_free_rate=0.02)
            metrics = calculator.calculate_all_metrics(result.history)

            # 執行對照策略
            st.text("⚔️ 執行對照策略中...")
            threshold_history = run_threshold_rebalance(
                data=data,
                initial_cash=params["initial_cash"],
                target_ratio=params["target_ratio"],
                threshold=0.05,
                commission_rate=params["commission_rate"]
            )
            yearly_history = run_yearly_rebalance(
                data=data,
                initial_cash=params["initial_cash"],
                target_ratio=params["target_ratio"],
                commission_rate=params["commission_rate"]
            )

            # 計算對照策略績效
            threshold_metrics = calculator.calculate_all_metrics(threshold_history)
            yearly_metrics = calculator.calculate_all_metrics(yearly_history)

        # 顯示成功訊息
        st.success("✅ 回測完成！")

        # 顯示結果（使用分頁）
        display_results_with_tabs(
            result, metrics, params, data, engine,
            threshold_history, threshold_metrics,
            yearly_history, yearly_metrics,
            calculator
        )

    except Exception as e:
        st.error(f"❌ 回測失敗：{str(e)}")
        with st.expander("錯誤詳情"):
            st.code(traceback.format_exc())


def display_results_with_tabs(
    result, metrics, params, data, engine,
    threshold_history, threshold_metrics,
    yearly_history, yearly_metrics,
    calculator
):
    """使用分頁顯示回測結果"""
    st.markdown("---")

    # 建立四個分頁
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 回測結果",
        "⚔️ 策略對比",
        "🎲 蒙地卡羅模擬",
        "🔬 樣本外測試"
    ])

    with tab1:
        render_tab_backtest_results(result, metrics, params)

    with tab2:
        render_tab_strategy_comparison(
            result, metrics, params,
            threshold_history, threshold_metrics,
            yearly_history, yearly_metrics
        )

    with tab3:
        render_tab_monte_carlo(result, params)

    with tab4:
        render_tab_out_of_sample(engine, data, params, calculator)


# =============================================================================
# Tab 1: 回測結果
# =============================================================================
def render_tab_backtest_results(result, metrics: dict, params: dict):
    """渲染回測結果分頁"""
    st.header("📈 回測結果")

    # 關鍵績效指標卡片
    st.subheader("🎯 關鍵績效指標")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_return = metrics.get("total_return", 0)
        st.metric(
            label="總報酬率",
            value=f"{total_return:.2%}",
            delta=f"年化 {metrics.get('annual_return', 0):.2%}"
        )
    with col2:
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_display = f"{sharpe:.2f}" if not np.isinf(sharpe) and not np.isnan(sharpe) else "N/A"
        st.metric(label="夏普比率", value=sharpe_display)
    with col3:
        max_dd = metrics.get("max_drawdown", 0)
        st.metric(label="最大回撤", value=f"{max_dd:.2%}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(label="總交易次數", value=f"{metrics.get('total_trades', 0):,}")
    with col5:
        st.metric(label="總手續費", value=f"${metrics.get('total_commission', 0):,.2f}")
    with col6:
        st.metric(label="勝率", value=f"{metrics.get('win_rate', 0):.1%}")

    # 資產淨值曲線
    st.markdown("---")
    st.subheader("📊 資產淨值曲線")
    history = result.history

    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(
        x=history.index,
        y=history["nav"],
        mode="lines",
        name="Smart Pilot",
        line=dict(color="#FFD700", width=2)
    ))
    fig_nav.update_layout(
        xaxis_title="日期",
        yaxis_title="淨值 (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_nav, use_container_width=True)

    # PID 控制訊號圖
    st.subheader("🎛️ PID 控制訊號")
    trade_days = history[history["trade_flag"] == True]
    if len(trade_days) > 0 and "delta_u" in history.columns:
        fig_signal = go.Figure()
        fig_signal.add_trace(go.Bar(
            x=trade_days.index,
            y=trade_days["delta_u"],
            name="調整量 (delta_u)",
            marker_color=np.where(trade_days["delta_u"] > 0, "#00CC00", "#CC0000")
        ))
        fig_signal.update_layout(
            xaxis_title="日期",
            yaxis_title="PID 調整量",
            hovermode="x unified"
        )
        st.plotly_chart(fig_signal, use_container_width=True)
        st.caption("📌 此圖展示 PID「該動才動」的特性：只有超過死區閾值時才執行交易。")
    else:
        st.info("無交易訊號可顯示")

    # 股票比例變化
    with st.expander("📉 股票比例變化"):
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=history.index, y=history["ratio"],
            mode="lines", name="實際比例", line=dict(color="#1f77b4")
        ))
        fig_ratio.add_hline(
            y=params["target_ratio"],
            line_dash="dash", line_color="red",
            annotation_text=f"目標 {params['target_ratio']:.0%}"
        )
        fig_ratio.update_layout(xaxis_title="日期", yaxis_title="股票比例")
        st.plotly_chart(fig_ratio, use_container_width=True)

    # 詳細指標表格
    st.markdown("---")
    st.subheader("📋 詳細績效指標")
    metrics_df = pd.DataFrame([
        {"類別": "報酬", "指標": "總報酬率", "數值": f"{metrics.get('total_return', 0):.2%}"},
        {"類別": "報酬", "指標": "年化報酬率", "數值": f"{metrics.get('annual_return', 0):.2%}"},
        {"類別": "風險", "指標": "最大回撤", "數值": f"{metrics.get('max_drawdown', 0):.2%}"},
        {"類別": "風險", "指標": "年化波動率", "數值": f"{metrics.get('volatility', 0):.2%}"},
        {"類別": "風險", "指標": "下行波動率", "數值": f"{metrics.get('downside_volatility', 0):.2%}"},
        {"類別": "風險調整", "指標": "夏普比率", "數值": format_metric(metrics.get('sharpe_ratio', 0))},
        {"類別": "風險調整", "指標": "索提諾比率", "數值": format_metric(metrics.get('sortino_ratio', 0))},
        {"類別": "風險調整", "指標": "卡瑪比率", "數值": format_metric(metrics.get('calmar_ratio', 0))},
        {"類別": "交易", "指標": "總交易次數", "數值": f"{metrics.get('total_trades', 0):,}"},
        {"類別": "交易", "指標": "總手續費", "數值": f"${metrics.get('total_commission', 0):,.2f}"},
        {"類別": "交易", "指標": "勝率", "數值": f"{metrics.get('win_rate', 0):.1%}"},
        {"類別": "交易", "指標": "平均獲利", "數值": f"{metrics.get('average_gain', 0):.2%}"},
        {"類別": "交易", "指標": "平均虧損", "數值": f"{metrics.get('average_loss', 0):.2%}"},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # 下載按鈕
    st.markdown("---")
    st.subheader("💾 下載資料")
    col1, col2 = st.columns(2)
    with col1:
        trade_history = history[history["trade_flag"] == True].copy()
        if len(trade_history) > 0:
            st.download_button(
                label="📥 下載交易明細 (CSV)",
                data=trade_history.to_csv(index=True),
                file_name="trade_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("無交易記錄可下載")
    with col2:
        st.download_button(
            label="📥 下載完整歷史 (CSV)",
            data=history.to_csv(index=True),
            file_name="backtest_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =============================================================================
# Tab 2: 策略對比
# =============================================================================
def render_tab_strategy_comparison(
    result, metrics, params,
    threshold_history, threshold_metrics,
    yearly_history, yearly_metrics
):
    """渲染策略對比分頁"""
    st.header("⚔️ 策略對比")

    history = result.history

    # 三策略淨值曲線對比
    st.subheader("📊 三策略淨值曲線對比")
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=history.index, y=history["nav"],
        mode="lines", name="Smart Pilot (PID)",
        line=dict(color="#FFD700", width=2)
    ))
    fig_compare.add_trace(go.Scatter(
        x=threshold_history.index, y=threshold_history["nav"],
        mode="lines", name="Threshold 5%",
        line=dict(color="gray", width=1.5, dash="dash")
    ))
    fig_compare.add_trace(go.Scatter(
        x=yearly_history.index, y=yearly_history["nav"],
        mode="lines", name="Yearly Rebalance",
        line=dict(color="cyan", width=1.5, dash="dot")
    ))
    fig_compare.update_layout(
        xaxis_title="日期",
        yaxis_title="淨值 (USD)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # 權重追蹤對比
    st.subheader("📉 三策略權重追蹤對比")
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=history.index, y=history["ratio"],
        mode="lines", name="Smart Pilot", line=dict(color="#FFD700")
    ))
    fig_ratio.add_trace(go.Scatter(
        x=threshold_history.index, y=threshold_history["ratio"],
        mode="lines", name="Threshold 5%", line=dict(color="gray", dash="dash")
    ))
    fig_ratio.add_trace(go.Scatter(
        x=yearly_history.index, y=yearly_history["ratio"],
        mode="lines", name="Yearly", line=dict(color="cyan", dash="dot")
    ))
    fig_ratio.add_hline(
        y=params["target_ratio"],
        line_dash="solid", line_color="red", line_width=2,
        annotation_text=f"目標 {params['target_ratio']:.0%}"
    )
    fig_ratio.update_layout(
        xaxis_title="日期",
        yaxis_title="股票比例",
        hovermode="x unified"
    )
    st.plotly_chart(fig_ratio, use_container_width=True)

    # 策略績效比較表
    st.subheader("📋 策略績效比較表")

    # 計算追蹤誤差
    rmse_smart = calculate_tracking_error(history["ratio"], params["target_ratio"])
    rmse_threshold = calculate_tracking_error(threshold_history["ratio"], params["target_ratio"])
    rmse_yearly = calculate_tracking_error(yearly_history["ratio"], params["target_ratio"])

    comparison_df = pd.DataFrame({
        "指標": [
            "年化報酬",
            "Sharpe",
            "最大回撤",
            "交易次數",
            "總手續費",
            "追蹤誤差(RMSE)"
        ],
        "Smart Pilot": [
            f"{metrics.get('annual_return', 0):.2%}",
            format_metric(metrics.get('sharpe_ratio', 0)),
            f"{metrics.get('max_drawdown', 0):.2%}",
            f"{metrics.get('total_trades', 0):,}",
            f"${metrics.get('total_commission', 0):,.2f}",
            f"{rmse_smart:.2%}"
        ],
        "Threshold 5%": [
            f"{threshold_metrics.get('annual_return', 0):.2%}",
            format_metric(threshold_metrics.get('sharpe_ratio', 0)),
            f"{threshold_metrics.get('max_drawdown', 0):.2%}",
            f"{threshold_metrics.get('total_trades', 0):,}",
            f"${threshold_metrics.get('total_commission', 0):,.2f}",
            f"{rmse_threshold:.2%}"
        ],
        "Yearly": [
            f"{yearly_metrics.get('annual_return', 0):.2%}",
            format_metric(yearly_metrics.get('sharpe_ratio', 0)),
            f"{yearly_metrics.get('max_drawdown', 0):.2%}",
            f"{yearly_metrics.get('total_trades', 0):,}",
            f"${yearly_metrics.get('total_commission', 0):,.2f}",
            f"{rmse_yearly:.2%}"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # 智慧診斷提示
    st.subheader("💡 智慧診斷")

    smart_sharpe = metrics.get('sharpe_ratio', 0)
    threshold_sharpe = threshold_metrics.get('sharpe_ratio', 0)
    yearly_sharpe = yearly_metrics.get('sharpe_ratio', 0)
    smart_trades = metrics.get('total_trades', 0)
    threshold_trades = threshold_metrics.get('total_trades', 0)
    yearly_trades = yearly_metrics.get('total_trades', 0)
    smart_commission = metrics.get('total_commission', 0)
    yearly_commission = yearly_metrics.get('total_commission', 0)

    # Sharpe 比較
    if not np.isinf(smart_sharpe) and not np.isnan(smart_sharpe):
        if smart_sharpe > threshold_sharpe and smart_sharpe > yearly_sharpe:
            st.success("✅ Smart Pilot 的風險調整報酬優於傳統策略")
        elif smart_sharpe > threshold_sharpe or smart_sharpe > yearly_sharpe:
            st.info("💡 Smart Pilot 的風險調整報酬優於部分傳統策略")

    # 交易次數比較
    if smart_trades < threshold_trades:
        st.success("✅ PID 控制有效減少了不必要的交易")

    # 追蹤誤差比較
    if rmse_smart > rmse_threshold:
        st.warning("⚠️ PID 追蹤精度不如門檻法，考慮增加 Ki 以提升穩態追蹤能力")

    # 手續費比較
    if yearly_commission > 0 and smart_commission > yearly_commission * 5:
        st.warning("⚠️ 交易頻率偏高，建議加大死區閾值 (deadband) 以減少交易次數")


# =============================================================================
# Tab 3: 蒙地卡羅模擬
# =============================================================================
def render_tab_monte_carlo(result, params: dict):
    """渲染蒙地卡羅模擬分頁"""
    st.header("🎲 蒙地卡羅模擬")

    st.markdown("""
    使用 Bootstrap 方法隨機重組歷史報酬率，模擬 10,000 次投資路徑，
    評估策略在不同市場情境下的表現。
    """)

    with st.spinner("執行蒙地卡羅模擬中..."):
        try:
            simulator = MonteCarloSimulator(random_seed=42)
            mc_result = simulator.simulate(result, n_simulations=10000, show_progress=False)
            stats = mc_result.statistics
        except Exception as e:
            st.error(f"蒙地卡羅模擬失敗：{str(e)}")
            return

    # 四個關鍵指標卡片
    st.subheader("🎯 關鍵統計指標")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("賺錢機率", f"{stats['prob_profit']:.1%}")
    with col2:
        st.metric("平均報酬", f"{stats['mean_return']:.2%}")
    with col3:
        st.metric("95% VaR", f"{stats['ci_95_lower']:.2%}")
    with col4:
        st.metric("破產風險", f"{stats['ruin_risk']:.1%}")

    # 報酬率分布直方圖
    st.subheader("📊 報酬率分布")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=mc_result.simulations,
        nbinsx=100,
        name="報酬分布",
        marker_color="#1f77b4"
    ))
    fig_hist.add_vline(x=0, line_dash="solid", line_color="black",
                       annotation_text="盈虧分界", annotation_position="top")
    fig_hist.add_vline(x=stats["mean_return"], line_dash="dash", line_color="red",
                       annotation_text=f"平均 {stats['mean_return']:.1%}")
    fig_hist.add_vline(x=stats["ci_95_lower"], line_dash="dot", line_color="blue",
                       annotation_text="95% 下界")
    fig_hist.add_vline(x=stats["ci_95_upper"], line_dash="dot", line_color="blue",
                       annotation_text="95% 上界")
    fig_hist.update_layout(
        xaxis_title="最終報酬率",
        yaxis_title="出現次數",
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 詳細統計表
    with st.expander("📋 詳細統計表"):
        stats_df = pd.DataFrame([
            {"統計量": "平均報酬率", "數值": f"{stats['mean_return']:.2%}"},
            {"統計量": "中位數報酬率", "數值": f"{stats['median_return']:.2%}"},
            {"統計量": "標準差", "數值": f"{stats['std_return']:.2%}"},
            {"統計量": "偏態", "數值": f"{stats['skewness']:.2f}"},
            {"統計量": "峰態", "數值": f"{stats['kurtosis']:.2f}"},
            {"統計量": "5th 百分位", "數值": f"{stats['percentile_5']:.2%}"},
            {"統計量": "25th 百分位", "數值": f"{stats['percentile_25']:.2%}"},
            {"統計量": "75th 百分位", "數值": f"{stats['percentile_75']:.2%}"},
            {"統計量": "95th 百分位", "數值": f"{stats['percentile_95']:.2%}"},
            {"統計量": "95% CI 下界", "數值": f"{stats['ci_95_lower']:.2%}"},
            {"統計量": "95% CI 上界", "數值": f"{stats['ci_95_upper']:.2%}"},
            {"統計量": "99% CI 下界", "數值": f"{stats['ci_99_lower']:.2%}"},
            {"統計量": "99% CI 上界", "數值": f"{stats['ci_99_upper']:.2%}"},
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # 智慧診斷提示
    st.subheader("💡 智慧診斷")

    # 賺錢機率診斷
    prob = stats["prob_profit"]
    if prob >= 0.9:
        st.success(f"✅ 賺錢機率 {prob:.1%}，策略在各種市場情境下都具有穩健的獲利能力。")
    elif prob >= 0.8:
        st.info(f"💡 賺錢機率 {prob:.1%}，策略整體穩健，但仍有約 {1-prob:.0%} 的情境可能虧損。建議檢查死區閾值 (deadband) 是否過小，導致交易過於頻繁。")
    elif prob >= 0.6:
        st.warning(f"⚠️ 賺錢機率 {prob:.1%}，偏低。可能原因：(1) 死區閾值太小，手續費侵蝕獲利 → 建議加大 deadband (2) Ki 過大導致過度交易 → 建議降低 Ki")
    else:
        st.error(f"🚨 賺錢機率 {prob:.1%}，策略存在根本性問題。建議：(1) 大幅降低 Kp 和 Ki (2) 加大 deadband 到 3% 以上 (3) 檢查標的組合是否合理")

    # VaR 診斷
    var_95 = stats["ci_95_lower"]
    if var_95 > -0.1:
        st.success(f"✅ 95% VaR 為 {var_95:.1%}，下行風險控制良好。")
    elif var_95 > -0.2:
        st.info(f"💡 95% VaR 為 {var_95:.1%}，下行風險在可接受範圍。Kd（微分項）正在發揮「煞車」作用。")
    elif var_95 > -0.35:
        st.warning(f"⚠️ 95% VaR 為 {var_95:.1%}，下行風險偏高。建議加大 Kd 以強化崩盤保護機制，讓系統在市場急跌時更快停止買入。")
    else:
        st.error(f"🚨 95% VaR 為 {var_95:.1%}，下行風險過大。建議：(1) 大幅加大 Kd 到 0.2 以上 (2) 降低股票目標比例 (3) 加大 deadband 減少在下跌市場中的交易")

    # 分布形狀診斷
    std = stats["std_return"]
    skew = stats["skewness"]
    if std > 0.5:
        st.warning(f"⚠️ 報酬率標準差 {std:.1%}，分布過寬，策略結果不穩定。可能原因：Kp 過大導致調整幅度過大。建議降低 Kp。")
    if skew < -0.5:
        st.warning(f"⚠️ 偏態係數 {skew:.2f}，報酬分布左偏，代表大虧損的機率高於預期。建議加大 Kd 以增強下行保護。")

    # 破產風險診斷
    ruin = stats["ruin_risk"]
    if ruin > 0.05:
        st.error(f"🚨 破產風險 {ruin:.1%}（虧損>50%），策略需要重大調整。")
    elif ruin > 0.01:
        st.warning(f"⚠️ 破產風險 {ruin:.1%}，建議適度降低風險曝露。")
    else:
        st.success(f"✅ 破產風險 {ruin:.1%}，風險控制在安全範圍內。")


# =============================================================================
# Tab 4: 樣本外測試
# =============================================================================
def render_tab_out_of_sample(engine, data, params: dict, calculator):
    """渲染樣本外測試分頁"""
    st.header("🔬 樣本外測試")

    st.markdown("""
    將資料分為樣本內（2013-2023）和樣本外（2024-2025），
    驗證策略在未見過的資料上是否依然有效，檢測過擬合風險。
    """)

    with st.spinner("執行樣本外測試中..."):
        try:
            # 重置引擎
            engine.reset()

            validator = OutOfSampleValidator(risk_free_rate=0.02)
            oos_result = validator.validate(engine, data, split_date="2024-01-01")
        except Exception as e:
            st.error(f"樣本外測試失敗：{str(e)}")
            return

    # 過擬合判定大標題
    if oos_result.is_overfitted:
        st.error("⚠️ 過擬合警告：樣本外績效顯著低於樣本內，當前 PID 參數可能過度適應歷史資料。")
    else:
        st.success("✅ 通過樣本外驗證：策略在未見過的資料上表現一致，參數穩定可靠。")

    # 樣本內 vs 樣本外績效對比表
    st.subheader("📋 樣本內 vs 樣本外績效對比")

    is_metrics = oos_result.in_sample_metrics
    oos_metrics = oos_result.out_of_sample_metrics
    comparison = oos_result.comparison

    def safe_ratio(oos_val, is_val):
        if is_val == 0 or np.isinf(is_val) or np.isnan(is_val):
            return "N/A"
        ratio = oos_val / is_val
        if np.isinf(ratio) or np.isnan(ratio):
            return "N/A"
        return f"{ratio:.0%}"

    comparison_df = pd.DataFrame({
        "指標": ["年化報酬", "Sharpe", "最大回撤", "波動率", "勝率", "交易次數"],
        "樣本內 (2013-2023)": [
            f"{is_metrics.get('annual_return', 0):.2%}",
            format_metric(is_metrics.get('sharpe_ratio', 0)),
            f"{is_metrics.get('max_drawdown', 0):.2%}",
            f"{is_metrics.get('volatility', 0):.2%}",
            f"{is_metrics.get('win_rate', 0):.1%}",
            f"{is_metrics.get('total_trades', 0):,}"
        ],
        "樣本外 (2024-2025)": [
            f"{oos_metrics.get('annual_return', 0):.2%}",
            format_metric(oos_metrics.get('sharpe_ratio', 0)),
            f"{oos_metrics.get('max_drawdown', 0):.2%}",
            f"{oos_metrics.get('volatility', 0):.2%}",
            f"{oos_metrics.get('win_rate', 0):.1%}",
            f"{oos_metrics.get('total_trades', 0):,}"
        ],
        "比率 (外/內)": [
            safe_ratio(oos_metrics.get('annual_return', 0), is_metrics.get('annual_return', 0)),
            safe_ratio(oos_metrics.get('sharpe_ratio', 0), is_metrics.get('sharpe_ratio', 0)),
            safe_ratio(oos_metrics.get('max_drawdown', 0), is_metrics.get('max_drawdown', 0)),
            safe_ratio(oos_metrics.get('volatility', 0), is_metrics.get('volatility', 0)),
            safe_ratio(oos_metrics.get('win_rate', 0), is_metrics.get('win_rate', 0)),
            safe_ratio(oos_metrics.get('total_trades', 0), is_metrics.get('total_trades', 0))
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # 拼接淨值曲線
    st.subheader("📊 拼接淨值曲線")

    is_history = oos_result.in_sample_result.history
    oos_history = oos_result.out_of_sample_result.history

    # 調整樣本外淨值，使其接續樣本內
    is_final_nav = is_history["nav"].iloc[-1]
    oos_first_nav = oos_history["nav"].iloc[0]
    scale_factor = is_final_nav / oos_first_nav if oos_first_nav != 0 else 1
    oos_nav_adjusted = oos_history["nav"] * scale_factor

    fig_oos = go.Figure()
    fig_oos.add_trace(go.Scatter(
        x=is_history.index, y=is_history["nav"],
        mode="lines", name="樣本內 (2013-2023)",
        line=dict(color="#1f77b4", width=2)
    ))
    fig_oos.add_trace(go.Scatter(
        x=oos_history.index, y=oos_nav_adjusted,
        mode="lines", name="樣本外 (2024-2025)",
        line=dict(color="#ff7f0e", width=2)
    ))
    fig_oos.add_shape(
        type="line",
        x0="2024-01-01", x1="2024-01-01",
        y0=0, y1=1, yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig_oos.add_annotation(
        x="2024-01-01", y=1.05, yref="paper",
        text="2024-01-01 分割點", showarrow=False, font=dict(color="red")
    )
    fig_oos.update_layout(
        xaxis_title="日期",
        yaxis_title="淨值 (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_oos, use_container_width=True)

    # 智慧診斷提示
    st.subheader("💡 智慧診斷")

    # 整體一致性
    consistency = comparison.get("consistency_score", 0)
    if consistency > 0.7:
        st.success(f"✅ 一致性分數 {consistency:.2f}/1.00，樣本內外績效高度一致，PID 參數穩定。")
    elif consistency > 0.5:
        st.info(f"💡 一致性分數 {consistency:.2f}/1.00，績效略有衰退但在可接受範圍。")
    elif consistency > 0.3:
        st.warning(f"⚠️ 一致性分數 {consistency:.2f}/1.00，績效衰退明顯。建議：(1) 降低 Kp 和 Ki，減少對歷史模式的依賴 (2) 加大 deadband 讓策略更保守")
    else:
        st.error(f"🚨 一致性分數 {consistency:.2f}/1.00，嚴重過擬合。建議：(1) 大幅降低所有 PID 參數 (2) 考慮使用「保守」風險偏好重新測試")

    # Sharpe 比率衰退
    sharpe_ratio_ratio = comparison.get("performance_ratio", {}).get("sharpe_ratio", 1.0)
    if not np.isinf(sharpe_ratio_ratio) and not np.isnan(sharpe_ratio_ratio):
        if sharpe_ratio_ratio > 0.8:
            st.success(f"✅ Sharpe 比率維持度 {sharpe_ratio_ratio:.0%}，風險調整報酬在樣本外依然穩健。")
        elif sharpe_ratio_ratio > 0.5:
            st.info(f"💡 Sharpe 比率維持度 {sharpe_ratio_ratio:.0%}，略有下降。可能是 2024-2025 市場環境與歷史不同，而非參數問題。")
        else:
            suggested_kp = params['pid_params']['kp'] * 0.7
            st.warning(f"⚠️ Sharpe 比率維持度 {sharpe_ratio_ratio:.0%}，下降嚴重。Kp={params['pid_params']['kp']} 可能過大，對歷史波動模式過度敏感。建議降低至 {suggested_kp:.2f}。")

    # 交易次數比較
    is_trades = is_metrics.get("total_trades", 0)
    oos_trades = oos_metrics.get("total_trades", 0)
    is_days = is_metrics.get("trading_days", 1)
    oos_days = oos_metrics.get("trading_days", 1)
    is_freq = is_trades / is_days if is_days > 0 else 0
    oos_freq = oos_trades / oos_days if oos_days > 0 else 0
    if oos_freq > is_freq * 1.5 and is_freq > 0:
        st.warning(f"⚠️ 樣本外交易頻率 ({oos_freq:.3f}/天) 明顯高於樣本內 ({is_freq:.3f}/天)。Ki 的積分累積可能在新市場環境中導致過度交易。建議降低 Ki。")


# =============================================================================
# 輔助函數
# =============================================================================
def format_metric(value: float) -> str:
    """格式化指標數值，處理無窮大和 NaN"""
    if np.isinf(value):
        return "∞" if value > 0 else "-∞"
    if np.isnan(value):
        return "N/A"
    return f"{value:.2f}"


# =============================================================================
# 主程式
# =============================================================================
def main():
    """主程式入口"""
    params = render_sidebar()
    render_main_content(params)


if __name__ == "__main__":
    main()
