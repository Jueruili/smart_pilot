"""
Smart Pilot - 投資組合再平衡系統

Streamlit 單頁應用程式

功能：
- 參數設定（初始資金、目標比例、風險偏好）
- 執行回測
- 顯示績效指標和圖表
- 下載交易明細

作者：Smart Pilot Team
版本：1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.data_loader import DataLoader
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator


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
# 風險偏好對應的 PID 參數
# =============================================================================
RISK_PROFILES = {
    "保守": {"kp": 0.15, "ki": 0.02, "kd": 0.05},
    "穩健": {"kp": 0.3, "ki": 0.05, "kd": 0.1},
    "積極": {"kp": 0.5, "ki": 0.1, "kd": 0.2},
}


# =============================================================================
# 快取函數
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """載入並處理資料（使用快取避免重複下載）

    Args:
        tickers: 股票代碼列表
        start_date: 開始日期
        end_date: 結束日期

    Returns:
        pd.DataFrame: 處理後的價格資料
    """
    loader = DataLoader()
    data = loader.load_and_process(tickers, start_date, end_date)
    return data


# =============================================================================
# 側邊欄
# =============================================================================
def render_sidebar() -> dict:
    """渲染側邊欄並返回使用者輸入的參數

    Returns:
        dict: 包含所有使用者設定的參數
    """
    st.sidebar.header("⚙️ 參數設定")

    # 初始資金
    st.sidebar.subheader("💰 初始資金")
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
    st.sidebar.subheader("📈 資產配置")
    target_ratio = st.sidebar.slider(
        "目標股票比例",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        format="%d%%",
        help="股票（SPY）在投資組合中的目標比例，其餘為債券（TLT）"
    )

    # 標的設定
    st.sidebar.subheader("📌 投資標的")
    stock_ticker = st.sidebar.text_input(
        "股票標的", value="VTI",
        help="股票型 ETF 代碼（如 SPY, QQQ, VTI）"
    ).upper().strip()
    bond_ticker = st.sidebar.text_input(
        "債券標的", value="BND",
        help="債券型 ETF 代碼（如 TLT, BND, AGG）"
    ).upper().strip()
    
    # 風險偏好
    st.sidebar.subheader("🎯 風險偏好")
    risk_profile = st.sidebar.selectbox(
        "選擇風險偏好",
        options=list(RISK_PROFILES.keys()),
        index=1,  # 預設選「穩健」
        help="風險偏好決定 PID 控制器的參數設定"
    )

    # 顯示 PID 參數
    pid_params = RISK_PROFILES[risk_profile]
    with st.sidebar.expander("📋 PID 參數詳情"):
        st.write(f"**Kp (比例增益):** {pid_params['kp']}")
        st.write(f"**Ki (積分增益):** {pid_params['ki']}")
        st.write(f"**Kd (微分增益):** {pid_params['kd']}")

        # 風險偏好說明
        if risk_profile == "保守":
            st.info("💡 保守策略：較低的 Kp 和 Ki，較高的 Kd，反應較慢但更穩定。")
        elif risk_profile == "穩健":
            st.info("💡 穩健策略：平衡的參數設定，適合大多數投資者。")
        else:
            st.info("💡 積極策略：較高的 Kp 和 Ki，反應快速但可能有較大波動。")

    # 進階設定
    with st.sidebar.expander("🔧 進階設定"):
        deadband = st.slider(
            "死區閾值",
            min_value=0.0,
            max_value=0.05,
            value=0.01,
            step=0.005,
            format="%.3f",
            help="PID 調整量低於此閾值時不執行交易"
        )
        commission_rate = st.slider(
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

    # 系統資訊
    with st.sidebar.expander("ℹ️ 系統資訊"):
        st.write("**版本:** 1.0.0")
        st.write("**資料來源:** Yahoo Finance")
        st.write(f"**股票標的:** {stock_ticker}")
        st.write(f"**債券標的:** {bond_ticker}")
        st.write("**資料期間:** 2013-01-01 ~ 2025-12-31")

    return {
        "initial_cash": initial_cash,
        "target_ratio": target_ratio / 100,  # 轉換為小數
        "risk_profile": risk_profile,
        "pid_params": pid_params,
        "deadband": deadband,
        "commission_rate": commission_rate,
        "stock_ticker": stock_ticker,    
        "bond_ticker": bond_ticker, 
    }


# =============================================================================
# 主畫面
# =============================================================================
def render_main_content(params: dict):
    """渲染主畫面內容

    Args:
        params: 側邊欄返回的參數字典
    """
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

    st.markdown("---")

    # 開始回測按鈕
    if st.button("🚀 開始回測", type="primary", use_container_width=True):
        run_backtest(params)


def run_backtest(params: dict):
    """執行回測並顯示結果

    Args:
        params: 參數字典
    """
    try:
        with st.spinner("回測進行中..."):
            # a. 載入資料
            st.text("📥 載入資料中...")
            data = load_data(
                tickers=[params["stock_ticker"], params["bond_ticker"]],
                start_date="2013-01-01",
                end_date="2025-12-31"
            )
            data = data[[params["stock_ticker"], params["bond_ticker"]]]

            # b. 執行回測
            st.text("⚙️ 執行回測中...")
            engine = BacktestEngine(
                initial_cash=params["initial_cash"],
                target_ratio=params["target_ratio"],
                pid_params=params["pid_params"],
                deadband=params["deadband"],
                commission_rate=params["commission_rate"]
            )
            result = engine.run(data)

            # c. 計算績效指標
            st.text("📊 計算績效指標中...")
            calculator = MetricsCalculator(risk_free_rate=0.02)
            metrics = calculator.calculate_all_metrics(result.history)

        # 顯示成功訊息
        st.success("✅ 回測完成！")

        # 顯示結果
        display_results(result, metrics, params)

    except Exception as e:
        st.error(f"❌ 回測失敗：{str(e)}")
        with st.expander("錯誤詳情"):
            import traceback
            st.code(traceback.format_exc())


def display_results(result, metrics: dict, params: dict):
    """顯示回測結果

    Args:
        result: BacktestResult 物件
        metrics: 績效指標字典
        params: 參數字典
    """
    st.markdown("---")
    st.header("📈 回測結果")

    # ==========================================================================
    # 1. 三欄指標卡片
    # ==========================================================================
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
        sharpe_display = f"{sharpe:.2f}" if not np.isinf(sharpe) else "N/A"
        st.metric(
            label="夏普比率",
            value=sharpe_display,
            help="風險調整後報酬，>1 表示良好"
        )

    with col3:
        max_dd = metrics.get("max_drawdown", 0)
        st.metric(
            label="最大回撤",
            value=f"{max_dd:.2%}",
            delta=None,
            delta_color="inverse"
        )

    # 額外的指標
    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            label="總交易次數",
            value=f"{metrics.get('total_trades', 0):,}"
        )

    with col5:
        st.metric(
            label="總手續費",
            value=f"${metrics.get('total_commission', 0):,.2f}"
        )

    with col6:
        st.metric(
            label="勝率",
            value=f"{metrics.get('win_rate', 0):.1%}"
        )

    # ==========================================================================
    # 2. 資產淨值曲線圖
    # ==========================================================================
    st.markdown("---")
    st.subheader("📊 資產淨值曲線")

    history = result.history

    # 準備圖表資料
    chart_data = pd.DataFrame({
        "淨值": history["nav"]
    }, index=history.index)

    st.line_chart(chart_data, use_container_width=True)

    # 顯示股票比例變化
    with st.expander("📉 股票比例變化"):
        ratio_data = pd.DataFrame({
            "實際比例": history["ratio"],
            "目標比例": params["target_ratio"]
        }, index=history.index)
        st.line_chart(ratio_data, use_container_width=True)

    # ==========================================================================
    # 3. 詳細指標表格
    # ==========================================================================
    st.markdown("---")
    st.subheader("📋 詳細績效指標")

    # 整理指標為表格格式
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

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================================================
    # 4. 下載按鈕
    # ==========================================================================
    st.markdown("---")
    st.subheader("💾 下載資料")

    col1, col2 = st.columns(2)

    with col1:
        # 準備交易明細 CSV
        trade_history = history[history["trade_flag"] == True].copy()
        if len(trade_history) > 0:
            csv_data = trade_history.to_csv(index=True)
            st.download_button(
                label="📥 下載交易明細 (CSV)",
                data=csv_data,
                file_name="trade_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("無交易記錄可下載")

    with col2:
        # 完整歷史記錄
        full_csv = history.to_csv(index=True)
        st.download_button(
            label="📥 下載完整歷史 (CSV)",
            data=full_csv,
            file_name="backtest_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ==========================================================================
    # 5. 回測摘要
    # ==========================================================================
    st.markdown("---")
    with st.expander("📝 回測摘要"):
        st.markdown(f"""
        ### 回測設定
        - **初始資金:** ${params['initial_cash']:,.0f}
        - **目標股票比例:** {params['target_ratio']:.0%}
        - **風險偏好:** {params['risk_profile']}
        - **PID 參數:** Kp={params['pid_params']['kp']}, Ki={params['pid_params']['ki']}, Kd={params['pid_params']['kd']}
        - **死區閾值:** {params['deadband']:.1%}
        - **手續費率:** {params['commission_rate']:.2%}

        ### 回測期間
        - **開始日期:** {history.index[0].strftime('%Y-%m-%d')}
        - **結束日期:** {history.index[-1].strftime('%Y-%m-%d')}
        - **總交易日數:** {len(history):,} 天

        ### 績效摘要
        - **期初淨值:** ${history['nav'].iloc[0]:,.2f}
        - **期末淨值:** ${history['nav'].iloc[-1]:,.2f}
        - **總報酬率:** {metrics.get('total_return', 0):.2%}
        - **年化報酬率:** {metrics.get('annual_return', 0):.2%}
        """)


def format_metric(value: float) -> str:
    """格式化指標數值，處理無窮大和 NaN

    Args:
        value: 數值

    Returns:
        str: 格式化後的字串
    """
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
    # 渲染側邊欄並取得參數
    params = render_sidebar()

    # 渲染主畫面
    render_main_content(params)


if __name__ == "__main__":
    main()
