"""
圖表生成模組

使用 Plotly 生成互動式圖表，適用於 Streamlit 應用。

圖表類型：
- 累積報酬曲線
- 回撤分析圖
- 權重分配圖
- PID 控制過程圖

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartGenerator:
    """圖表生成器

    生成各種投資組合分析圖表。

    Attributes:
        theme: 圖表主題（light/dark）

    Example:
        >>> generator = ChartGenerator(theme="dark")
        >>> fig = generator.cumulative_returns(returns_df)
        >>> fig.show()  # 或在 Streamlit 中使用 st.plotly_chart(fig)
    """

    def __init__(self, theme: str = "light") -> None:
        """初始化圖表生成器

        Args:
            theme: 圖表主題，"light" 或 "dark"
        """
        self.theme = theme
        self._template = "plotly_white" if theme == "light" else "plotly_dark"

    def cumulative_returns(
        self,
        returns: pd.DataFrame,
        title: str = "累積報酬曲線",
        benchmark: Optional[pd.Series] = None
    ) -> go.Figure:
        """繪製累積報酬曲線

        Args:
            returns: 報酬率 DataFrame，columns 為策略名稱
            title: 圖表標題
            benchmark: 基準報酬率序列（可選）

        Returns:
            go.Figure: Plotly 圖表物件
        """
        fig = go.Figure()

        # 計算累積報酬
        for col in returns.columns:
            cumulative = (1 + returns[col]).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values * 100,  # 轉為百分比
                mode='lines',
                name=col,
                hovertemplate='%{x}<br>報酬: %{y:.2f}%<extra></extra>'
            ))

        # 加入基準
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark.values * 100,
                mode='lines',
                name='基準',
                line=dict(dash='dash', color='gray'),
                hovertemplate='%{x}<br>基準: %{y:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="累積報酬 (%)",
            template=self._template,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def drawdown(
        self,
        returns: pd.Series,
        title: str = "回撤分析"
    ) -> go.Figure:
        """繪製回撤圖

        Args:
            returns: 報酬率序列
            title: 圖表標題

        Returns:
            go.Figure: Plotly 圖表物件
        """
        # 計算累積淨值和回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('淨值曲線', '回撤')
        )

        # 淨值曲線
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name='淨值',
                line=dict(color='#2196F3')
            ),
            row=1, col=1
        )

        # 歷史高點
        fig.add_trace(
            go.Scatter(
                x=running_max.index,
                y=running_max.values,
                mode='lines',
                name='歷史高點',
                line=dict(dash='dash', color='gray')
            ),
            row=1, col=1
        )

        # 回撤
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='回撤',
                fill='tozeroy',
                line=dict(color='#F44336')
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=title,
            template=self._template,
            height=600,
            showlegend=True
        )

        fig.update_yaxes(title_text="淨值", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        return fig

    def weight_allocation(
        self,
        weights: pd.DataFrame,
        title: str = "權重配置變化"
    ) -> go.Figure:
        """繪製權重配置變化圖

        Args:
            weights: 權重 DataFrame，index 為日期，columns 為資產名稱
            title: 圖表標題

        Returns:
            go.Figure: Plotly 圖表物件
        """
        fig = go.Figure()

        for col in weights.columns:
            fig.add_trace(go.Scatter(
                x=weights.index,
                y=weights[col].values * 100,
                mode='lines',
                name=col,
                stackgroup='one',  # 堆疊面積圖
                hovertemplate='%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="權重 (%)",
            template=self._template,
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )

        return fig

    def weight_pie(
        self,
        weights: dict[str, float],
        title: str = "當前權重配置"
    ) -> go.Figure:
        """繪製權重圓餅圖

        Args:
            weights: 權重字典
            title: 圖表標題

        Returns:
            go.Figure: Plotly 圖表物件
        """
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=[v * 100 for v in weights.values()],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])

        fig.update_layout(
            title=title,
            template=self._template,
        )

        return fig

    def pid_control_process(
        self,
        errors: pd.Series,
        adjustments: pd.Series,
        actual_weights: pd.Series,
        target_weight: float,
        title: str = "PID 控制過程"
    ) -> go.Figure:
        """繪製 PID 控制過程圖

        Args:
            errors: 誤差序列
            adjustments: 調整量序列
            actual_weights: 實際權重序列
            target_weight: 目標權重
            title: 圖表標題

        Returns:
            go.Figure: Plotly 圖表物件
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('權重變化', '誤差 (e)', '調整量 (Δu)')
        )

        # 實際權重 vs 目標權重
        fig.add_trace(
            go.Scatter(
                x=actual_weights.index,
                y=actual_weights.values * 100,
                mode='lines',
                name='實際權重',
                line=dict(color='#2196F3')
            ),
            row=1, col=1
        )

        fig.add_hline(
            y=target_weight * 100,
            line_dash="dash",
            line_color="green",
            annotation_text="目標",
            row=1, col=1
        )

        # 誤差
        fig.add_trace(
            go.Scatter(
                x=errors.index,
                y=errors.values * 100,
                mode='lines+markers',
                name='誤差',
                line=dict(color='#FF9800')
            ),
            row=2, col=1
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        # 調整量
        fig.add_trace(
            go.Bar(
                x=adjustments.index,
                y=adjustments.values * 100,
                name='調整量',
                marker_color=np.where(
                    adjustments.values >= 0, '#4CAF50', '#F44336'
                )
            ),
            row=3, col=1
        )

        fig.update_layout(
            title=title,
            template=self._template,
            height=700,
            showlegend=True
        )

        fig.update_yaxes(title_text="權重 (%)", row=1, col=1)
        fig.update_yaxes(title_text="誤差 (%)", row=2, col=1)
        fig.update_yaxes(title_text="調整量 (%)", row=3, col=1)
        fig.update_xaxes(title_text="時間", row=3, col=1)

        return fig

    def comparison_bar(
        self,
        metrics: dict[str, dict[str, float]],
        metric_names: Optional[list[str]] = None,
        title: str = "策略比較"
    ) -> go.Figure:
        """繪製策略比較長條圖

        Args:
            metrics: 指標字典，格式為 {策略名: {指標名: 值}}
            metric_names: 要顯示的指標名稱列表
            title: 圖表標題

        Returns:
            go.Figure: Plotly 圖表物件
        """
        strategies = list(metrics.keys())

        if metric_names is None:
            # 使用第一個策略的所有指標
            metric_names = list(metrics[strategies[0]].keys())

        fig = go.Figure()

        for metric in metric_names:
            values = [metrics[s].get(metric, 0) for s in strategies]

            fig.add_trace(go.Bar(
                name=metric,
                x=strategies,
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="策略",
            yaxis_title="指標值",
            template=self._template,
            barmode='group'
        )

        return fig

    def monte_carlo_distribution(
        self,
        returns: np.ndarray,
        percentiles: dict[int, float],
        title: str = "蒙地卡羅模擬分布"
    ) -> go.Figure:
        """繪製蒙地卡羅模擬報酬分布

        Args:
            returns: 模擬報酬率陣列
            percentiles: 百分位數字典
            title: 圖表標題

        Returns:
            go.Figure: Plotly 圖表物件
        """
        fig = go.Figure()

        # 直方圖
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='分布',
            marker_color='#2196F3',
            opacity=0.7
        ))

        # 加入百分位數線
        colors = {5: '#F44336', 50: '#4CAF50', 95: '#F44336'}
        for pct, value in percentiles.items():
            if pct in [5, 50, 95]:
                fig.add_vline(
                    x=value * 100,
                    line_dash="dash",
                    line_color=colors.get(pct, 'gray'),
                    annotation_text=f'{pct}th: {value*100:.1f}%'
                )

        fig.update_layout(
            title=title,
            xaxis_title="報酬率 (%)",
            yaxis_title="頻率",
            template=self._template,
        )

        return fig
