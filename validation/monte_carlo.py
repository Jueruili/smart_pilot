"""
蒙地卡羅模擬模組

使用蒙地卡羅方法評估策略的統計特性和風險。

方法包括：
- 報酬序列重排（Bootstrap）
- 參數敏感度分析
- 情境模擬

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional, Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class SimulationResult:
    """模擬結果

    Attributes:
        n_simulations: 模擬次數
        mean_return: 平均報酬率
        std_return: 報酬率標準差
        percentiles: 報酬率百分位數
        var_95: 95% VaR
        cvar_95: 95% CVaR (Expected Shortfall)
        prob_profit: 獲利機率
        prob_drawdown_gt_10: 回撤超過 10% 的機率
    """
    n_simulations: int
    mean_return: float
    std_return: float
    percentiles: dict[int, float]
    var_95: float
    cvar_95: float
    prob_profit: float
    prob_drawdown_gt_10: float


class MonteCarloSimulator:
    """蒙地卡羅模擬器

    使用蒙地卡羅方法分析策略的統計特性。

    Attributes:
        returns: 歷史報酬率序列
        n_simulations: 模擬次數

    Example:
        >>> simulator = MonteCarloSimulator(returns, n_simulations=10000)
        >>> result = simulator.bootstrap_simulation(n_periods=252)
    """

    def __init__(
        self,
        returns: pd.Series,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> None:
        """初始化模擬器

        Args:
            returns: 歷史報酬率序列
            n_simulations: 模擬次數，預設為 10000
            random_seed: 隨機種子，用於結果可重現

        Raises:
            ValueError: 當報酬率序列為空或模擬次數為非正數時拋出
        """
        if len(returns) == 0:
            raise ValueError("報酬率序列不能為空")
        if n_simulations <= 0:
            raise ValueError(f"模擬次數必須為正數，收到：{n_simulations}")

        self.returns = returns
        self.n_simulations = n_simulations

        if random_seed is not None:
            np.random.seed(random_seed)

    def bootstrap_simulation(
        self,
        n_periods: int = 252
    ) -> SimulationResult:
        """Bootstrap 模擬

        從歷史報酬率中隨機抽樣（有放回），模擬未來可能的報酬序列。

        Args:
            n_periods: 模擬期數，預設為 252（一年交易日）

        Returns:
            SimulationResult: 模擬結果

        Note:
            這種方法假設報酬率是獨立同分布的（i.i.d.），
            可能低估真實的尾部風險。
        """
        # 儲存每次模擬的總報酬
        final_returns = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)

        returns_array = self.returns.values

        for i in range(self.n_simulations):
            # 隨機抽樣報酬率
            sampled_returns = np.random.choice(
                returns_array,
                size=n_periods,
                replace=True
            )

            # 計算累積報酬
            cumulative = np.cumprod(1 + sampled_returns)
            final_returns[i] = cumulative[-1] - 1

            # 計算最大回撤
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown)

        # 計算統計量
        return self._compute_statistics(final_returns, max_drawdowns)

    def block_bootstrap(
        self,
        n_periods: int = 252,
        block_size: int = 21
    ) -> SimulationResult:
        """區塊 Bootstrap 模擬

        以區塊方式抽樣，保留報酬率序列中的自相關性。

        Args:
            n_periods: 模擬期數
            block_size: 區塊大小，預設為 21（約一個月）

        Returns:
            SimulationResult: 模擬結果

        Note:
            區塊 Bootstrap 比普通 Bootstrap 更能保留序列的時間相關性。
        """
        final_returns = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)

        returns_array = self.returns.values
        n_blocks = (n_periods + block_size - 1) // block_size

        for i in range(self.n_simulations):
            sampled_returns = []

            for _ in range(n_blocks):
                # 隨機選擇區塊起始點
                start_idx = np.random.randint(0, len(returns_array) - block_size + 1)
                block = returns_array[start_idx:start_idx + block_size]
                sampled_returns.extend(block)

            # 截取到指定期數
            sampled_returns = np.array(sampled_returns[:n_periods])

            # 計算累積報酬
            cumulative = np.cumprod(1 + sampled_returns)
            final_returns[i] = cumulative[-1] - 1

            # 計算最大回撤
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown)

        return self._compute_statistics(final_returns, max_drawdowns)

    def parameter_sensitivity(
        self,
        strategy_func: Callable[[dict], float],
        base_params: dict,
        param_ranges: dict[str, tuple[float, float]]
    ) -> dict[str, np.ndarray]:
        """參數敏感度分析

        評估策略績效對參數變化的敏感度。

        Args:
            strategy_func: 策略函數，接受參數字典，返回績效指標
            base_params: 基準參數
            param_ranges: 參數範圍，格式為 {參數名: (最小值, 最大值)}

        Returns:
            dict: 各參數的敏感度結果

        TODO:
            - 實作參數敏感度分析
        """
        # TODO: 實作參數敏感度分析
        raise NotImplementedError("參數敏感度分析尚未實作")

    def _compute_statistics(
        self,
        final_returns: np.ndarray,
        max_drawdowns: np.ndarray
    ) -> SimulationResult:
        """計算模擬統計量

        Args:
            final_returns: 最終報酬率陣列
            max_drawdowns: 最大回撤陣列

        Returns:
            SimulationResult: 統計結果
        """
        # 計算百分位數
        percentiles = {
            5: np.percentile(final_returns, 5),
            25: np.percentile(final_returns, 25),
            50: np.percentile(final_returns, 50),
            75: np.percentile(final_returns, 75),
            95: np.percentile(final_returns, 95),
        }

        # 計算 VaR 和 CVaR
        var_95 = np.percentile(final_returns, 5)  # 5% 最差情況
        cvar_95 = final_returns[final_returns <= var_95].mean()

        # 計算機率
        prob_profit = np.mean(final_returns > 0)
        prob_drawdown_gt_10 = np.mean(max_drawdowns < -0.10)

        return SimulationResult(
            n_simulations=self.n_simulations,
            mean_return=np.mean(final_returns),
            std_return=np.std(final_returns),
            percentiles=percentiles,
            var_95=var_95,
            cvar_95=cvar_95,
            prob_profit=prob_profit,
            prob_drawdown_gt_10=prob_drawdown_gt_10,
        )

    def plot_distribution(
        self,
        result: SimulationResult,
        title: str = "報酬分布"
    ) -> None:
        """繪製報酬分布圖

        Args:
            result: 模擬結果
            title: 圖表標題

        TODO:
            - 移至 visualization 模組
        """
        # TODO: 實作繪圖功能
        raise NotImplementedError("繪圖功能請使用 visualization 模組")

    def summary_report(self, result: SimulationResult) -> str:
        """生成模擬摘要報告

        Args:
            result: 模擬結果

        Returns:
            str: 格式化的報告文字
        """
        report = f"""
=====================================
     蒙地卡羅模擬報告
=====================================

模擬次數: {result.n_simulations:,}

【報酬統計】
  平均報酬率:     {result.mean_return:>10.2%}
  報酬率標準差:   {result.std_return:>10.2%}

【百分位數】
   5th:          {result.percentiles[5]:>10.2%}
  25th:          {result.percentiles[25]:>10.2%}
  50th (中位數): {result.percentiles[50]:>10.2%}
  75th:          {result.percentiles[75]:>10.2%}
  95th:          {result.percentiles[95]:>10.2%}

【風險指標】
  95% VaR:       {result.var_95:>10.2%}
  95% CVaR:      {result.cvar_95:>10.2%}

【機率估計】
  獲利機率:      {result.prob_profit:>10.1%}
  回撤>10%機率:  {result.prob_drawdown_gt_10:>10.1%}

=====================================
"""
        return report
