"""
蒙地卡羅模擬模組

使用 Bootstrap 方法對回測結果進行蒙地卡羅模擬，
評估策略的統計特性和風險。

主要功能：
- 報酬序列重組（Bootstrap with replacement）
- 統計指標分析（信賴區間、破產風險等）
- 可重現的隨機模擬

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from tqdm import tqdm

from core.backtest_engine import BacktestResult


# 預設參數
DEFAULT_N_SIMULATIONS: int = 10000
DEFAULT_RANDOM_SEED: Optional[int] = None


@dataclass
class MonteCarloResult:
    """蒙地卡羅模擬結果

    Attributes:
        simulations: 模擬結果陣列（每次模擬的最終報酬率）
        statistics: 統計指標字典
        n_simulations: 模擬次數
        n_periods: 模擬期數（交易日數）
        random_seed: 使用的隨機種子
    """
    simulations: np.ndarray
    statistics: dict
    n_simulations: int
    n_periods: int
    random_seed: Optional[int] = None


class MonteCarloSimulator:
    """蒙地卡羅模擬器

    使用 Bootstrap 方法從歷史回測結果中隨機重組報酬率，
    模擬多次投資路徑，評估策略的統計特性和風險。

    Attributes:
        random_seed: 隨機種子（用於結果可重現）
        rng: numpy 隨機數生成器

    Example:
        >>> from core.backtest_engine import BacktestEngine
        >>> from data.data_loader import DataLoader
        >>>
        >>> # 載入資料並執行回測
        >>> loader = DataLoader()
        >>> data = loader.load_and_process()
        >>> engine = BacktestEngine(target_ratio=0.6)
        >>> backtest_result = engine.run(data)
        >>>
        >>> # 執行蒙地卡羅模擬
        >>> simulator = MonteCarloSimulator(random_seed=42)
        >>> result = simulator.simulate(backtest_result, n_simulations=10000)
        >>>
        >>> # 查看統計結果
        >>> print(f"平均報酬率: {result.statistics['mean_return']:.2%}")
        >>> print(f"賺錢機率: {result.statistics['prob_profit']:.1%}")
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """初始化模擬器

        Args:
            random_seed: 隨機種子，設定後模擬結果可重現
                - None: 每次執行結果不同
                - 整數: 固定隨機種子，結果可重現
        """
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        print(f"[MonteCarloSimulator] 初始化完成")
        if random_seed is not None:
            print(f"[MonteCarloSimulator] 隨機種子: {random_seed}")

    def simulate(
        self,
        backtest_result: BacktestResult,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        show_progress: bool = True
    ) -> MonteCarloResult:
        """執行蒙地卡羅模擬

        從歷史回測結果提取每日報酬率，使用 Bootstrap 方法
        （有放回抽樣）隨機重組報酬率序列，模擬多次投資路徑。

        Bootstrap 方法說明：
        1. 從歷史的 n 個日報酬率中，有放回地隨機抽取 n 個
        2. 將抽取的報酬率累積計算最終淨值
        3. 重複 n_simulations 次

        Args:
            backtest_result: 回測引擎執行的結果
            n_simulations: 模擬次數，預設 10000 次
            show_progress: 是否顯示進度條

        Returns:
            MonteCarloResult: 包含模擬結果和統計指標

        Raises:
            ValueError: 當回測結果為空或無效時拋出

        Example:
            >>> result = simulator.simulate(backtest_result, n_simulations=10000)
            >>> print(f"模擬次數: {result.n_simulations}")
        """
        # 1. 提取每日報酬率
        daily_returns = self._extract_daily_returns(backtest_result)
        n_periods = len(daily_returns)

        print(f"\n[MonteCarloSimulator] 開始蒙地卡羅模擬")
        print(f"[MonteCarloSimulator] 歷史交易日數: {n_periods}")
        print(f"[MonteCarloSimulator] 模擬次數: {n_simulations:,}")

        # 2. 執行 Bootstrap 模擬
        simulations = self._run_bootstrap(
            daily_returns,
            n_simulations,
            show_progress
        )

        # 3. 分析結果
        statistics = self.analyze_results(simulations)

        print(f"[MonteCarloSimulator] 模擬完成")

        return MonteCarloResult(
            simulations=simulations,
            statistics=statistics,
            n_simulations=n_simulations,
            n_periods=n_periods,
            random_seed=self.random_seed
        )

    def _extract_daily_returns(self, backtest_result: BacktestResult) -> np.ndarray:
        """從回測結果提取每日報酬率

        計算公式：
            daily_return = (NAV_t - NAV_{t-1}) / NAV_{t-1}

        Args:
            backtest_result: 回測結果

        Returns:
            np.ndarray: 每日報酬率陣列

        Raises:
            ValueError: 當資料無效時拋出
        """
        history = backtest_result.history

        if history is None or len(history) < 2:
            raise ValueError("回測結果無效：歷史記錄為空或資料不足")

        # 檢查是否有 nav 欄位
        if "nav" not in history.columns:
            raise ValueError("回測結果無效：缺少 'nav' 欄位")

        # 計算每日報酬率
        nav = history["nav"].values
        daily_returns = np.diff(nav) / nav[:-1]

        # 移除 NaN 和 Inf
        valid_mask = np.isfinite(daily_returns)
        daily_returns = daily_returns[valid_mask]

        if len(daily_returns) == 0:
            raise ValueError("無有效的每日報酬率資料")

        return daily_returns

    def _run_bootstrap(
        self,
        daily_returns: np.ndarray,
        n_simulations: int,
        show_progress: bool
    ) -> np.ndarray:
        """執行 Bootstrap 模擬（向量化加速版本）

        使用 numpy 向量化操作加速計算，避免 Python 迴圈。

        Args:
            daily_returns: 歷史每日報酬率
            n_simulations: 模擬次數
            show_progress: 是否顯示進度條

        Returns:
            np.ndarray: 每次模擬的最終報酬率
        """
        n_periods = len(daily_returns)

        # 向量化方式：一次生成所有模擬的隨機索引
        # 形狀: (n_simulations, n_periods)
        random_indices = self.rng.integers(
            0, n_periods, size=(n_simulations, n_periods)
        )

        # 根據索引取得抽樣的報酬率
        # 形狀: (n_simulations, n_periods)
        sampled_returns = daily_returns[random_indices]

        # 計算累積報酬（沿著每次模擬的時間軸）
        # 使用 1 + r 的累積乘積
        if show_progress:
            # 分批處理以顯示進度
            batch_size = 1000
            n_batches = (n_simulations + batch_size - 1) // batch_size
            final_returns = np.zeros(n_simulations)

            with tqdm(total=n_simulations, desc="蒙地卡羅模擬") as pbar:
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_simulations)
                    batch_returns = sampled_returns[start_idx:end_idx]

                    # 累積報酬計算
                    cumulative = np.cumprod(1 + batch_returns, axis=1)
                    final_returns[start_idx:end_idx] = cumulative[:, -1] - 1

                    pbar.update(end_idx - start_idx)
        else:
            # 不顯示進度時直接計算
            cumulative = np.cumprod(1 + sampled_returns, axis=1)
            final_returns = cumulative[:, -1] - 1

        return final_returns

    def analyze_results(self, simulations: np.ndarray) -> dict:
        """分析模擬結果，計算統計指標

        計算的指標包括：
        - 平均報酬率（mean_return）
        - 標準差（std_return）
        - 賺錢機率（prob_profit）：報酬 > 0 的比例
        - 95% 信賴區間（ci_95）：2.5% 和 97.5% 分位數
        - 99% 信賴區間（ci_99）：0.5% 和 99.5% 分位數
        - 最好情況（best_case）：最高報酬
        - 最壞情況（worst_case）：最低報酬
        - 破產風險（ruin_risk）：虧損 > 50% 的比例

        Args:
            simulations: 模擬結果陣列（每次模擬的最終報酬率）

        Returns:
            dict: 統計指標字典，包含以下鍵值：
                - mean_return: float - 平均報酬率
                - std_return: float - 報酬率標準差
                - prob_profit: float - 賺錢機率
                - ci_95_lower: float - 95% CI 下界
                - ci_95_upper: float - 95% CI 上界
                - ci_99_lower: float - 99% CI 下界
                - ci_99_upper: float - 99% CI 上界
                - best_case: float - 最佳情況報酬
                - worst_case: float - 最差情況報酬
                - ruin_risk: float - 破產風險（虧損>50%）

        Example:
            >>> stats = simulator.analyze_results(simulations)
            >>> print(f"平均報酬: {stats['mean_return']:.2%}")
            >>> print(f"95% CI: [{stats['ci_95_lower']:.2%}, {stats['ci_95_upper']:.2%}]")
        """
        if len(simulations) == 0:
            raise ValueError("模擬結果為空")

        # 平均報酬率
        # mean = (1/n) * Σ(r_i)
        mean_return = np.mean(simulations)

        # 標準差
        # std = sqrt((1/n) * Σ(r_i - mean)^2)
        std_return = np.std(simulations)

        # 賺錢機率
        # prob_profit = count(r > 0) / n
        prob_profit = np.mean(simulations > 0)

        # 95% 信賴區間（2.5% 和 97.5% 分位數）
        ci_95_lower = np.percentile(simulations, 2.5)
        ci_95_upper = np.percentile(simulations, 97.5)

        # 99% 信賴區間（0.5% 和 99.5% 分位數）
        ci_99_lower = np.percentile(simulations, 0.5)
        ci_99_upper = np.percentile(simulations, 99.5)

        # 最好情況（最高報酬）
        best_case = np.max(simulations)

        # 最壞情況（最低報酬）
        worst_case = np.min(simulations)

        # 破產風險（虧損 > 50% 的比例）
        # ruin_risk = count(r < -0.5) / n
        ruin_risk = np.mean(simulations < -0.5)

        # 額外有用的統計量
        median_return = np.median(simulations)
        skewness = self._calculate_skewness(simulations, mean_return, std_return)
        kurtosis = self._calculate_kurtosis(simulations, mean_return, std_return)

        statistics = {
            # 基本統計
            "mean_return": mean_return,
            "std_return": std_return,
            "median_return": median_return,
            "skewness": skewness,
            "kurtosis": kurtosis,

            # 機率
            "prob_profit": prob_profit,
            "ruin_risk": ruin_risk,

            # 信賴區間
            "ci_95_lower": ci_95_lower,
            "ci_95_upper": ci_95_upper,
            "ci_99_lower": ci_99_lower,
            "ci_99_upper": ci_99_upper,

            # 極端情況
            "best_case": best_case,
            "worst_case": worst_case,

            # 其他分位數
            "percentile_5": np.percentile(simulations, 5),
            "percentile_25": np.percentile(simulations, 25),
            "percentile_75": np.percentile(simulations, 75),
            "percentile_95": np.percentile(simulations, 95),
        }

        return statistics

    def _calculate_skewness(
        self,
        data: np.ndarray,
        mean: float,
        std: float
    ) -> float:
        """計算偏態係數

        偏態係數公式：
            skewness = (1/n) * Σ((x_i - mean) / std)^3

        Args:
            data: 資料陣列
            mean: 平均值
            std: 標準差

        Returns:
            float: 偏態係數
        """
        if std == 0:
            return 0.0
        n = len(data)
        return np.sum(((data - mean) / std) ** 3) / n

    def _calculate_kurtosis(
        self,
        data: np.ndarray,
        mean: float,
        std: float
    ) -> float:
        """計算峰態係數（超額峰態）

        峰態係數公式：
            kurtosis = (1/n) * Σ((x_i - mean) / std)^4 - 3

        減去 3 是為了讓常態分布的峰態為 0（超額峰態）

        Args:
            data: 資料陣列
            mean: 平均值
            std: 標準差

        Returns:
            float: 峰態係數（超額峰態）
        """
        if std == 0:
            return 0.0
        n = len(data)
        return np.sum(((data - mean) / std) ** 4) / n - 3

    def get_report(self, result: MonteCarloResult) -> str:
        """生成模擬報告

        Args:
            result: 蒙地卡羅模擬結果

        Returns:
            str: 格式化的報告文字
        """
        stats = result.statistics

        report = f"""
=====================================
      蒙地卡羅模擬報告
=====================================

模擬設定:
  模擬次數:         {result.n_simulations:>12,}
  模擬期數(交易日): {result.n_periods:>12}
  隨機種子:         {result.random_seed if result.random_seed else '未設定':>12}

【報酬統計】
  平均報酬率:       {stats['mean_return']:>12.2%}
  中位數報酬率:     {stats['median_return']:>12.2%}
  標準差:           {stats['std_return']:>12.2%}
  偏態:             {stats['skewness']:>12.2f}
  峰態:             {stats['kurtosis']:>12.2f}

【機率估計】
  賺錢機率:         {stats['prob_profit']:>12.1%}
  破產風險(>50%虧損): {stats['ruin_risk']:>12.1%}

【信賴區間】
  95% CI:           [{stats['ci_95_lower']:>8.2%}, {stats['ci_95_upper']:>8.2%}]
  99% CI:           [{stats['ci_99_lower']:>8.2%}, {stats['ci_99_upper']:>8.2%}]

【極端情況】
  最佳情況:         {stats['best_case']:>12.2%}
  最差情況:         {stats['worst_case']:>12.2%}

【報酬分布】
   5th percentile:  {stats['percentile_5']:>12.2%}
  25th percentile:  {stats['percentile_25']:>12.2%}
  50th percentile:  {stats['median_return']:>12.2%}
  75th percentile:  {stats['percentile_75']:>12.2%}
  95th percentile:  {stats['percentile_95']:>12.2%}

=====================================
"""
        return report

    def reset_seed(self, new_seed: Optional[int] = None) -> None:
        """重設隨機種子

        Args:
            new_seed: 新的隨機種子，None 表示不固定種子
        """
        self.random_seed = new_seed
        self.rng = np.random.default_rng(new_seed)
        print(f"[MonteCarloSimulator] 隨機種子已重設為: {new_seed}")


def quick_simulate(
    backtest_result: BacktestResult,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    random_seed: Optional[int] = None
) -> MonteCarloResult:
    """快速執行蒙地卡羅模擬（便捷函數）

    Args:
        backtest_result: 回測結果
        n_simulations: 模擬次數
        random_seed: 隨機種子

    Returns:
        MonteCarloResult: 模擬結果

    Example:
        >>> from core.backtest_engine import BacktestEngine
        >>> engine = BacktestEngine()
        >>> backtest_result = engine.run(data)
        >>> mc_result = quick_simulate(backtest_result, n_simulations=10000, random_seed=42)
        >>> print(f"平均報酬: {mc_result.statistics['mean_return']:.2%}")
    """
    simulator = MonteCarloSimulator(random_seed=random_seed)
    return simulator.simulate(backtest_result, n_simulations)
