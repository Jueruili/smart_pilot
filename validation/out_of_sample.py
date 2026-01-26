"""
樣本外測試模組

實作樣本外驗證方法，用於評估策略的泛化能力。

方法包括：
- 簡單樣本外測試（Train/Test Split）
- 滾動窗口測試（Rolling Window）
- Walk-Forward 測試

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional, Callable, Generator
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """驗證結果

    Attributes:
        train_period: 訓練期間
        test_period: 測試期間
        train_metrics: 訓練期績效指標
        test_metrics: 測試期績效指標
        parameters: 使用的參數
    """
    train_period: tuple[str, str]
    test_period: tuple[str, str]
    train_metrics: dict
    test_metrics: dict
    parameters: dict


class OutOfSampleValidator:
    """樣本外測試驗證器

    用於執行各種樣本外測試方法，評估策略在未見過數據上的表現。

    Attributes:
        data: 歷史數據
        test_ratio: 測試集比例

    Example:
        >>> validator = OutOfSampleValidator(price_data, test_ratio=0.3)
        >>> results = validator.simple_split(strategy_func)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        test_ratio: float = 0.3
    ) -> None:
        """初始化驗證器

        Args:
            data: 歷史數據，index 為日期
            test_ratio: 測試集比例，預設為 30%

        Raises:
            ValueError: 當測試比例不在 0 到 1 之間時拋出
        """
        if not 0 < test_ratio < 1:
            raise ValueError(f"測試比例必須在 0 到 1 之間，收到：{test_ratio}")

        if data.empty:
            raise ValueError("數據不能為空")

        self.data = data
        self.test_ratio = test_ratio

    def simple_split(
        self,
        strategy_func: Callable[[pd.DataFrame, dict], dict],
        parameters: dict
    ) -> ValidationResult:
        """簡單分割測試

        將數據分為訓練集和測試集，分別評估策略表現。

        Args:
            strategy_func: 策略函數，接受 (data, parameters) 返回績效指標
            parameters: 策略參數

        Returns:
            ValidationResult: 驗證結果

        Example:
            >>> def my_strategy(data, params):
            ...     # 執行策略並返回指標
            ...     return {"sharpe": 1.5, "return": 0.15}
            >>> result = validator.simple_split(my_strategy, {"kp": 0.5})

        TODO:
            - 實作完整的分割測試邏輯
        """
        # 計算分割點
        n = len(self.data)
        split_idx = int(n * (1 - self.test_ratio))

        # 分割數據
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]

        # TODO: 執行策略並計算指標
        raise NotImplementedError("簡單分割測試尚未完整實作")

    def rolling_window(
        self,
        strategy_func: Callable[[pd.DataFrame, dict], dict],
        parameters: dict,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 21
    ) -> list[ValidationResult]:
        """滾動窗口測試

        使用滾動窗口方法，模擬實際交易中的策略表現。

        Args:
            strategy_func: 策略函數
            parameters: 策略參數
            train_window: 訓練窗口大小（交易日），預設為 252（一年）
            test_window: 測試窗口大小（交易日），預設為 63（一季）
            step: 滾動步長（交易日），預設為 21（一個月）

        Returns:
            list[ValidationResult]: 每個窗口的驗證結果列表

        TODO:
            - 實作滾動窗口邏輯
        """
        results = []

        # TODO: 實作滾動窗口測試
        raise NotImplementedError("滾動窗口測試尚未實作")

        return results

    def walk_forward(
        self,
        strategy_func: Callable[[pd.DataFrame, dict], dict],
        parameter_space: dict,
        train_window: int = 252,
        test_window: int = 63
    ) -> list[ValidationResult]:
        """Walk-Forward 測試

        在每個測試期之前，使用訓練期數據優化參數。
        這是最嚴格的樣本外測試方法。

        Args:
            strategy_func: 策略函數
            parameter_space: 參數搜索空間
            train_window: 訓練窗口大小
            test_window: 測試窗口大小

        Returns:
            list[ValidationResult]: 驗證結果列表

        TODO:
            - 實作 Walk-Forward 測試
            - 整合參數優化
        """
        # TODO: 實作 Walk-Forward 測試
        raise NotImplementedError("Walk-Forward 測試尚未實作")

    def generate_windows(
        self,
        train_window: int,
        test_window: int,
        step: int
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """生成訓練/測試窗口

        Args:
            train_window: 訓練窗口大小
            test_window: 測試窗口大小
            step: 滾動步長

        Yields:
            tuple: (訓練數據, 測試數據)
        """
        n = len(self.data)
        start = 0

        while start + train_window + test_window <= n:
            train_end = start + train_window
            test_end = train_end + test_window

            train_data = self.data.iloc[start:train_end]
            test_data = self.data.iloc[train_end:test_end]

            yield train_data, test_data

            start += step

    def calculate_stability_ratio(
        self,
        results: list[ValidationResult]
    ) -> float:
        """計算穩定性比率

        衡量訓練期和測試期績效的一致性。

        Args:
            results: 驗證結果列表

        Returns:
            float: 穩定性比率（0-1，越高越穩定）

        Formula:
            穩定性比率 = 測試期平均績效 / 訓練期平均績效

        Note:
            比率接近 1 表示策略穩定，過高或過低都可能有問題。
        """
        if not results:
            return 0.0

        train_sharpes = [r.train_metrics.get("sharpe_ratio", 0) for r in results]
        test_sharpes = [r.test_metrics.get("sharpe_ratio", 0) for r in results]

        avg_train = np.mean(train_sharpes) if train_sharpes else 0
        avg_test = np.mean(test_sharpes) if test_sharpes else 0

        if avg_train == 0:
            return 0.0

        return avg_test / avg_train

    def summary_report(self, results: list[ValidationResult]) -> str:
        """生成驗證摘要報告

        Args:
            results: 驗證結果列表

        Returns:
            str: 格式化的報告文字
        """
        if not results:
            return "沒有驗證結果"

        # 收集測試期指標
        test_returns = [r.test_metrics.get("total_return", 0) for r in results]
        test_sharpes = [r.test_metrics.get("sharpe_ratio", 0) for r in results]
        test_max_dds = [r.test_metrics.get("max_drawdown", 0) for r in results]

        stability = self.calculate_stability_ratio(results)

        report = f"""
=====================================
     樣本外驗證摘要報告
=====================================

測試窗口數量: {len(results)}

【測試期績效統計】
  平均報酬率:   {np.mean(test_returns):>10.2%}
  報酬率標準差: {np.std(test_returns):>10.2%}
  最佳報酬率:   {np.max(test_returns):>10.2%}
  最差報酬率:   {np.min(test_returns):>10.2%}

  平均夏普比率: {np.mean(test_sharpes):>10.2f}
  平均最大回撤: {np.mean(test_max_dds):>10.2%}

【穩定性指標】
  穩定性比率:   {stability:>10.2f}

=====================================
"""
        return report
