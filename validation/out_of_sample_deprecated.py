# DEPRECATED: 此檔案已被新架構取代，保留供參考用
# 取代者：core/pd_controller.py, core/optimizer.py
# 棄用日期：2025-03

"""
樣本外測試模組

實作樣本外驗證方法，用於評估策略的泛化能力，檢測過擬合問題。

主要功能：
- 資料分割（樣本內/樣本外）
- 樣本外驗證
- 績效比較與過擬合檢測

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

from core.backtest_engine import BacktestEngine, BacktestResult
from core.metrics import MetricsCalculator


# 預設參數
DEFAULT_SPLIT_DATE: str = "2024-01-01"


@dataclass
class ValidationResult:
    """樣本外驗證結果

    Attributes:
        in_sample_result: 樣本內回測結果
        out_of_sample_result: 樣本外回測結果
        in_sample_metrics: 樣本內績效指標
        out_of_sample_metrics: 樣本外績效指標
        comparison: 績效比較報告
        is_overfitted: 是否過擬合
        split_date: 分割日期
    """
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult
    in_sample_metrics: dict
    out_of_sample_metrics: dict
    comparison: dict
    is_overfitted: bool
    split_date: str


class OutOfSampleValidator:
    """樣本外測試驗證器

    用於執行樣本外測試，評估策略在未見過數據上的表現，檢測過擬合。

    Attributes:
        metrics_calculator: 績效指標計算器

    Example:
        >>> from core.backtest_engine import BacktestEngine
        >>> from data.data_loader import DataLoader
        >>>
        >>> # 載入資料
        >>> loader = DataLoader()
        >>> data = loader.load_and_process()
        >>>
        >>> # 建立驗證器
        >>> validator = OutOfSampleValidator()
        >>>
        >>> # 建立回測引擎
        >>> engine = BacktestEngine(target_ratio=0.6)
        >>>
        >>> # 執行樣本外驗證
        >>> result = validator.validate(engine, data, split_date="2023-01-01")
        >>> print(f"是否過擬合: {result.is_overfitted}")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        overfitting_threshold: float = 0.5
    ) -> None:
        """初始化驗證器

        Args:
            risk_free_rate: 無風險利率，用於計算績效指標
            overfitting_threshold: 過擬合閾值
                - 當樣本外績效 / 樣本內績效 < 此閾值時，判定為過擬合
                - 預設為 0.5（樣本外績效低於樣本內的 50%）
        """
        self.metrics_calculator = MetricsCalculator(risk_free_rate=risk_free_rate)
        self.overfitting_threshold = overfitting_threshold

        print(f"[OutOfSampleValidator] 初始化完成")
        print(f"[OutOfSampleValidator] 過擬合閾值: {overfitting_threshold:.0%}")

    def split_data(
        self,
        data: pd.DataFrame,
        split_date: str = DEFAULT_SPLIT_DATE
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """分割資料為樣本內和樣本外

        Args:
            data: 完整的價格資料 DataFrame
            split_date: 分割日期，格式 "YYYY-MM-DD"
                - 此日期之前的資料為樣本內（訓練集）
                - 此日期及之後的資料為樣本外（測試集）

        Returns:
            tuple: (in_sample_data, out_of_sample_data)
                - in_sample_data: 樣本內資料
                - out_of_sample_data: 樣本外資料

        Raises:
            ValueError: 當資料為空或分割日期無效時拋出

        Example:
            >>> in_sample, out_of_sample = validator.split_data(data, "2023-01-01")
            >>> print(f"樣本內: {len(in_sample)} 天, 樣本外: {len(out_of_sample)} 天")
        """
        if data.empty:
            raise ValueError("輸入資料為空")

        # 將分割日期轉換為 datetime
        try:
            split_datetime = pd.to_datetime(split_date)
        except Exception as e:
            raise ValueError(f"無效的分割日期格式: {split_date}") from e

        # 確保資料索引是 datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # 分割資料
        in_sample_data = data[data.index < split_datetime]
        out_of_sample_data = data[data.index >= split_datetime]

        # 驗證分割結果
        if len(in_sample_data) == 0:
            raise ValueError(
                f"樣本內資料為空，分割日期 {split_date} 可能在資料範圍之前。"
                f"資料範圍: {data.index[0].date()} ~ {data.index[-1].date()}"
            )
        if len(out_of_sample_data) == 0:
            raise ValueError(
                f"樣本外資料為空，分割日期 {split_date} 可能在資料範圍之後。"
                f"資料範圍: {data.index[0].date()} ~ {data.index[-1].date()}"
            )

        print(f"[OutOfSampleValidator] 資料分割完成")
        print(f"[OutOfSampleValidator] 分割日期: {split_date}")
        print(f"[OutOfSampleValidator] 樣本內: {in_sample_data.index[0].date()} ~ "
              f"{in_sample_data.index[-1].date()} ({len(in_sample_data)} 天)")
        print(f"[OutOfSampleValidator] 樣本外: {out_of_sample_data.index[0].date()} ~ "
              f"{out_of_sample_data.index[-1].date()} ({len(out_of_sample_data)} 天)")

        return in_sample_data, out_of_sample_data

    def validate(
        self,
        backtest_engine: BacktestEngine,
        data: pd.DataFrame,
        split_date: str = DEFAULT_SPLIT_DATE
    ) -> ValidationResult:
        """執行樣本外驗證

        使用相同的參數在樣本內和樣本外資料上執行回測，比較績效。

        Args:
            backtest_engine: 已配置好參數的回測引擎
            data: 完整的價格資料
            split_date: 分割日期

        Returns:
            ValidationResult: 驗證結果，包含兩個時期的回測結果和比較

        Example:
            >>> engine = BacktestEngine(target_ratio=0.6, pid_params={"kp": 1.0})
            >>> result = validator.validate(engine, data, "2023-01-01")
            >>> print(f"樣本內報酬: {result.in_sample_metrics['total_return']:.2%}")
            >>> print(f"樣本外報酬: {result.out_of_sample_metrics['total_return']:.2%}")
        """
        print(f"\n[OutOfSampleValidator] 開始樣本外驗證")

        # 1. 分割資料
        in_sample_data, out_of_sample_data = self.split_data(data, split_date)

        # 2. 樣本內回測
        print(f"\n[OutOfSampleValidator] 執行樣本內回測...")
        backtest_engine.reset()
        in_sample_result = backtest_engine.run(in_sample_data)

        # 3. 樣本外回測（使用相同參數）
        print(f"\n[OutOfSampleValidator] 執行樣本外回測...")
        backtest_engine.reset()
        out_of_sample_result = backtest_engine.run(out_of_sample_data)

        # 4. 計算績效指標
        in_sample_metrics = self.metrics_calculator.calculate_all_metrics(
            in_sample_result.history
        )
        out_of_sample_metrics = self.metrics_calculator.calculate_all_metrics(
            out_of_sample_result.history
        )

        # 5. 比較績效
        comparison = self.compare_performance(in_sample_metrics, out_of_sample_metrics)

        # 6. 判斷是否過擬合
        is_overfitted = self._check_overfitting(comparison)

        print(f"\n[OutOfSampleValidator] 驗證完成")
        print(f"[OutOfSampleValidator] 是否過擬合: {'是' if is_overfitted else '否'}")

        return ValidationResult(
            in_sample_result=in_sample_result,
            out_of_sample_result=out_of_sample_result,
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=out_of_sample_metrics,
            comparison=comparison,
            is_overfitted=is_overfitted,
            split_date=split_date,
        )

    def compare_performance(
        self,
        in_sample_metrics: dict,
        out_of_sample_metrics: dict
    ) -> dict:
        """比較樣本內 vs 樣本外的績效差異

        Args:
            in_sample_metrics: 樣本內績效指標
            out_of_sample_metrics: 樣本外績效指標

        Returns:
            dict: 比較報告，包含：
                - metrics_comparison: 各指標的比較
                - performance_ratio: 績效比率（樣本外/樣本內）
                - degradation: 績效衰退程度
                - consistency_score: 一致性分數

        Example:
            >>> comparison = validator.compare_performance(is_metrics, oos_metrics)
            >>> print(f"績效比率: {comparison['performance_ratio']:.2f}")
        """
        comparison = {
            "metrics_comparison": {},
            "performance_ratio": {},
            "degradation": {},
            "consistency_score": 0.0,
        }

        # 比較各項指標
        metrics_to_compare = [
            "total_return",
            "annual_return",
            "max_drawdown",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "win_rate",
        ]

        consistency_scores = []

        for metric in metrics_to_compare:
            is_value = in_sample_metrics.get(metric, 0)
            oos_value = out_of_sample_metrics.get(metric, 0)

            comparison["metrics_comparison"][metric] = {
                "in_sample": is_value,
                "out_of_sample": oos_value,
                "difference": oos_value - is_value,
            }

            # 計算比率（避免除以零）
            if is_value != 0 and not np.isinf(is_value):
                ratio = oos_value / is_value if not np.isinf(oos_value) else 0
            else:
                ratio = 1.0 if oos_value == 0 else 0.0

            comparison["performance_ratio"][metric] = ratio

            # 計算衰退程度（正值表示衰退）
            if metric == "max_drawdown":
                # 回撤是負值，樣本外更負表示更差
                degradation = oos_value - is_value  # 更負 = 負的衰退
            else:
                # 其他指標，樣本外較低表示衰退
                degradation = is_value - oos_value

            comparison["degradation"][metric] = degradation

            # 一致性分數（比率接近 1 為好）
            if metric not in ["max_drawdown"]:  # 排除回撤
                if not np.isinf(ratio) and not np.isnan(ratio):
                    # 比率在 0.5-1.5 之間給高分
                    if 0.5 <= ratio <= 1.5:
                        consistency_scores.append(1.0 - abs(1.0 - ratio))
                    else:
                        consistency_scores.append(0.0)

        # 計算整體一致性分數
        if consistency_scores:
            comparison["consistency_score"] = np.mean(consistency_scores)

        # 添加總結
        comparison["summary"] = {
            "sharpe_ratio_ratio": comparison["performance_ratio"].get("sharpe_ratio", 0),
            "return_ratio": comparison["performance_ratio"].get("total_return", 0),
            "is_consistent": comparison["consistency_score"] > 0.5,
        }

        return comparison

    def _check_overfitting(self, comparison: dict) -> bool:
        """檢查是否過擬合

        過擬合判斷標準：
        1. 夏普比率大幅下降
        2. 報酬率大幅下降
        3. 一致性分數過低

        Args:
            comparison: 績效比較結果

        Returns:
            bool: True 表示可能過擬合
        """
        # 取得各項比率
        sharpe_ratio = comparison["performance_ratio"].get("sharpe_ratio", 1.0)
        return_ratio = comparison["performance_ratio"].get("total_return", 1.0)
        consistency = comparison["consistency_score"]

        # 處理 inf 和 nan
        if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
            sharpe_ratio = 0.0
        if np.isinf(return_ratio) or np.isnan(return_ratio):
            return_ratio = 0.0

        # 過擬合判斷
        # 1. 夏普比率下降超過閾值
        sharpe_degraded = sharpe_ratio < self.overfitting_threshold

        # 2. 報酬率下降超過閾值
        return_degraded = return_ratio < self.overfitting_threshold

        # 3. 一致性分數過低
        inconsistent = consistency < 0.3

        # 任一條件成立即判定為過擬合
        is_overfitted = sharpe_degraded or return_degraded or inconsistent

        return is_overfitted

    def get_report(self, result: ValidationResult) -> str:
        """生成驗證報告

        Args:
            result: 驗證結果

        Returns:
            str: 格式化的報告文字
        """
        is_metrics = result.in_sample_metrics
        oos_metrics = result.out_of_sample_metrics
        comp = result.comparison

        # 處理 inf 值
        def safe_format(value: float, format_spec: str) -> str:
            if np.isinf(value):
                return "∞" if value > 0 else "-∞"
            if np.isnan(value):
                return "N/A"
            return format(value, format_spec)

        report = f"""
=====================================
       樣本外驗證報告
=====================================

分割日期: {result.split_date}
過擬合判定: {'⚠️ 是' if result.is_overfitted else '✅ 否'}

【樣本內績效】({is_metrics['trading_days']} 天)
  總報酬率:      {is_metrics['total_return']:>10.2%}
  年化報酬率:    {is_metrics['annual_return']:>10.2%}
  最大回撤:      {is_metrics['max_drawdown']:>10.2%}
  夏普比率:      {safe_format(is_metrics['sharpe_ratio'], '>10.2f')}
  勝率:          {is_metrics['win_rate']:>10.1%}

【樣本外績效】({oos_metrics['trading_days']} 天)
  總報酬率:      {oos_metrics['total_return']:>10.2%}
  年化報酬率:    {oos_metrics['annual_return']:>10.2%}
  最大回撤:      {oos_metrics['max_drawdown']:>10.2%}
  夏普比率:      {safe_format(oos_metrics['sharpe_ratio'], '>10.2f')}
  勝率:          {oos_metrics['win_rate']:>10.1%}

【績效比較】(樣本外 / 樣本內)
  報酬比率:      {safe_format(comp['performance_ratio'].get('total_return', 0), '>10.2f')}
  夏普比率比:    {safe_format(comp['performance_ratio'].get('sharpe_ratio', 0), '>10.2f')}
  一致性分數:    {comp['consistency_score']:>10.2f}

【過擬合診斷】
  夏普比率衰退:  {'是' if comp['performance_ratio'].get('sharpe_ratio', 1) < self.overfitting_threshold else '否'}
  報酬率衰退:    {'是' if comp['performance_ratio'].get('total_return', 1) < self.overfitting_threshold else '否'}
  一致性不足:    {'是' if comp['consistency_score'] < 0.3 else '否'}

=====================================
"""
        return report


def quick_validate(
    data: pd.DataFrame,
    split_date: str = DEFAULT_SPLIT_DATE,
    target_ratio: float = 0.6,
    pid_params: Optional[dict] = None
) -> ValidationResult:
    """快速執行樣本外驗證（便捷函數）

    Args:
        data: 價格資料
        split_date: 分割日期
        target_ratio: 目標股票比例
        pid_params: PID 參數

    Returns:
        ValidationResult: 驗證結果

    Example:
        >>> from data.data_loader import quick_load
        >>> data = quick_load()
        >>> result = quick_validate(data, "2023-01-01")
        >>> print(f"過擬合: {result.is_overfitted}")
    """
    if pid_params is None:
        pid_params = {"kp": 1.0, "ki": 0.1, "kd": 2.0}

    engine = BacktestEngine(
        target_ratio=target_ratio,
        pid_params=pid_params
    )

    validator = OutOfSampleValidator()
    return validator.validate(engine, data, split_date)
