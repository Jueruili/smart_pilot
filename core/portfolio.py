"""
投資組合管理模組

管理投資組合的持倉、權重計算和再平衡操作。

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


@dataclass
class Position:
    """持倉資訊

    Attributes:
        symbol: 資產代碼
        shares: 持有股數
        avg_cost: 平均成本
        current_price: 當前價格
    """
    symbol: str
    shares: float
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """計算市值"""
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """計算未實現損益"""
        return self.shares * (self.current_price - self.avg_cost)

    @property
    def unrealized_pnl_pct(self) -> float:
        """計算未實現損益百分比"""
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class Portfolio:
    """投資組合

    管理多個資產的持倉和權重。

    Attributes:
        positions: 持倉字典，key 為資產代碼
        cash: 現金餘額
        target_weights: 目標權重

    Example:
        >>> portfolio = Portfolio(cash=100000)
        >>> portfolio.add_position("AAPL", shares=100, avg_cost=150.0)
        >>> portfolio.update_prices({"AAPL": 160.0})
        >>> print(portfolio.total_value)
    """
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    target_weights: dict[str, float] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """計算投資組合總價值（含現金）"""
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return positions_value + self.cash

    def add_position(
        self,
        symbol: str,
        shares: float,
        avg_cost: float,
        current_price: Optional[float] = None
    ) -> None:
        """新增或更新持倉

        Args:
            symbol: 資產代碼
            shares: 股數
            avg_cost: 平均成本
            current_price: 當前價格（可選）

        Raises:
            ValueError: 當股數或成本為負數時拋出
        """
        if shares < 0:
            raise ValueError(f"股數不能為負數，收到：{shares}")
        if avg_cost < 0:
            raise ValueError(f"平均成本不能為負數，收到：{avg_cost}")

        price = current_price if current_price is not None else avg_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            avg_cost=avg_cost,
            current_price=price
        )

    def remove_position(self, symbol: str) -> Optional[Position]:
        """移除持倉

        Args:
            symbol: 資產代碼

        Returns:
            被移除的持倉，如果不存在則返回 None
        """
        return self.positions.pop(symbol, None)

    def update_prices(self, prices: dict[str, float]) -> None:
        """更新資產價格

        Args:
            prices: 價格字典，key 為資產代碼，value 為價格
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def get_current_weights(self) -> dict[str, float]:
        """計算當前各資產權重

        Returns:
            dict: 當前權重，包含各資產和現金

        Note:
            權重 = 市值 / 總價值
            現金也會包含在權重計算中
        """
        total = self.total_value
        if total == 0:
            return {}

        weights = {}
        for symbol, pos in self.positions.items():
            weights[symbol] = pos.market_value / total

        # 現金權重
        weights["_CASH"] = self.cash / total

        return weights

    def get_weight_deviations(self) -> dict[str, float]:
        """計算權重偏離（誤差）

        Returns:
            dict: 各資產的權重偏離值
                正值表示目標權重 > 當前權重（需要買入）
                負值表示目標權重 < 當前權重（需要賣出）
        """
        current_weights = self.get_current_weights()
        deviations = {}

        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            deviations[symbol] = target_weight - current_weight

        return deviations

    def set_target_weights(self, weights: dict[str, float]) -> None:
        """設定目標權重

        Args:
            weights: 目標權重字典

        Raises:
            ValueError: 當權重總和超過 1 或有負數權重時拋出
        """
        if any(w < 0 for w in weights.values()):
            raise ValueError("權重不能為負數")

        weight_sum = sum(weights.values())
        if weight_sum > 1.0 + 1e-6:
            raise ValueError(f"權重總和不能超過 1，收到：{weight_sum}")

        self.target_weights = weights.copy()

    def calculate_rebalance_orders(
        self,
        adjustment_ratios: dict[str, float]
    ) -> list[dict]:
        """根據調整比例計算再平衡訂單

        Args:
            adjustment_ratios: 各資產的調整比例（來自 PID 控制器）

        Returns:
            list[dict]: 訂單列表，每個訂單包含 symbol, action, shares, amount

        TODO:
            - 實作完整的訂單計算邏輯
            - 考慮最小交易單位
            - 考慮交易成本
        """
        # TODO: 實作訂單計算邏輯
        raise NotImplementedError("訂單計算尚未實作")

    def to_dataframe(self) -> pd.DataFrame:
        """將投資組合轉換為 DataFrame

        Returns:
            pd.DataFrame: 包含所有持倉資訊的 DataFrame
        """
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, pos in self.positions.items():
            current_weight = pos.market_value / self.total_value if self.total_value > 0 else 0
            target_weight = self.target_weights.get(symbol, 0.0)

            data.append({
                "symbol": symbol,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "weight_deviation": target_weight - current_weight,
            })

        return pd.DataFrame(data)

    def __repr__(self) -> str:
        """返回投資組合的字串表示"""
        return (
            f"Portfolio(total_value={self.total_value:,.2f}, "
            f"cash={self.cash:,.2f}, "
            f"positions={len(self.positions)})"
        )
