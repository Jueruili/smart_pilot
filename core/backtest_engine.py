"""
回測引擎模組

執行投資組合再平衡策略的歷史回測。
使用 PD 控制器 + Log-Space 卡爾曼濾波器。

核心功能：
- 使用 KF 濾波價格雜訊，取得乾淨速度信號
- 使用 PD 控制器計算再平衡調整量
- 模擬交易執行（含手續費）
- 記錄完整的回測歷史（含 KF 速度、PD 控制訊號）

作者：Smart Pilot Team
版本：3.0.0
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.kalman_filter import LogKalmanFilter
from core.pd_controller import PDController
from core.benchmark import get_metrics, calc_rmse


# 預設參數
DEFAULT_TARGET_RATIO: float = 0.6
DEFAULT_KF_Q: float = 0.001
DEFAULT_KP: float = 0.5
DEFAULT_KD: float = 0.5
DEFAULT_DEADBAND: float = 0.0125
DEFAULT_FEE_RATE: float = 0.003
DEFAULT_WARMUP: int = 30
DEFAULT_STOCK_TICKER: str = "VTI"
DEFAULT_BOND_TICKER: str = "BND"


class BacktestEngine:
    """回測引擎（v3.0 — KF + PD 架構）

    使用 Log-Space 卡爾曼濾波器 + PD 控制器執行投資組合再平衡。

    Attributes:
        target_ratio: 目標股票比例
        kf_q: 卡爾曼濾波器 Q 值
        kp: PD 控制器 Kp
        kd: PD 控制器 Kd
        deadband: 死區閾值
        fee_rate: 手續費率
        warmup: KF 暖機天數
        stock_ticker: 股票代碼
        bond_ticker: 債券代碼
    """

    def __init__(
        self,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        kf_q: float = DEFAULT_KF_Q,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        deadband: float = DEFAULT_DEADBAND,
        fee_rate: float = DEFAULT_FEE_RATE,
        warmup: int = DEFAULT_WARMUP,
        stock_ticker: str = DEFAULT_STOCK_TICKER,
        bond_ticker: str = DEFAULT_BOND_TICKER,
    ) -> None:
        if not 0 <= target_ratio <= 1:
            raise ValueError(f"目標比例必須在 0-1 之間，收到：{target_ratio}")
        if deadband < 0:
            raise ValueError(f"死區閾值不能為負數，收到：{deadband}")
        if fee_rate < 0:
            raise ValueError(f"手續費率不能為負數，收到：{fee_rate}")

        self.target_ratio = target_ratio
        self.kf_q = kf_q
        self.kp = kp
        self.kd = kd
        self.deadband = deadband
        self.fee_rate = fee_rate
        self.warmup = warmup
        self.stock_ticker = stock_ticker
        self.bond_ticker = bond_ticker

    def run(self, data: pd.DataFrame) -> dict:
        """執行回測

        使用 KF + PD 架構：
        1. 前 warmup 天暖機 KF
        2. 從第 warmup 天開始正式回測
        3. 每天更新 KF，取得速度信號
        4. PD 控制器計算控制量 u
        5. |u| > deadband 才執行交易

        Args:
            data: 價格資料 DataFrame
                - index: 日期
                - columns: [stock_ticker, bond_ticker]

        Returns:
            dict: {
                "nav_list": list,
                "weights": list,
                "trade_count": int,
                "turnover": float,
                "actions": list,
                "rmse": float,
                "cost": float,
                "metrics": dict,
                "kf_stock_velocities": list,
                "kf_bond_velocities": list,
                "p_terms": list,
                "d_terms": list,
                "u_values": list,
                "dates": list,
            }
        """
        if data.empty:
            raise ValueError("輸入資料為空")

        # 確保欄位順序
        stock_col = self.stock_ticker
        bond_col = self.bond_ticker
        if stock_col not in data.columns or bond_col not in data.columns:
            # 如果指定欄位不存在，用前兩個欄位
            stock_col = data.columns[0]
            bond_col = data.columns[1]

        prices_stock = data[stock_col].values.astype(np.float64)
        prices_bond = data[bond_col].values.astype(np.float64)
        dates = data.index.tolist()
        n = len(prices_stock)

        # 計算日報酬率
        rets_stock = np.zeros(n, dtype=np.float64)
        rets_bond = np.zeros(n, dtype=np.float64)
        rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
        rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

        # 初始化 KF（暖機）
        kf_stock = LogKalmanFilter(np.log(prices_stock[0]), q=self.kf_q)
        kf_bond = LogKalmanFilter(np.log(prices_bond[0]), q=self.kf_q)
        for i in range(self.warmup):
            kf_stock.predict()
            kf_stock.update(np.log(prices_stock[i]))
            kf_bond.predict()
            kf_bond.update(np.log(prices_bond[i]))

        # 初始化 PD 控制器
        pd_ctrl = PDController(kp=self.kp, kd=self.kd)

        # 回測狀態
        wealth = 1.0
        p_stock = self.target_ratio
        turnover = 0.0
        trade_count = 0

        # 記錄列表
        nav_list = [1.0]
        weights = []
        actions = []
        kf_stock_velocities = []
        kf_bond_velocities = []
        p_terms = []
        d_terms = []
        u_values = []
        backtest_dates = []

        for i in range(self.warmup, n):
            # 更新 KF
            kf_stock.predict()
            kf_stock.update(np.log(prices_stock[i]))
            kf_bond.predict()
            kf_bond.update(np.log(prices_bond[i]))

            vel_stock = kf_stock.velocity
            vel_bond = kf_bond.velocity

            # 更新資產價值
            val_stock = p_stock * wealth * (1.0 + rets_stock[i])
            val_bond = (1.0 - p_stock) * wealth * (1.0 + rets_bond[i])
            total = val_stock + val_bond
            curr_w = val_stock / total

            # PD 控制
            error = self.target_ratio - curr_w
            p_term = pd_ctrl.kp * error
            d_term = float(np.clip(pd_ctrl.kd * (vel_stock - vel_bond),
                                   -pd_ctrl.d_clip, pd_ctrl.d_clip))
            u = pd_ctrl.calculate(error, vel_stock, vel_bond)

            # 記錄控制訊號
            kf_stock_velocities.append(vel_stock)
            kf_bond_velocities.append(vel_bond)
            p_terms.append(p_term)
            d_terms.append(d_term)
            u_values.append(u)
            backtest_dates.append(dates[i])

            if abs(u) > self.deadband:
                trade_pct = abs(u)
                cost = trade_pct * total * self.fee_rate
                total -= cost
                turnover += trade_pct
                trade_count += 1
                new_w = np.clip(curr_w + u, 0.0, 1.0)
                p_stock = new_w
                actions.append(u)
            else:
                p_stock = curr_w
                actions.append(0.0)

            wealth = total
            nav_list.append(wealth)
            weights.append(curr_w)

        rmse = calc_rmse(weights, self.target_ratio)
        cost = turnover * self.fee_rate
        metrics = get_metrics(nav_list)

        return {
            "nav_list": nav_list,
            "weights": weights,
            "trade_count": trade_count,
            "turnover": turnover,
            "actions": actions,
            "rmse": rmse,
            "cost": cost,
            "metrics": metrics,
            "kf_stock_velocities": kf_stock_velocities,
            "kf_bond_velocities": kf_bond_velocities,
            "p_terms": p_terms,
            "d_terms": d_terms,
            "u_values": u_values,
            "dates": backtest_dates,
        }

    def reset(self) -> None:
        """重置回測引擎（預留介面）"""
        pass
