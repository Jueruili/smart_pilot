"""
回測引擎模組

執行投資組合再平衡策略的歷史回測。

核心功能：
- 使用 PID 控制器計算再平衡調整量
- 模擬交易執行（含手續費）
- 記錄完整的回測歷史

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from core.pid_controller import IncrementalPID


# 預設參數
DEFAULT_INITIAL_CASH: float = 1_000_000.0
DEFAULT_TARGET_RATIO: float = 0.6
DEFAULT_PID_PARAMS: dict = {"kp": 0.3, "ki": 0.05, "kd": 0.1}
DEFAULT_DEADBAND: float = 0.02
DEFAULT_COMMISSION_RATE: float = 0.001
DEFAULT_EXCHANGE_RATE: float = 30.5


@dataclass
class BacktestResult:
    """回測結果

    Attributes:
        history: 每日記錄的 DataFrame
        total_return: 總報酬率
        annualized_return: 年化報酬率
        max_drawdown: 最大回撤
        sharpe_ratio: 夏普比率
        total_trades: 總交易次數
        total_commission: 總手續費
    """
    history: pd.DataFrame
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    total_commission: float = 0.0


class BacktestEngine:
    """回測引擎

    使用 PID 控制器執行投資組合再平衡策略的回測。

    Attributes:
        initial_cash: 初始資金
        target_ratio: 目標股票比例
        pid: PID 控制器實例
        deadband: 死區閾值
        commission_rate: 手續費率
        exchange_rate: 匯率

    Example:
        >>> from data.data_loader import DataLoader
        >>> loader = DataLoader()
        >>> data = loader.load_and_process(["SPY", "TLT"], "2020-01-01", "2023-12-31")
        >>>
        >>> engine = BacktestEngine(
        ...     initial_cash=1_000_000,
        ...     target_ratio=0.6,
        ...     pid_params={"kp": 1.0, "ki": 0.1, "kd": 2.0}
        ... )
        >>> result = engine.run(data)
        >>> print(f"總報酬率: {result.total_return:.2%}")
    """

    def __init__(
        self,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        pid_params: Optional[dict] = None,
        deadband: float = DEFAULT_DEADBAND,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        exchange_rate: float = DEFAULT_EXCHANGE_RATE
    ) -> None:
        """初始化回測引擎

        Args:
            initial_cash: 初始資金，預設為 1,000,000
            target_ratio: 目標股票比例（0-1），預設為 0.6（60% 股票）
            pid_params: PID 參數字典，預設為 {"kp": 1.0, "ki": 0.1, "kd": 2.0}
            deadband: 死區閾值，調整量小於此值不執行交易，預設為 0.01
            commission_rate: 手續費率，預設為 0.001（0.1%）
            exchange_rate: 匯率（TWD/USD），預設為 30.5

        Raises:
            ValueError: 當參數不合法時拋出
        """
        # 驗證參數
        if initial_cash <= 0:
            raise ValueError(f"初始資金必須為正數，收到：{initial_cash}")
        if not 0 <= target_ratio <= 1:
            raise ValueError(f"目標比例必須在 0-1 之間，收到：{target_ratio}")
        if deadband < 0:
            raise ValueError(f"死區閾值不能為負數，收到：{deadband}")
        if commission_rate < 0:
            raise ValueError(f"手續費率不能為負數，收到：{commission_rate}")

        self.initial_cash = initial_cash
        self.target_ratio = target_ratio
        self.deadband = deadband
        self.commission_rate = commission_rate
        self.exchange_rate = exchange_rate

        # 初始化 PID 控制器
        if pid_params is None:
            pid_params = DEFAULT_PID_PARAMS.copy()

        self.pid = IncrementalPID(
            kp=pid_params.get("kp", 1.0),
            ki=pid_params.get("ki", 0.1),
            kd=pid_params.get("kd", 2.0)
        )

        # 儲存 PID 參數供參考
        self.pid_params = pid_params

        # 交易統計
        self._total_trades = 0
        self._total_commission = 0.0

        print(f"[BacktestEngine] 初始化完成")
        print(f"[BacktestEngine] 初始資金: ${initial_cash:,.0f}")
        print(f"[BacktestEngine] 目標比例: {target_ratio:.1%} 股票 / {1-target_ratio:.1%} 債券")
        print(f"[BacktestEngine] PID 參數: kp={pid_params.get('kp')}, "
              f"ki={pid_params.get('ki')}, kd={pid_params.get('kd')}")
        print(f"[BacktestEngine] 死區閾值: {deadband:.2%}")

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """執行回測

        每日執行流程：
        1. 用「昨天」的收盤價計算持倉比例
        2. 計算誤差 = 目標比例 - 當前比例
        3. PID 計算調整量 delta_u
        4. 如果 |delta_u| > deadband，執行交易
        5. 用「今天」的收盤價成交
        6. 記錄當天狀態

        Args:
            data: 價格資料 DataFrame
                - index: 日期
                - columns: [股票代碼, 債券代碼]（例如 ["SPY", "TLT"]）

        Returns:
            BacktestResult: 回測結果

        Raises:
            ValueError: 當資料格式不正確時拋出
        """
        # 驗證資料
        if data.empty:
            raise ValueError("輸入資料為空")
        if len(data.columns) != 2:
            raise ValueError(f"需要兩個標的的資料，收到 {len(data.columns)} 個")

        # 取得標的名稱
        stock_col = data.columns[0]  # 第一個為股票
        bond_col = data.columns[1]   # 第二個為債券

        print(f"\n[BacktestEngine] 開始回測")
        print(f"[BacktestEngine] 股票標的: {stock_col}")
        print(f"[BacktestEngine] 債券標的: {bond_col}")
        print(f"[BacktestEngine] 回測期間: {data.index[0].date()} ~ {data.index[-1].date()}")
        print(f"[BacktestEngine] 總交易日: {len(data)} 天")

        # 重置 PID 控制器
        self.pid.reset()

        # 重置交易統計
        self._total_trades = 0
        self._total_commission = 0.0

        # 初始化持倉
        cash = self.initial_cash
        holdings_stock = 0.0  # 股票持股數
        holdings_bond = 0.0   # 債券持股數

        # 記錄歷史
        history_records = []

        # 遍歷每個交易日
        for i, (date, row) in enumerate(data.iterrows()):
            price_stock = row[stock_col]
            price_bond = row[bond_col]

            # === 第一天：初始化買入 ===
            if i == 0:
                # 按目標比例分配資金
                stock_allocation = self.initial_cash * self.target_ratio
                bond_allocation = self.initial_cash * (1 - self.target_ratio)

                # 計算可買入的股數（扣除手續費）
                stock_value_after_fee = stock_allocation / (1 + self.commission_rate)
                bond_value_after_fee = bond_allocation / (1 + self.commission_rate)

                holdings_stock = stock_value_after_fee / price_stock
                holdings_bond = bond_value_after_fee / price_bond

                # 手續費
                commission = (stock_allocation - stock_value_after_fee) + \
                             (bond_allocation - bond_value_after_fee)
                self._total_commission += commission
                self._total_trades += 2  # 初始買入算兩筆交易

                cash = 0.0  # 全部投入

                # 記錄第一天
                value_stock = holdings_stock * price_stock
                value_bond = holdings_bond * price_bond
                nav = value_stock + value_bond + cash
                ratio = value_stock / nav if nav > 0 else 0

                history_records.append({
                    "date": date,
                    "nav": nav,
                    "cash": cash,
                    "holdings_stock": holdings_stock,
                    "holdings_bond": holdings_bond,
                    "value_stock": value_stock,
                    "value_bond": value_bond,
                    "ratio": ratio,
                    "price_stock": price_stock,
                    "price_bond": price_bond,
                    "error": 0.0,
                    "delta_u": 0.0,
                    "trade_flag": True,
                    "commission": commission,
                })
                continue

            # === 其餘交易日 ===
            # Step 1: 用「昨天」的收盤價計算持倉比例
            prev_price_stock = data.iloc[i - 1][stock_col]
            prev_price_bond = data.iloc[i - 1][bond_col]

            prev_value_stock = holdings_stock * prev_price_stock
            prev_value_bond = holdings_bond * prev_price_bond
            prev_nav = prev_value_stock + prev_value_bond + cash

            if prev_nav > 0:
                current_ratio = prev_value_stock / prev_nav
            else:
                current_ratio = 0

            # Step 2: 計算誤差
            error = self.target_ratio - current_ratio

            # Step 3: PID 計算調整量
            delta_u = self.pid.calculate(error)
            MAX_ADJUSTMENT = 0.05  # 單次最多調整 5%
            delta_u = np.clip(delta_u, -MAX_ADJUSTMENT, MAX_ADJUSTMENT)

            # Step 4: 判斷是否執行交易
            trade_flag = abs(delta_u) > self.deadband
            commission = 0.0

            if trade_flag:
                # Step 5: 用「今天」的收盤價成交
                # 計算調整金額
                current_nav = holdings_stock * price_stock + \
                              holdings_bond * price_bond + cash
                trade_value = abs(delta_u) * current_nav

                if delta_u > 0:
                    # 買股票、賣債券
                    # 賣出債券
                    bond_shares_to_sell = trade_value / price_bond
                    bond_shares_to_sell = min(bond_shares_to_sell, holdings_bond)
                    sell_proceeds = bond_shares_to_sell * price_bond
                    sell_commission = sell_proceeds * self.commission_rate
                    net_proceeds = sell_proceeds - sell_commission

                    holdings_bond -= bond_shares_to_sell

                    # 買入股票
                    buy_value = net_proceeds / (1 + self.commission_rate)
                    buy_commission = net_proceeds - buy_value
                    stock_shares_to_buy = buy_value / price_stock

                    holdings_stock += stock_shares_to_buy

                    commission = sell_commission + buy_commission

                else:
                    # 賣股票、買債券
                    # 賣出股票
                    stock_shares_to_sell = trade_value / price_stock
                    stock_shares_to_sell = min(stock_shares_to_sell, holdings_stock)
                    sell_proceeds = stock_shares_to_sell * price_stock
                    sell_commission = sell_proceeds * self.commission_rate
                    net_proceeds = sell_proceeds - sell_commission

                    holdings_stock -= stock_shares_to_sell

                    # 買入債券
                    buy_value = net_proceeds / (1 + self.commission_rate)
                    buy_commission = net_proceeds - buy_value
                    bond_shares_to_buy = buy_value / price_bond

                    holdings_bond += bond_shares_to_buy

                    commission = sell_commission + buy_commission

                self._total_trades += 2  # 買賣各一筆
                self._total_commission += commission

            # Step 6: 記錄當天狀態
            value_stock = holdings_stock * price_stock
            value_bond = holdings_bond * price_bond
            nav = value_stock + value_bond + cash
            ratio = value_stock / nav if nav > 0 else 0

            history_records.append({
                "date": date,
                "nav": nav,
                "cash": cash,
                "holdings_stock": holdings_stock,
                "holdings_bond": holdings_bond,
                "value_stock": value_stock,
                "value_bond": value_bond,
                "ratio": ratio,
                "price_stock": price_stock,
                "price_bond": price_bond,
                "error": error,
                "delta_u": delta_u,
                "trade_flag": trade_flag,
                "commission": commission,
            })

        # 建立歷史 DataFrame
        history = pd.DataFrame(history_records)
        history.set_index("date", inplace=True)

        # 計算績效指標
        result = self._calculate_metrics(history)

        print(f"\n[BacktestEngine] 回測完成")
        print(f"[BacktestEngine] 總報酬率: {result.total_return:.2%}")
        print(f"[BacktestEngine] 年化報酬率: {result.annualized_return:.2%}")
        print(f"[BacktestEngine] 最大回撤: {result.max_drawdown:.2%}")
        print(f"[BacktestEngine] 夏普比率: {result.sharpe_ratio:.2f}")
        print(f"[BacktestEngine] 總交易次數: {result.total_trades}")
        print(f"[BacktestEngine] 總手續費: ${result.total_commission:,.2f}")

        return result

    def _calculate_metrics(self, history: pd.DataFrame) -> BacktestResult:
        """計算績效指標

        Args:
            history: 歷史記錄 DataFrame

        Returns:
            BacktestResult: 包含績效指標的結果
        """
        # 總報酬率
        initial_nav = history["nav"].iloc[0]
        final_nav = history["nav"].iloc[-1]
        total_return = (final_nav - initial_nav) / initial_nav

        # 年化報酬率
        n_days = len(history)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        # 最大回撤
        cummax = history["nav"].cummax()
        drawdown = (history["nav"] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 夏普比率（假設無風險利率為 0）
        daily_returns = history["nav"].pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        return BacktestResult(
            history=history,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=self._total_trades,
            total_commission=self._total_commission,
        )

    def get_summary(self, result: BacktestResult) -> str:
        """生成回測摘要報告

        Args:
            result: 回測結果

        Returns:
            str: 格式化的報告文字
        """
        report = f"""
=====================================
        回測結果摘要
=====================================

【回測設定】
  初始資金:        ${self.initial_cash:>15,.0f}
  目標股票比例:    {self.target_ratio:>15.1%}
  死區閾值:        {self.deadband:>15.2%}
  手續費率:        {self.commission_rate:>15.3%}

【PID 參數】
  Kp:              {self.pid_params.get('kp', 0):>15.2f}
  Ki:              {self.pid_params.get('ki', 0):>15.2f}
  Kd:              {self.pid_params.get('kd', 0):>15.2f}

【績效指標】
  總報酬率:        {result.total_return:>15.2%}
  年化報酬率:      {result.annualized_return:>15.2%}
  最大回撤:        {result.max_drawdown:>15.2%}
  夏普比率:        {result.sharpe_ratio:>15.2f}

【交易統計】
  總交易次數:      {result.total_trades:>15,d}
  總手續費:        ${result.total_commission:>14,.2f}
  最終淨值:        ${result.history['nav'].iloc[-1]:>14,.2f}

【期間統計】
  回測天數:        {len(result.history):>15,d}
  平均股票比例:    {result.history['ratio'].mean():>15.2%}
  比例標準差:      {result.history['ratio'].std():>15.2%}

=====================================
"""
        return report

    def reset(self) -> None:
        """重置回測引擎

        重置 PID 控制器和交易統計。
        """
        self.pid.reset()
        self._total_trades = 0
        self._total_commission = 0.0
        print("[BacktestEngine] 已重置")
