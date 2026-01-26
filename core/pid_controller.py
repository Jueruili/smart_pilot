"""
PID 控制器模組

實作增量型 PID 控制器，用於投資組合再平衡系統。
增量型 PID 相較於位置型 PID 的優點：
1. 無積分飽和問題
2. 輸出為調整量，更適合金融應用
3. 運算更穩定，不易產生大幅突變

公式：Δu(k) = Kp[e(k) - e(k-1)] + Ki*e(k) + Kd[e(k) - 2e(k-1) + e(k-2)]

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional


class IncrementalPID:
    """增量型 PID 控制器

    增量型 PID 控制器計算的是控制量的增量 Δu(k)，而非控制量本身。
    這種方式特別適合投資組合再平衡，因為我們關心的是「需要調整多少」。

    公式：
        Δu(k) = Kp * [e(k) - e(k-1)] + Ki * e(k) + Kd * [e(k) - 2*e(k-1) + e(k-2)]

        其中：
        - e(k): 當前誤差（目標權重 - 實際權重）
        - e(k-1): 前一次誤差
        - e(k-2): 前兩次誤差
        - Kp: 比例增益，響應誤差變化速度
        - Ki: 積分增益，消除穩態誤差
        - Kd: 微分增益，抑制誤差變化，增加穩定性

    Attributes:
        kp (float): 比例增益係數
        ki (float): 積分增益係數
        kd (float): 微分增益係數
        e_k (float): 當前誤差 e(k)
        e_k1 (float): 前一次誤差 e(k-1)
        e_k2 (float): 前兩次誤差 e(k-2)

    Example:
        >>> pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
        >>> error = 0.05  # 目標權重與實際權重差距 5%
        >>> adjustment = pid.calculate(error)
        >>> print(f"建議調整量: {adjustment:.4f}")
    """

    def __init__(self, kp: float, ki: float, kd: float) -> None:
        """初始化增量型 PID 控制器

        Args:
            kp: 比例增益係數 (Proportional gain)
                - 較大值：響應更快，但可能造成過度調整
                - 較小值：響應較慢，但更穩定
                - 建議範圍：0.1 ~ 1.0
            ki: 積分增益係數 (Integral gain)
                - 用於消除穩態誤差
                - 較大值：消除誤差更快，但可能造成振盪
                - 建議範圍：0.01 ~ 0.5
            kd: 微分增益係數 (Derivative gain)
                - 用於抑制誤差變化，增加系統穩定性
                - 較大值：抑制效果更強，但可能響應過慢
                - 建議範圍：0.01 ~ 0.2

        Raises:
            ValueError: 當任何增益係數為負數時拋出

        Example:
            >>> pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
        """
        # 驗證參數有效性
        if kp < 0 or ki < 0 or kd < 0:
            raise ValueError(
                f"PID 增益係數必須為非負數，收到：kp={kp}, ki={ki}, kd={kd}"
            )

        # 儲存 PID 參數
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd

        # 初始化誤差歷史狀態
        # e_k: 當前誤差, e_k1: 前一次誤差, e_k2: 前兩次誤差
        self.e_k: float = 0.0
        self.e_k1: float = 0.0
        self.e_k2: float = 0.0

    def calculate(self, error: float) -> float:
        """計算 PID 調整量

        根據增量型 PID 公式計算本次應調整的量。

        公式：Δu(k) = Kp * [e(k) - e(k-1)] + Ki * e(k) + Kd * [e(k) - 2*e(k-1) + e(k-2)]

        Args:
            error: 當前誤差值 e(k)
                - 正值表示實際權重低於目標，需要買入
                - 負值表示實際權重高於目標，需要賣出

        Returns:
            float: 調整量 Δu(k)
                - 正值表示應增加持倉
                - 負值表示應減少持倉

        Example:
            >>> pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
            >>> # 第一次計算：誤差 5%
            >>> delta1 = pid.calculate(0.05)
            >>> # 第二次計算：誤差減少到 3%
            >>> delta2 = pid.calculate(0.03)

        Note:
            每次呼叫此方法後，內部誤差歷史會自動更新：
            - e_k2 <- e_k1 (前兩次誤差更新為前一次誤差)
            - e_k1 <- e_k (前一次誤差更新為當前誤差)
            - e_k <- error (當前誤差更新為新輸入)
        """
        # 更新誤差歷史（移位操作）
        self.e_k2 = self.e_k1  # 前兩次誤差 = 舊的前一次誤差
        self.e_k1 = self.e_k    # 前一次誤差 = 舊的當前誤差
        self.e_k = error        # 當前誤差 = 新輸入的誤差

        # 計算增量型 PID 各項
        # 比例項：Kp * [e(k) - e(k-1)]
        # 反映誤差的變化速度
        proportional_term = self.kp * (self.e_k - self.e_k1)

        # 積分項：Ki * e(k)
        # 用於消除穩態誤差
        integral_term = self.ki * self.e_k

        # 微分項：Kd * [e(k) - 2*e(k-1) + e(k-2)]
        # 這是二階差分，用於預測誤差變化趨勢
        derivative_term = self.kd * (self.e_k - 2 * self.e_k1 + self.e_k2)

        # 計算總調整量
        delta_u = proportional_term + integral_term + derivative_term

        return delta_u

    def reset(self) -> None:
        """重置 PID 控制器的歷史誤差狀態

        將所有誤差歷史歸零，適用於以下情況：
        1. 開始新的交易週期
        2. 投資組合發生重大變化
        3. 系統重新初始化

        Note:
            此方法不會重置 PID 參數（kp, ki, kd），
            只會重置誤差歷史狀態。

        Example:
            >>> pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
            >>> pid.calculate(0.05)
            >>> pid.calculate(0.03)
            >>> pid.reset()  # 重置歷史
            >>> # 現在 e_k, e_k1, e_k2 都是 0.0
        """
        self.e_k = 0.0
        self.e_k1 = 0.0
        self.e_k2 = 0.0

    def get_state(self) -> dict[str, float]:
        """取得當前 PID 控制器的完整狀態

        Returns:
            dict: 包含所有參數和狀態的字典
                - kp, ki, kd: PID 參數
                - e_k, e_k1, e_k2: 誤差歷史

        Example:
            >>> pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
            >>> state = pid.get_state()
            >>> print(state)
            {'kp': 0.5, 'ki': 0.1, 'kd': 0.05, 'e_k': 0.0, 'e_k1': 0.0, 'e_k2': 0.0}
        """
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "e_k": self.e_k,
            "e_k1": self.e_k1,
            "e_k2": self.e_k2,
        }

    def set_parameters(self, kp: Optional[float] = None,
                       ki: Optional[float] = None,
                       kd: Optional[float] = None) -> None:
        """動態調整 PID 參數

        允許在運行時調整 PID 參數，不影響誤差歷史。
        只有傳入的參數會被更新。

        Args:
            kp: 新的比例增益係數（可選）
            ki: 新的積分增益係數（可選）
            kd: 新的微分增益係數（可選）

        Raises:
            ValueError: 當任何傳入的增益係數為負數時拋出

        Example:
            >>> pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
            >>> pid.set_parameters(kp=0.3)  # 只更新 kp
            >>> pid.set_parameters(ki=0.2, kd=0.1)  # 更新 ki 和 kd
        """
        if kp is not None:
            if kp < 0:
                raise ValueError(f"kp 必須為非負數，收到：{kp}")
            self.kp = kp

        if ki is not None:
            if ki < 0:
                raise ValueError(f"ki 必須為非負數，收到：{ki}")
            self.ki = ki

        if kd is not None:
            if kd < 0:
                raise ValueError(f"kd 必須為非負數，收到：{kd}")
            self.kd = kd

    def __repr__(self) -> str:
        """返回 PID 控制器的字串表示

        Returns:
            str: 格式化的控制器資訊
        """
        return (
            f"IncrementalPID(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"e_k={self.e_k:.6f}, e_k1={self.e_k1:.6f}, e_k2={self.e_k2:.6f})"
        )
