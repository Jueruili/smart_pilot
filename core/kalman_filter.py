"""
卡爾曼濾波器模組

實作 Log-Space 卡爾曼濾波器，用於過濾資產價格雜訊。
對價格取 log 後進行濾波，提供乾淨的趨勢信號和速度估計。

作者：Smart Pilot Team
版本：2.0.0
"""

import numpy as np
from typing import Optional


class LogKalmanFilter:
    """Log-Space 卡爾曼濾波器

    對資產價格取 log 後進行濾波，輸出：
    - x[0]: 濾波後的 log 價格
    - x[1]: 濾波後的速度（log 價格的變化率）

    狀態模型：
        x(k) = F * x(k-1) + w    (慣性模型：價格 = 前一天價格 + 速度)
        z(k) = H * x(k) + v      (只能觀測價格)

    Args:
        initial_log_price: 初始 log 價格
        q: 過程雜訊 (Q 值)，控制濾波器的靈敏度
            - Q 大：更相信觀測值，反應快但雜訊多
            - Q 小：更相信預測模型，反應慢但更平滑
            - 建議範圍：0.0001 ~ 10.0（用 log scale 調整）
        r: 觀測雜訊，固定為 0.005
    """

    def __init__(self, initial_log_price: float, q: float = 0.001, r: float = 0.005) -> None:
        if q < 0:
            raise ValueError(f"Q 值必須為非負數，收到：{q}")

        self.x = np.array([float(initial_log_price), 0.0], dtype=np.float64)
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.array([[q, 0.0], [0.0, q * 0.1]], dtype=np.float64)
        self.R = np.array([[r]], dtype=np.float64)
        self.q_value = q
        self.r_value = r

    def predict(self) -> None:
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, log_price: float) -> None:
        z = np.array([float(log_price)])
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    @property
    def filtered_log_price(self) -> float:
        return self.x[0]

    @property
    def filtered_price(self) -> float:
        return np.exp(self.x[0])

    @property
    def velocity(self) -> float:
        return self.x[1]

    def reset(self, initial_log_price: float) -> None:
        self.x = np.array([float(initial_log_price), 0.0], dtype=np.float64)
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    def get_state(self) -> dict:
        return {
            "log_price": self.x[0], "velocity": self.x[1],
            "price": np.exp(self.x[0]), "q": self.q_value, "r": self.r_value,
        }
