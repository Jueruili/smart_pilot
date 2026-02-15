"""
PD 控制器模組

P 項：Kp * (target_weight - current_weight)
D 項：clip(Kd * (velocity_stock - velocity_bond), -0.15, 0.15)
總輸出：clip(P + D, -0.2, 0.2)

作者：Smart Pilot Team
版本：2.0.0
"""

import numpy as np
from typing import Optional


class PDController:
    def __init__(self, kp: float, kd: float) -> None:
        if kp < 0 or kd < 0:
            raise ValueError(f"增益係數必須為非負數，收到：kp={kp}, kd={kd}")
        self.kp = kp
        self.kd = kd
        self.d_clip = 0.15
        self.output_clip = 0.2

    def calculate(self, error: float, vel_stock: float, vel_bond: float) -> float:
        p_term = self.kp * error
        d_term = np.clip(self.kd * (vel_stock - vel_bond), -self.d_clip, self.d_clip)
        u = np.clip(p_term + d_term, -self.output_clip, self.output_clip)
        return u

    def reset(self) -> None:
        pass

    def set_parameters(self, kp: Optional[float] = None, kd: Optional[float] = None) -> None:
        if kp is not None:
            if kp < 0: raise ValueError(f"kp 必須非負")
            self.kp = kp
        if kd is not None:
            if kd < 0: raise ValueError(f"kd 必須非負")
            self.kd = kd

    def get_state(self) -> dict:
        return {"kp": self.kp, "kd": self.kd}
