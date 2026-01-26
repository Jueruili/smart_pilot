"""
PID 控制器單元測試

測試 IncrementalPID 類別的各項功能。

作者：Smart Pilot Team
版本：1.0.0
"""

import pytest
import sys
from pathlib import Path

# 將專案根目錄加入路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pid_controller import IncrementalPID


class TestIncrementalPIDInit:
    """測試 IncrementalPID 初始化"""

    def test_valid_initialization(self):
        """測試有效的初始化"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        assert pid.kp == 0.5
        assert pid.ki == 0.1
        assert pid.kd == 0.05
        assert pid.e_k == 0.0
        assert pid.e_k1 == 0.0
        assert pid.e_k2 == 0.0

    def test_zero_parameters(self):
        """測試零參數（有效）"""
        pid = IncrementalPID(kp=0, ki=0, kd=0)

        assert pid.kp == 0
        assert pid.ki == 0
        assert pid.kd == 0

    def test_negative_kp_raises_error(self):
        """測試負數 kp 應拋出錯誤"""
        with pytest.raises(ValueError) as excinfo:
            IncrementalPID(kp=-0.5, ki=0.1, kd=0.05)

        assert "非負數" in str(excinfo.value)

    def test_negative_ki_raises_error(self):
        """測試負數 ki 應拋出錯誤"""
        with pytest.raises(ValueError) as excinfo:
            IncrementalPID(kp=0.5, ki=-0.1, kd=0.05)

        assert "非負數" in str(excinfo.value)

    def test_negative_kd_raises_error(self):
        """測試負數 kd 應拋出錯誤"""
        with pytest.raises(ValueError) as excinfo:
            IncrementalPID(kp=0.5, ki=0.1, kd=-0.05)

        assert "非負數" in str(excinfo.value)


class TestIncrementalPIDCalculate:
    """測試 PID calculate 方法"""

    def test_first_calculation(self):
        """測試第一次計算"""
        pid = IncrementalPID(kp=1.0, ki=0.5, kd=0.2)
        error = 0.1

        # 第一次計算
        # Δu = Kp*(e_k - e_k1) + Ki*e_k + Kd*(e_k - 2*e_k1 + e_k2)
        # Δu = 1.0*(0.1 - 0) + 0.5*0.1 + 0.2*(0.1 - 0 + 0)
        # Δu = 0.1 + 0.05 + 0.02 = 0.17
        delta_u = pid.calculate(error)

        assert pytest.approx(delta_u, rel=1e-6) == 0.17

    def test_second_calculation(self):
        """測試第二次計算"""
        pid = IncrementalPID(kp=1.0, ki=0.5, kd=0.2)

        # 第一次
        pid.calculate(0.1)

        # 第二次計算，誤差減少
        # e_k=0.05, e_k1=0.1, e_k2=0
        # Δu = 1.0*(0.05 - 0.1) + 0.5*0.05 + 0.2*(0.05 - 0.2 + 0)
        # Δu = -0.05 + 0.025 + (-0.03) = -0.055
        delta_u = pid.calculate(0.05)

        assert pytest.approx(delta_u, rel=1e-6) == -0.055

    def test_third_calculation(self):
        """測試第三次計算（驗證誤差移位）"""
        pid = IncrementalPID(kp=1.0, ki=0.5, kd=0.2)

        pid.calculate(0.1)   # e_k=0.1, e_k1=0, e_k2=0
        pid.calculate(0.05)  # e_k=0.05, e_k1=0.1, e_k2=0

        # 第三次
        # e_k=0.02, e_k1=0.05, e_k2=0.1
        # Δu = 1.0*(0.02 - 0.05) + 0.5*0.02 + 0.2*(0.02 - 0.1 + 0.1)
        # Δu = -0.03 + 0.01 + 0.004 = -0.016
        delta_u = pid.calculate(0.02)

        assert pytest.approx(delta_u, rel=1e-6) == -0.016

    def test_error_history_update(self):
        """測試誤差歷史更新"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        pid.calculate(0.1)
        assert pid.e_k == 0.1
        assert pid.e_k1 == 0.0
        assert pid.e_k2 == 0.0

        pid.calculate(0.2)
        assert pid.e_k == 0.2
        assert pid.e_k1 == 0.1
        assert pid.e_k2 == 0.0

        pid.calculate(0.3)
        assert pid.e_k == 0.3
        assert pid.e_k1 == 0.2
        assert pid.e_k2 == 0.1

    def test_negative_error(self):
        """測試負誤差（需要賣出的情況）"""
        pid = IncrementalPID(kp=1.0, ki=0.5, kd=0.2)

        # 負誤差表示實際權重高於目標
        delta_u = pid.calculate(-0.1)

        # Δu = 1.0*(-0.1 - 0) + 0.5*(-0.1) + 0.2*(-0.1 - 0 + 0)
        # Δu = -0.1 - 0.05 - 0.02 = -0.17
        assert pytest.approx(delta_u, rel=1e-6) == -0.17

    def test_zero_error(self):
        """測試零誤差"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        # 先有一些誤差
        pid.calculate(0.1)
        pid.calculate(0.05)

        # 然後誤差變為零
        # e_k=0, e_k1=0.05, e_k2=0.1
        # Δu = 0.5*(0 - 0.05) + 0.1*0 + 0.05*(0 - 0.1 + 0.1)
        # Δu = -0.025 + 0 + 0 = -0.025
        delta_u = pid.calculate(0.0)

        assert pytest.approx(delta_u, rel=1e-6) == -0.025


class TestIncrementalPIDReset:
    """測試 PID reset 方法"""

    def test_reset_clears_history(self):
        """測試重置清除歷史"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        # 執行幾次計算
        pid.calculate(0.1)
        pid.calculate(0.2)
        pid.calculate(0.3)

        # 重置
        pid.reset()

        assert pid.e_k == 0.0
        assert pid.e_k1 == 0.0
        assert pid.e_k2 == 0.0

    def test_reset_preserves_parameters(self):
        """測試重置保留參數"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        pid.calculate(0.1)
        pid.reset()

        assert pid.kp == 0.5
        assert pid.ki == 0.1
        assert pid.kd == 0.05

    def test_calculation_after_reset(self):
        """測試重置後的計算"""
        pid = IncrementalPID(kp=1.0, ki=0.5, kd=0.2)

        # 第一輪計算
        pid.calculate(0.1)
        pid.calculate(0.2)

        # 重置
        pid.reset()

        # 重置後的計算應該和全新的 PID 一樣
        delta_u = pid.calculate(0.1)

        # 和新實例比較
        pid_new = IncrementalPID(kp=1.0, ki=0.5, kd=0.2)
        delta_u_new = pid_new.calculate(0.1)

        assert pytest.approx(delta_u, rel=1e-6) == delta_u_new


class TestIncrementalPIDSetParameters:
    """測試動態參數調整"""

    def test_set_single_parameter(self):
        """測試設定單一參數"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        pid.set_parameters(kp=0.8)

        assert pid.kp == 0.8
        assert pid.ki == 0.1  # 未改變
        assert pid.kd == 0.05  # 未改變

    def test_set_multiple_parameters(self):
        """測試設定多個參數"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        pid.set_parameters(ki=0.2, kd=0.1)

        assert pid.kp == 0.5  # 未改變
        assert pid.ki == 0.2
        assert pid.kd == 0.1

    def test_set_negative_parameter_raises_error(self):
        """測試設定負數參數應拋出錯誤"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        with pytest.raises(ValueError):
            pid.set_parameters(kp=-0.5)

    def test_set_parameters_preserves_history(self):
        """測試設定參數保留歷史"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        pid.calculate(0.1)
        pid.calculate(0.2)

        pid.set_parameters(kp=0.8)

        # 歷史應該保留
        assert pid.e_k == 0.2
        assert pid.e_k1 == 0.1
        assert pid.e_k2 == 0.0


class TestIncrementalPIDGetState:
    """測試取得狀態"""

    def test_get_state(self):
        """測試取得狀態"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)
        pid.calculate(0.1)

        state = pid.get_state()

        assert state["kp"] == 0.5
        assert state["ki"] == 0.1
        assert state["kd"] == 0.05
        assert state["e_k"] == 0.1
        assert state["e_k1"] == 0.0
        assert state["e_k2"] == 0.0


class TestIncrementalPIDRepr:
    """測試字串表示"""

    def test_repr(self):
        """測試 __repr__"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        repr_str = repr(pid)

        assert "IncrementalPID" in repr_str
        assert "kp=0.5" in repr_str
        assert "ki=0.1" in repr_str
        assert "kd=0.05" in repr_str


class TestIncrementalPIDIntegration:
    """整合測試：模擬實際再平衡場景"""

    def test_convergence_to_target(self):
        """測試收斂到目標權重"""
        pid = IncrementalPID(kp=0.5, ki=0.1, kd=0.05)

        target_weight = 0.3
        current_weight = 0.25

        # 模擬 10 次調整
        adjustments = []
        for _ in range(10):
            error = target_weight - current_weight
            delta_u = pid.calculate(error)
            adjustments.append(delta_u)

            # 模擬調整效果（假設調整量的 80% 生效）
            current_weight += delta_u * 0.8

        # 驗證最終權重接近目標
        assert abs(current_weight - target_weight) < 0.01

    def test_oscillation_damping(self):
        """測試振盪抑制"""
        pid = IncrementalPID(kp=0.3, ki=0.05, kd=0.1)

        errors = []
        current = 0.25
        target = 0.30

        # 模擬 20 次
        for _ in range(20):
            error = target - current
            errors.append(abs(error))
            delta_u = pid.calculate(error)
            current += delta_u * 0.9

        # 驗證誤差逐漸減小（振盪被抑制）
        # 後半段的平均誤差應小於前半段
        first_half_avg = sum(errors[:10]) / 10
        second_half_avg = sum(errors[10:]) / 10

        assert second_half_avg < first_half_avg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
