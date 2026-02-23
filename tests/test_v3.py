import numpy as np
from data.data_loader import DataLoader
from core.optimizer import run_slsqp, find_best_from_grid
from core.benchmark import run_smart_pilot
import json

# 載入資料
loader = DataLoader()
data = loader.load_and_process(["VTI", "BND"], "2013-01-01", "2024-01-01")
prices_stock = data["VTI"].values
prices_bond  = data["BND"].values
dates = data.index.tolist()
rets_stock = np.diff(prices_stock) / prices_stock[:-1]
rets_bond  = np.diff(prices_bond)  / prices_bond[:-1]
prices_stock = prices_stock[1:]
prices_bond  = prices_bond[1:]
dates = dates[1:]

# 載入 Grid Search 最佳起點
with open("data/cache/grid_search_results.json", "r") as f:
    grid_data = json.load(f)
best = grid_data["best"]
print(f"Grid Search 最佳起點：Kp={best['kp']}, Kd={best['kd']}, Q={best['q']}")

# 執行 SLSQP（只測一個 deadband）
test_deadband = 0.02
slsqp_results = run_slsqp(
    rets_stock, rets_bond, prices_stock, prices_bond, dates,
    deadband_values=[test_deadband],
    initial_params={"kp": best["kp"], "kd": best["kd"], "q": best["q"]},
)
slsqp_best = slsqp_results[0]
print(f"\nSLSQP 找到的最佳參數（deadband={test_deadband}）：")
print(f"  Kp={slsqp_best['kp']:.4f}, Kd={slsqp_best['kd']:.4f}, Q={slsqp_best['q']:.6f}")
print(f"  RMSE={slsqp_best['rmse']:.6f}")
print(f"  成功收斂：{slsqp_best['success']}")
print(f"  訊息：{slsqp_best['message']}")

# 和起點比較
baseline = run_smart_pilot(
    rets_stock, rets_bond, prices_stock, prices_bond, dates,
    kp=best["kp"], kd=best["kd"], kf_q=best["q"], deadband=test_deadband
)
print(f"\n起點的 RMSE：{baseline['rmse']:.6f}")
print(f"SLSQP 改善了：{(baseline['rmse'] - slsqp_best['rmse']):.6f}")

# 驗證：手動試幾個附近的點
print("\n手動驗證附近幾個點：")
for kp_test in [slsqp_best['kp'] - 0.05, slsqp_best['kp'], slsqp_best['kp'] + 0.05]:
    r = run_smart_pilot(
        rets_stock, rets_bond, prices_stock, prices_bond, dates,
        kp=kp_test, kd=slsqp_best['kd'], kf_q=slsqp_best['q'], deadband=test_deadband
    )
    marker = " ← SLSQP找到的" if kp_test == slsqp_best['kp'] else ""
    print(f"  Kp={kp_test:.4f} → RMSE={r['rmse']:.6f}{marker}")