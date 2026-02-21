import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.data_loader import DataLoader
from core.kalman_filter import LogKalmanFilter

PERIODS = [
    ("2008金融海嘯", "2008-01-01", "2009-06-01"),
    ("2020新冠崩盤", "2020-01-01", "2020-06-01"),
    ("2022升息崩盤", "2022-01-01", "2022-12-01"),
]

loader = DataLoader()

for name, start, end in PERIODS:
    print(f"\n{'='*60}")
    print(f"  {name}  ({start} ~ {end})")
    print(f"{'='*60}")

    df = loader.load_and_process(["VTI", "BND"], start_date=start, end_date=end)
    df = df[["VTI", "BND"]]

    prices_stock = df["VTI"].values
    prices_bond  = df["BND"].values
    dates        = df.index.tolist()

    kf_stock = LogKalmanFilter(np.log(prices_stock[0]), q=0.001)
    kf_bond  = LogKalmanFilter(np.log(prices_bond[0]),  q=0.001)

    for i in range(30):
        kf_stock.predict()
        kf_stock.update(np.log(prices_stock[i]))
        kf_bond.predict()
        kf_bond.update(np.log(prices_bond[i]))

    print(f"{'日期':<12} {'VTI速度':>10} {'BND速度':>10} {'速度差':>10} {'D項(Kd=0.5)':>12}")
    print("-" * 58)

    min_diff = 0
    min_date = ""

    for i in range(30, len(prices_stock)):
        kf_stock.predict()
        kf_stock.update(np.log(prices_stock[i]))
        kf_bond.predict()
        kf_bond.update(np.log(prices_bond[i]))

        vel_s  = kf_stock.velocity
        vel_b  = kf_bond.velocity
        diff   = vel_s - vel_b
        d_term = 0.5 * diff

        if diff < min_diff:
            min_diff = diff
            min_date = dates[i].strftime("%Y-%m-%d")

        # 只印速度差小於 -0.005 的天（崩盤信號明顯的日子）
        if diff < -0.005:
            date_str = dates[i].strftime("%Y-%m-%d")
            print(f"{date_str:<12} {vel_s:>10.5f} {vel_b:>10.5f} {diff:>10.5f} {d_term:>12.5f}")

    print(f"\n>>> 最大煞車日：{min_date}，速度差 = {min_diff:.5f}，D項 = {min_diff*0.5:.5f}")