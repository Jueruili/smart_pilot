#!/usr/bin/env python3
"""
Grid Search 預計算腳本

執行參數網格搜索並將結果儲存為 JSON。
供 Streamlit 前端讀取，避免重複計算。

用法：
    python scripts/run_grid_search.py

輸出：
    data/cache/grid_search_results.json
"""

import json
import os
import sys
import numpy as np

# 確保專案根目錄在 Python 路徑中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from core.optimizer import run_grid_search, find_best_from_grid


# 設定
CACHE_PATH = "data/cache/grid_search_results.json"
STOCK_TICKER = "VTI"
BOND_TICKER = "BND"
START_DATE = "2013-01-01"
END_DATE = "2025-12-31"
TARGET_W = 0.6
FEE_RATE = 0.003

# Grid Search 網格設定
KP_RANGE = np.linspace(0.1, 1.0, 10).tolist()
KD_RANGE = np.linspace(0.1, 1.0, 10).tolist()
Q_VALUES = [0.0001, 0.001, 0.01]
DEADBAND_VALUES = np.linspace(0.005, 0.10, 15).tolist()


def main():
    print("=" * 60)
    print("Grid Search 預計算腳本")
    print("=" * 60)

    # 載入資料
    print("\n[1/3] 載入資料...")
    loader = DataLoader()
    data = loader.load_and_process([STOCK_TICKER, BOND_TICKER], START_DATE, END_DATE)
    data = data[[STOCK_TICKER, BOND_TICKER]]  # 強制排序

    prices_stock = data[STOCK_TICKER].values
    prices_bond = data[BOND_TICKER].values
    dates = data.index.tolist()

    # 計算日報酬率
    rets_stock = np.zeros(len(prices_stock))
    rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
    rets_bond = np.zeros(len(prices_bond))
    rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

    print(f"   資料期間: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"   總天數: {len(dates)}")

    # 執行 Grid Search
    total_combinations = len(KP_RANGE) * len(KD_RANGE) * len(Q_VALUES)
    print(f"\n[2/3] 執行 Grid Search ({total_combinations} 組合)...")
    print(f"   Kp: {len(KP_RANGE)} 個點")
    print(f"   Kd: {len(KD_RANGE)} 個點")
    print(f"   Q: {len(Q_VALUES)} 個值")
    print(f"   Deadband: {len(DEADBAND_VALUES)} 個點")
    print("   使用 joblib 平行化，請稍候...")

    grid_results = run_grid_search(
        rets_stock=rets_stock,
        rets_bond=rets_bond,
        prices_stock=prices_stock,
        prices_bond=prices_bond,
        dates=dates,
        target_w=TARGET_W,
        fee_rate=FEE_RATE,
        kp_range=KP_RANGE,
        kd_range=KD_RANGE,
        q_values=Q_VALUES,
        deadband_values=DEADBAND_VALUES,
        n_jobs=-1,
    )

    # 找最佳參數
    best = find_best_from_grid(grid_results)
    print(f"\n   最佳參數:")
    print(f"     Kp = {best['kp']:.3f}")
    print(f"     Kd = {best['kd']:.3f}")
    print(f"     Q = {best['q']:.6f}")
    print(f"     Hypervolume = {best['hypervolume']:.6f}")

    # 儲存結果
    print(f"\n[3/3] 儲存結果至 {CACHE_PATH}...")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    output = {
        "metadata": {
            "stock_ticker": STOCK_TICKER,
            "bond_ticker": BOND_TICKER,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "target_w": TARGET_W,
            "fee_rate": FEE_RATE,
            "total_days": len(dates),
            "kp_range": KP_RANGE,
            "kd_range": KD_RANGE,
            "q_values": Q_VALUES,
            "deadband_values": DEADBAND_VALUES,
        },
        "results": grid_results,
        "best": best,
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"   儲存完成！({len(grid_results)} 筆結果)")
    print("\n" + "=" * 60)
    print("Grid Search 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
