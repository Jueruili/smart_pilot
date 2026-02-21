#!/usr/bin/env python3
"""
SLSQP 精確最佳化預計算腳本

基於 Grid Search 最佳結果，對每個 deadband 執行 SLSQP 最佳化。
結果儲存為 JSON 供 Streamlit 前端讀取。

用法：
    python scripts/run_slsqp.py

輸入：
    data/cache/grid_search_results.json（需先執行 run_grid_search.py）

輸出：
    data/cache/slsqp_results.json
"""

import json
import os
import sys
import numpy as np

# 確保專案根目錄在 Python 路徑中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from core.optimizer import run_slsqp


# 設定
GRID_CACHE_PATH = "data/cache/grid_search_results.json"
SLSQP_CACHE_PATH = "data/cache/slsqp_results.json"
STOCK_TICKER = "VTI"
BOND_TICKER = "BND"
START_DATE = "2013-01-01"
END_DATE = "2025-12-31"
TARGET_W = 0.6
FEE_RATE = 0.003

# SLSQP deadband 範圍（比 Grid Search 更密集）
DEADBAND_VALUES = np.linspace(0.005, 0.10, 20).tolist()


def main():
    print("=" * 60)
    print("SLSQP 精確最佳化預計算腳本")
    print("=" * 60)

    # 讀取 Grid Search 結果
    print("\n[1/4] 讀取 Grid Search 結果...")
    if not os.path.exists(GRID_CACHE_PATH):
        print(f"   錯誤：找不到 {GRID_CACHE_PATH}")
        print("   請先執行 python scripts/run_grid_search.py")
        sys.exit(1)

    with open(GRID_CACHE_PATH, "r", encoding="utf-8") as f:
        grid_data = json.load(f)

    best = grid_data["best"]
    print(f"   Grid Search 最佳參數:")
    print(f"     Kp = {best['kp']:.3f}")
    print(f"     Kd = {best['kd']:.3f}")
    print(f"     Q = {best['q']:.6f}")

    # 載入資料
    print("\n[2/4] 載入資料...")
    loader = DataLoader()
    data = loader.load_and_process([STOCK_TICKER, BOND_TICKER], START_DATE, END_DATE)
    data = data[[STOCK_TICKER, BOND_TICKER]]

    prices_stock = data[STOCK_TICKER].values
    prices_bond = data[BOND_TICKER].values
    dates = data.index.tolist()

    rets_stock = np.zeros(len(prices_stock))
    rets_stock[1:] = np.diff(prices_stock) / prices_stock[:-1]
    rets_bond = np.zeros(len(prices_bond))
    rets_bond[1:] = np.diff(prices_bond) / prices_bond[:-1]

    print(f"   資料期間: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")

    # 執行 SLSQP
    print(f"\n[3/4] 執行 SLSQP 最佳化 ({len(DEADBAND_VALUES)} 個 deadband)...")
    print("   每個 deadband 獨立最佳化 (Kp, Kd, Q)...")

    slsqp_results = run_slsqp(
        rets_stock=rets_stock,
        rets_bond=rets_bond,
        prices_stock=prices_stock,
        prices_bond=prices_bond,
        dates=dates,
        target_w=TARGET_W,
        fee_rate=FEE_RATE,
        deadband_values=DEADBAND_VALUES,
        initial_params={
            "kp": best["kp"],
            "kd": best["kd"],
            "q": best["q"],
        },
    )

    # 顯示結果摘要
    print("\n   SLSQP 結果摘要:")
    print("   " + "-" * 50)
    print(f"   {'Deadband':>10} | {'Kp':>6} | {'Kd':>6} | {'Q':>10} | {'RMSE':>8} | {'AnnCost':>8}")
    print("   " + "-" * 50)
    for r in slsqp_results[:5]:  # 只顯示前 5 筆
        print(f"   {r['deadband']:>10.4f} | {r['kp']:>6.3f} | {r['kd']:>6.3f} | {r['q']:>10.6f} | {r['rmse']:>8.4f} | {r['ann_cost']:>8.4f}")
    if len(slsqp_results) > 5:
        print(f"   ... 共 {len(slsqp_results)} 筆結果")

    # 儲存結果
    print(f"\n[4/4] 儲存結果至 {SLSQP_CACHE_PATH}...")
    os.makedirs(os.path.dirname(SLSQP_CACHE_PATH), exist_ok=True)

    output = {
        "metadata": {
            "stock_ticker": STOCK_TICKER,
            "bond_ticker": BOND_TICKER,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "target_w": TARGET_W,
            "fee_rate": FEE_RATE,
            "total_days": len(dates),
            "deadband_values": DEADBAND_VALUES,
            "initial_params": {
                "kp": best["kp"],
                "kd": best["kd"],
                "q": best["q"],
            },
        },
        "results": slsqp_results,
    }

    with open(SLSQP_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"   儲存完成！")
    print("\n" + "=" * 60)
    print("SLSQP 最佳化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
