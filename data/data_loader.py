"""
資料載入模組

使用 yfinance 下載股票歷史數據，並提供本地快取功能。

主要功能：
- 下載股價資料（已處理除權息）
- 資料處理與對齊
- CSV 快取機制

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


# 預設參數
DEFAULT_TICKERS: list[str] = ["VTI", "BND"]
DEFAULT_START_DATE: str = "2013-01-01"
DEFAULT_END_DATE: str = "2025-12-31"


class DataLoader:
    """資料載入器

    從 yfinance 下載股票收盤價數據，支援本地 CSV 快取。

    Attributes:
        cache_dir: 快取目錄路徑

    Example:
        >>> loader = DataLoader()
        >>> data = loader.download_data(["SPY", "TLT"], "2020-01-01", "2023-12-31")
        >>> processed = loader.process_data(data)
        >>> print(processed.head())
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """初始化資料載入器

        Args:
            cache_dir: 快取目錄路徑，預設為 data/cache
        """
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)

        # 確保快取目錄存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DataLoader] 初始化完成，快取目錄: {self.cache_dir}")

    def _get_cache_filename(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str
    ) -> str:
        """生成快取檔案名稱

        Args:
            tickers: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            str: 快取檔案名稱，格式為 {ticker1}_{ticker2}_{start}_{end}.csv
        """
        # 排序 tickers 確保一致性
        sorted_tickers = sorted(tickers)
        tickers_str = "_".join(sorted_tickers)

        # 移除日期中的連字號
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")

        return f"{tickers_str}_{start_clean}_{end_clean}.csv"

    def _get_cache_path(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str
    ) -> Path:
        """取得快取檔案完整路徑

        Args:
            tickers: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            Path: 快取檔案路徑
        """
        filename = self._get_cache_filename(tickers, start_date, end_date)
        return self.cache_dir / filename

    def _load_from_cache(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """從快取載入資料

        Args:
            tickers: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期

        Returns:
            Optional[pd.DataFrame]: 快取的資料，如果不存在則返回 None
        """
        cache_path = self._get_cache_path(tickers, start_date, end_date)

        if not cache_path.exists():
            return None

        try:
            print(f"[DataLoader] 從快取載入: {cache_path.name}")
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"[DataLoader] 快取載入成功，共 {len(data)} 筆資料")
            return data
        except Exception as e:
            print(f"[DataLoader] 警告：載入快取失敗 - {e}")
            return None

    def _save_to_cache(
        self,
        data: pd.DataFrame,
        tickers: list[str],
        start_date: str,
        end_date: str
    ) -> None:
        """將資料儲存到快取

        Args:
            data: 要快取的資料
            tickers: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期
        """
        cache_path = self._get_cache_path(tickers, start_date, end_date)

        try:
            data.to_csv(cache_path)
            print(f"[DataLoader] 資料已快取至: {cache_path.name}")
        except Exception as e:
            print(f"[DataLoader] 警告：儲存快取失敗 - {e}")

    def download_data(
        self,
        tickers: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """下載股票收盤價資料

        從 yfinance 下載指定標的的歷史收盤價，自動處理除權息。

        Args:
            tickers: 股票代碼列表，預設為 ["SPY", "TLT"]
            start_date: 開始日期，格式 "YYYY-MM-DD"，預設為 "2013-01-01"
            end_date: 結束日期，格式 "YYYY-MM-DD"，預設為 "2025-12-31"
            use_cache: 是否使用快取，預設為 True

        Returns:
            pd.DataFrame: 收盤價 DataFrame，index 為日期，columns 為股票代碼

        Raises:
            ValueError: 當下載失敗或資料為空時拋出

        Example:
            >>> loader = DataLoader()
            >>> data = loader.download_data(["SPY", "TLT"], "2020-01-01", "2023-12-31")
            >>> print(data.head())
                              SPY         TLT
            Date
            2020-01-02  324.87   141.03
            2020-01-03  322.41   142.26
            ...
        """
        # 設定預設值
        if tickers is None:
            tickers = DEFAULT_TICKERS
        if start_date is None:
            start_date = DEFAULT_START_DATE
        if end_date is None:
            end_date = DEFAULT_END_DATE

        print(f"[DataLoader] 下載資料: {tickers}")
        print(f"[DataLoader] 期間: {start_date} ~ {end_date}")

        # 嘗試從快取載入
        if use_cache:
            cached_data = self._load_from_cache(tickers, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # 從 yfinance 下載
        print(f"[DataLoader] 正在從 yfinance 下載...")
        try:
            # 使用 auto_adjust=True 自動處理除權息
            raw_data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,  # 自動調整除權息
                progress=True
            )
        except Exception as e:
            raise ValueError(f"下載資料失敗: {e}") from e

        # 檢查資料是否為空
        if raw_data.empty:
            raise ValueError(f"下載的資料為空，請檢查股票代碼和日期範圍")

        # 只取收盤價 (Close)
        if isinstance(raw_data.columns, pd.MultiIndex):
            # 多檔股票的情況
            close_data = raw_data["Close"]
        else:
            # 單檔股票的情況
            close_data = raw_data[["Close"]]
            close_data.columns = tickers

        print(f"[DataLoader] 下載完成，共 {len(close_data)} 筆資料")

        # 儲存到快取
        if use_cache:
            self._save_to_cache(close_data, tickers, start_date, end_date)

        return close_data

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """處理資料

        對下載的資料進行處理：
        1. 使用 ffill() 填補休市日的缺失值
        2. 使用 dropna() 刪除開頭的缺失值
        3. 確保所有標的的交易日對齊

        Args:
            data: 從 download_data() 取得的原始資料

        Returns:
            pd.DataFrame: 處理後的資料

        Raises:
            ValueError: 當輸入資料為空時拋出

        Example:
            >>> loader = DataLoader()
            >>> raw_data = loader.download_data()
            >>> processed = loader.process_data(raw_data)
            >>> print(f"處理前: {len(raw_data)} 筆, 處理後: {len(processed)} 筆")
        """
        if data.empty:
            raise ValueError("輸入資料為空")

        print(f"[DataLoader] 開始處理資料...")
        print(f"[DataLoader] 原始資料: {len(data)} 筆")

        # 1. 使用 forward fill 填補休市日
        # 某些市場可能在不同日期休市，造成 NaN
        filled_data = data.ffill()

        # 2. 刪除開頭的缺失值（可能是某些標的較晚開始交易）
        cleaned_data = filled_data.dropna()

        # 3. 確保交易日對齊（取所有標的都有資料的日期）
        aligned_data = cleaned_data.dropna(axis=0, how="any")

        print(f"[DataLoader] 處理完成: {len(aligned_data)} 筆")
        print(f"[DataLoader] 日期範圍: {aligned_data.index[0].date()} ~ "
              f"{aligned_data.index[-1].date()}")

        # 顯示缺失值統計
        missing_count = len(data) - len(aligned_data)
        if missing_count > 0:
            print(f"[DataLoader] 移除/填補了 {missing_count} 筆不完整資料")

        return aligned_data

    def align_trading_dates(self, data: pd.DataFrame) -> list[datetime]:
        """取得對齊後的交易日期

        返回所有標的都有交易的日期列表。

        Args:
            data: 股價資料 DataFrame

        Returns:
            list[datetime]: 對齊後的交易日期列表

        Example:
            >>> loader = DataLoader()
            >>> data = loader.download_data()
            >>> dates = loader.align_trading_dates(data)
            >>> print(f"共 {len(dates)} 個對齊交易日")
        """
        # 取得所有標的都有資料的日期
        aligned_data = data.dropna(axis=0, how="any")

        # 轉換為日期列表
        trading_dates = aligned_data.index.tolist()

        print(f"[DataLoader] 對齊交易日: {len(trading_dates)} 天")

        return trading_dates

    def get_returns(
        self,
        data: pd.DataFrame,
        method: str = "simple"
    ) -> pd.DataFrame:
        """計算報酬率

        Args:
            data: 收盤價資料
            method: 計算方法
                - "simple": 簡單報酬率 (r = P1/P0 - 1)
                - "log": 對數報酬率 (r = ln(P1/P0))

        Returns:
            pd.DataFrame: 報酬率序列

        Raises:
            ValueError: 當方法不支援時拋出
        """
        if method == "simple":
            returns = data.pct_change()
        elif method == "log":
            import numpy as np
            returns = np.log(data / data.shift(1))
        else:
            raise ValueError(f"不支援的計算方法: {method}，請使用 'simple' 或 'log'")

        return returns.dropna()

    def clear_cache(self) -> int:
        """清除所有快取

        Returns:
            int: 刪除的快取檔案數量
        """
        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.csv"):
            try:
                cache_file.unlink()
                deleted_count += 1
                print(f"[DataLoader] 已刪除: {cache_file.name}")
            except Exception as e:
                print(f"[DataLoader] 警告：刪除失敗 {cache_file.name} - {e}")

        print(f"[DataLoader] 共刪除 {deleted_count} 個快取檔案")
        return deleted_count

    def get_cache_info(self) -> dict:
        """取得快取資訊

        Returns:
            dict: 包含快取數量和大小的資訊
        """
        cache_files = list(self.cache_dir.glob("*.csv"))
        total_size = sum(f.stat().st_size for f in cache_files)

        info = {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_kb": total_size / 1024,
            "files": [f.name for f in cache_files],
        }

        return info

    def load_and_process(
        self,
        tickers: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """下載並處理資料（便捷方法）

        結合 download_data() 和 process_data() 的便捷方法。

        Args:
            tickers: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期
            use_cache: 是否使用快取

        Returns:
            pd.DataFrame: 處理後的收盤價資料

        Example:
            >>> loader = DataLoader()
            >>> data = loader.load_and_process()  # 使用預設參數
            >>> print(data.head())
        """
        raw_data = self.download_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        return self.process_data(raw_data)


# 便捷函數
def quick_load(
    tickers: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """快速載入資料的便捷函數

    Args:
        tickers: 股票代碼列表，預設為 ["SPY", "TLT"]
        start_date: 開始日期，預設為 "2013-01-01"
        end_date: 結束日期，預設為 "2025-12-31"

    Returns:
        pd.DataFrame: 處理後的收盤價資料

    Example:
        >>> from data.data_loader import quick_load
        >>> data = quick_load()  # 使用預設參數載入 SPY 和 TLT
        >>> data = quick_load(["AAPL", "MSFT"], "2020-01-01", "2023-12-31")
    """
    loader = DataLoader()
    return loader.load_and_process(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
