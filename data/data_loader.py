"""
資料載入模組

使用 yfinance 下載股票歷史數據，並提供本地快取功能。

作者：Smart Pilot Team
版本：1.0.0
"""

from typing import Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle

import pandas as pd
import yfinance as yf


class DataLoader:
    """資料載入器

    從 yfinance 下載股票數據，支援本地快取以加速重複查詢。

    Attributes:
        cache_dir: 快取目錄路徑
        cache_expiry_hours: 快取過期時間（小時）

    Example:
        >>> loader = DataLoader()
        >>> data = loader.download(["AAPL", "GOOGL"], start="2023-01-01")
        >>> prices = loader.get_adjusted_close(data)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_expiry_hours: int = 24
    ) -> None:
        """初始化資料載入器

        Args:
            cache_dir: 快取目錄路徑，預設為 data/cache
            cache_expiry_hours: 快取過期時間（小時），預設為 24 小時

        Raises:
            ValueError: 當過期時間為負數時拋出
        """
        if cache_expiry_hours < 0:
            raise ValueError(f"快取過期時間不能為負數，收到：{cache_expiry_hours}")

        if cache_dir is None:
            self.cache_dir = Path(__file__).parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_expiry_hours = cache_expiry_hours

        # 確保快取目錄存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        symbols: list[str],
        start: str,
        end: str
    ) -> str:
        """生成快取鍵值

        Args:
            symbols: 股票代碼列表
            start: 開始日期
            end: 結束日期

        Returns:
            str: 快取鍵值（MD5 雜湊）
        """
        key_str = f"{sorted(symbols)}_{start}_{end}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """取得快取檔案路徑

        Args:
            cache_key: 快取鍵值

        Returns:
            Path: 快取檔案路徑
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """檢查快取是否有效

        Args:
            cache_path: 快取檔案路徑

        Returns:
            bool: 快取是否有效
        """
        if not cache_path.exists():
            return False

        # 檢查快取是否過期
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = modified_time + timedelta(hours=self.cache_expiry_hours)

        return datetime.now() < expiry_time

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """從快取載入資料

        Args:
            cache_path: 快取檔案路徑

        Returns:
            Optional[pd.DataFrame]: 快取的資料，如果載入失敗則返回 None
        """
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            print(f"警告：載入快取失敗 - {e}")
            return None

    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """將資料儲存到快取

        Args:
            data: 要快取的資料
            cache_path: 快取檔案路徑
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except (pickle.PickleError, IOError) as e:
            print(f"警告：儲存快取失敗 - {e}")

    def download(
        self,
        symbols: Union[str, list[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """下載股票歷史數據

        Args:
            symbols: 股票代碼，可以是單一代碼或代碼列表
            start: 開始日期，格式 "YYYY-MM-DD"
            end: 結束日期，格式 "YYYY-MM-DD"（預設為今天）
            period: 期間，例如 "1y", "6mo", "3mo"（與 start/end 互斥）
            use_cache: 是否使用快取，預設為 True

        Returns:
            pd.DataFrame: 股票歷史數據，包含 Open, High, Low, Close, Adj Close, Volume

        Raises:
            ValueError: 當沒有指定 start 或 period 時拋出

        Example:
            >>> loader = DataLoader()
            >>> # 使用日期範圍
            >>> data = loader.download(["AAPL", "GOOGL"], start="2023-01-01", end="2023-12-31")
            >>> # 使用期間
            >>> data = loader.download("AAPL", period="1y")
        """
        # 標準化輸入
        if isinstance(symbols, str):
            symbols = [symbols]

        # 參數驗證
        if start is None and period is None:
            raise ValueError("必須指定 start 或 period 其中之一")

        # 設定結束日期
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # 嘗試從快取載入
        if use_cache and start is not None:
            cache_key = self._get_cache_key(symbols, start, end)
            cache_path = self._get_cache_path(cache_key)

            if self._is_cache_valid(cache_path):
                cached_data = self._load_from_cache(cache_path)
                if cached_data is not None:
                    print(f"從快取載入資料: {symbols}")
                    return cached_data

        # 從 yfinance 下載
        print(f"從 yfinance 下載資料: {symbols}")
        try:
            if period:
                data = yf.download(
                    symbols,
                    period=period,
                    progress=False,
                    auto_adjust=False
                )
            else:
                data = yf.download(
                    symbols,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False
                )
        except Exception as e:
            raise RuntimeError(f"下載資料失敗: {e}") from e

        # 儲存到快取
        if use_cache and start is not None and not data.empty:
            self._save_to_cache(data, cache_path)

        return data

    def get_adjusted_close(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取調整後收盤價

        Args:
            data: 從 download() 取得的原始資料

        Returns:
            pd.DataFrame: 調整後收盤價，columns 為股票代碼

        Note:
            調整後收盤價已考慮股票分割和股息，適合用於計算報酬。
        """
        if data.empty:
            return pd.DataFrame()

        # 處理單一股票和多股票的不同格式
        if isinstance(data.columns, pd.MultiIndex):
            # 多股票情況
            return data["Adj Close"]
        else:
            # 單一股票情況
            return data[["Adj Close"]].rename(columns={"Adj Close": "price"})

    def get_returns(
        self,
        data: pd.DataFrame,
        method: str = "simple"
    ) -> pd.DataFrame:
        """計算報酬率

        Args:
            data: 從 download() 或 get_adjusted_close() 取得的資料
            method: 計算方法，"simple" 為簡單報酬，"log" 為對數報酬

        Returns:
            pd.DataFrame: 報酬率序列

        Raises:
            ValueError: 當方法不支援時拋出
        """
        prices = self.get_adjusted_close(data) if "Adj Close" in str(data.columns) else data

        if method == "simple":
            returns = prices.pct_change()
        elif method == "log":
            import numpy as np
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"不支援的計算方法: {method}，請使用 'simple' 或 'log'")

        return returns.dropna()

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """清除快取

        Args:
            older_than_hours: 清除超過指定小時數的快取，預設清除全部

        Returns:
            int: 刪除的快取檔案數量
        """
        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            if older_than_hours is not None:
                modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age_hours = (datetime.now() - modified_time).total_seconds() / 3600

                if age_hours < older_than_hours:
                    continue

            try:
                cache_file.unlink()
                deleted_count += 1
            except IOError as e:
                print(f"警告：刪除快取失敗 {cache_file} - {e}")

        return deleted_count

    def get_cache_info(self) -> dict:
        """取得快取資訊

        Returns:
            dict: 包含快取數量和大小的資訊
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "expiry_hours": self.cache_expiry_hours,
        }
