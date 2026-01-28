"""
DataLoader 單元測試

測試 data/data_loader.py 中的 DataLoader 類別功能。

作者：Smart Pilot Team
版本：1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, quick_load, DEFAULT_TICKERS, DEFAULT_START_DATE, DEFAULT_END_DATE


class TestDataLoaderInit:
    """測試 DataLoader 初始化"""

    def test_default_cache_dir(self):
        """測試預設快取目錄"""
        loader = DataLoader()
        assert loader.cache_dir.exists()
        assert "cache" in str(loader.cache_dir)

    def test_custom_cache_dir(self):
        """測試自訂快取目錄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_cache"
            loader = DataLoader(cache_dir=str(custom_path))
            assert loader.cache_dir == custom_path
            assert custom_path.exists()


class TestCacheFilename:
    """測試快取檔名生成"""

    def test_cache_filename_format(self):
        """測試快取檔名格式"""
        loader = DataLoader()
        filename = loader._get_cache_filename(
            tickers=["SPY", "TLT"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        assert filename == "SPY_TLT_20200101_20231231.csv"

    def test_cache_filename_sorted_tickers(self):
        """測試 tickers 排序一致性"""
        loader = DataLoader()
        filename1 = loader._get_cache_filename(
            tickers=["TLT", "SPY"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        filename2 = loader._get_cache_filename(
            tickers=["SPY", "TLT"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        # 無論順序如何，檔名應該相同
        assert filename1 == filename2

    def test_cache_filename_multiple_tickers(self):
        """測試多個 tickers 的檔名"""
        loader = DataLoader()
        filename = loader._get_cache_filename(
            tickers=["AAPL", "GOOGL", "MSFT"],
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        assert filename == "AAPL_GOOGL_MSFT_20200101_20231231.csv"


class TestProcessData:
    """測試資料處理功能"""

    def test_process_data_ffill(self):
        """測試 forward fill 功能"""
        loader = DataLoader()

        # 建立含有 NaN 的測試資料
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "SPY": [100, np.nan, 102, np.nan, 104],
            "TLT": [50, 51, np.nan, 53, 54]
        }, index=dates)

        processed = loader.process_data(data)

        # 檢查沒有 NaN
        assert not processed.isna().any().any()

    def test_process_data_dropna_head(self):
        """測試刪除開頭 NaN"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "SPY": [np.nan, np.nan, 102, 103, 104],
            "TLT": [np.nan, 51, 52, 53, 54]
        }, index=dates)

        processed = loader.process_data(data)

        # 開頭的 NaN 應該被刪除
        assert len(processed) < len(data)
        assert not processed.isna().any().any()

    def test_process_data_empty_raises(self):
        """測試空資料拋出例外"""
        loader = DataLoader()
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="輸入資料為空"):
            loader.process_data(empty_data)

    def test_process_data_alignment(self):
        """測試交易日對齊"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "SPY": [100, 101, 102, 103, 104],
            "TLT": [50, 51, 52, 53, 54]
        }, index=dates)

        processed = loader.process_data(data)

        # 所有列應該對齊
        assert len(processed) == 5
        assert list(processed.columns) == ["SPY", "TLT"]


class TestAlignTradingDates:
    """測試交易日對齊功能"""

    def test_align_trading_dates_returns_list(self):
        """測試返回日期列表"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "SPY": [100, 101, 102, 103, 104],
            "TLT": [50, 51, 52, 53, 54]
        }, index=dates)

        trading_dates = loader.align_trading_dates(data)

        assert isinstance(trading_dates, list)
        assert len(trading_dates) == 5

    def test_align_trading_dates_excludes_nan(self):
        """測試排除含 NaN 的日期"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "SPY": [100, np.nan, 102, 103, 104],
            "TLT": [50, 51, 52, np.nan, 54]
        }, index=dates)

        trading_dates = loader.align_trading_dates(data)

        # 只有 3 天兩邊都有資料
        assert len(trading_dates) == 3


class TestGetReturns:
    """測試報酬率計算"""

    def test_simple_returns(self):
        """測試簡單報酬率"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        data = pd.DataFrame({
            "SPY": [100, 110, 121, 133.1]
        }, index=dates)

        returns = loader.get_returns(data, method="simple")

        # 檢查報酬率計算
        assert len(returns) == 3  # dropna 後少一筆
        assert abs(returns["SPY"].iloc[0] - 0.10) < 0.001  # 10%
        assert abs(returns["SPY"].iloc[1] - 0.10) < 0.001  # 10%

    def test_log_returns(self):
        """測試對數報酬率"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        data = pd.DataFrame({
            "SPY": [100, 110, 121]
        }, index=dates)

        returns = loader.get_returns(data, method="log")

        # 對數報酬率
        expected = np.log(110 / 100)
        assert abs(returns["SPY"].iloc[0] - expected) < 0.001

    def test_invalid_method_raises(self):
        """測試無效方法拋出例外"""
        loader = DataLoader()

        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        data = pd.DataFrame({"SPY": [100, 110, 121]}, index=dates)

        with pytest.raises(ValueError, match="不支援的計算方法"):
            loader.get_returns(data, method="invalid")


class TestCacheOperations:
    """測試快取操作"""

    def test_save_and_load_cache(self):
        """測試儲存和載入快取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            # 建立測試資料
            dates = pd.date_range("2020-01-01", periods=5, freq="D")
            data = pd.DataFrame({
                "SPY": [100, 101, 102, 103, 104],
                "TLT": [50, 51, 52, 53, 54]
            }, index=dates)

            tickers = ["SPY", "TLT"]
            start = "2020-01-01"
            end = "2020-01-05"

            # 儲存快取
            loader._save_to_cache(data, tickers, start, end)

            # 載入快取
            loaded = loader._load_from_cache(tickers, start, end)

            assert loaded is not None
            assert len(loaded) == len(data)
            pd.testing.assert_frame_equal(loaded, data, check_freq=False)

    def test_load_nonexistent_cache(self):
        """測試載入不存在的快取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            result = loader._load_from_cache(
                ["SPY", "TLT"],
                "2020-01-01",
                "2020-12-31"
            )

            assert result is None

    def test_clear_cache(self):
        """測試清除快取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            # 建立測試快取檔案
            (Path(tmpdir) / "test1.csv").touch()
            (Path(tmpdir) / "test2.csv").touch()

            deleted = loader.clear_cache()

            assert deleted == 2
            assert len(list(Path(tmpdir).glob("*.csv"))) == 0

    def test_get_cache_info(self):
        """測試取得快取資訊"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            # 建立測試快取檔案
            test_file = Path(tmpdir) / "test.csv"
            test_file.write_text("test data")

            info = loader.get_cache_info()

            assert info["file_count"] == 1
            assert "test.csv" in info["files"]


class TestDownloadDataMocked:
    """測試下載資料（使用 mock）"""

    @patch('data.data_loader.yf.download')
    def test_download_data_calls_yfinance(self, mock_download):
        """測試呼叫 yfinance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            # 設定 mock 返回值
            dates = pd.date_range("2020-01-01", periods=5, freq="D")
            mock_data = pd.DataFrame({
                ("Close", "SPY"): [100, 101, 102, 103, 104],
                ("Close", "TLT"): [50, 51, 52, 53, 54]
            }, index=dates)
            mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
            mock_download.return_value = mock_data

            result = loader.download_data(
                tickers=["SPY", "TLT"],
                start_date="2020-01-01",
                end_date="2020-01-05",
                use_cache=False
            )

            # 驗證 yfinance 被呼叫
            mock_download.assert_called_once()

            # 驗證返回 Close 價格
            assert "SPY" in result.columns
            assert "TLT" in result.columns

    @patch('data.data_loader.yf.download')
    def test_download_uses_cache(self, mock_download):
        """測試使用快取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            # 建立快取資料
            dates = pd.date_range("2020-01-01", periods=5, freq="D")
            cached_data = pd.DataFrame({
                "SPY": [100, 101, 102, 103, 104],
                "TLT": [50, 51, 52, 53, 54]
            }, index=dates)

            loader._save_to_cache(
                cached_data,
                ["SPY", "TLT"],
                "2020-01-01",
                "2020-01-05"
            )

            # 下載資料（應該從快取載入）
            result = loader.download_data(
                tickers=["SPY", "TLT"],
                start_date="2020-01-01",
                end_date="2020-01-05",
                use_cache=True
            )

            # yfinance 不應該被呼叫
            mock_download.assert_not_called()

            # 資料應該從快取載入
            assert len(result) == 5

    @patch('data.data_loader.yf.download')
    def test_download_empty_raises(self, mock_download):
        """測試下載空資料拋出例外"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            mock_download.return_value = pd.DataFrame()

            with pytest.raises(ValueError, match="下載的資料為空"):
                loader.download_data(
                    tickers=["INVALID"],
                    start_date="2020-01-01",
                    end_date="2020-01-05",
                    use_cache=False
                )


class TestDefaultValues:
    """測試預設值"""

    def test_default_tickers(self):
        """測試預設標的"""
        assert DEFAULT_TICKERS == ["SPY", "TLT"]

    def test_default_start_date(self):
        """測試預設開始日期"""
        assert DEFAULT_START_DATE == "2013-01-01"

    def test_default_end_date(self):
        """測試預設結束日期"""
        assert DEFAULT_END_DATE == "2025-12-31"


class TestLoadAndProcess:
    """測試 load_and_process 便捷方法"""

    @patch('data.data_loader.yf.download')
    def test_load_and_process_combines_methods(self, mock_download):
        """測試結合下載和處理"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(cache_dir=tmpdir)

            # 設定 mock
            dates = pd.date_range("2020-01-01", periods=5, freq="D")
            mock_data = pd.DataFrame({
                ("Close", "SPY"): [100, 101, 102, 103, 104],
                ("Close", "TLT"): [50, 51, 52, 53, 54]
            }, index=dates)
            mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
            mock_download.return_value = mock_data

            result = loader.load_and_process(
                tickers=["SPY", "TLT"],
                start_date="2020-01-01",
                end_date="2020-01-05",
                use_cache=False
            )

            # 結果應該是處理過的資料
            assert not result.isna().any().any()
            assert len(result) == 5


# 執行測試的入口
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
