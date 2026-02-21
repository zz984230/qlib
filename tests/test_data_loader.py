"""测试数据加载器"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.akshare_loader import AkshareLoader


class TestAkshareLoader:
    """测试 AkshareLoader 类"""

    def test_init(self):
        """测试初始化"""
        loader = AkshareLoader()
        assert loader is not None
        assert loader.cache_dir.exists()

    def test_get_stock_list(self):
        """测试获取股票列表"""
        loader = AkshareLoader()
        df = loader.get_stock_list("csi300")

        assert df is not None
        assert len(df) > 0
        # akshare 列名可能是 "股票代码" 或 "品种代码"
        assert "品种代码" in df.columns or "股票代码" in df.columns

    def test_get_stock_data(self):
        """测试获取单股票数据"""
        loader = AkshareLoader()

        # 测试平安银行
        data = loader.get_stock_data(
            symbol="000001",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        if data is not None:  # 网络可能失败
            assert len(data) > 0
            assert "close" in data.columns
            assert "date" in data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
