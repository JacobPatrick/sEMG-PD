import pytest
from src.core.factories import DataLoaderFactory
from src.pipeline.data.full_data_loader import FullDataLoader
from src.pipeline.data.data_loader import LazyDataLoader, BatchDataLoader


class TestDataLoaderFactory:
    """数据加载器工厂测试"""

    def test_factory_registration(self):
        """测试工厂注册功能"""
        factory = DataLoaderFactory()

        # 注册加载器
        factory.register("full_loader", FullDataLoader)
        factory.register("lazy_loader", LazyDataLoader)

        # 检查注册是否成功
        assert len(factory._loaders) == 2
        assert "full_loader" in factory._loaders
        assert "lazy_loader" in factory._loaders

    def test_create_loader(self):
        """测试创建加载器实例"""
        factory = DataLoaderFactory()
        factory.register("full_loader", FullDataLoader)

        # 创建加载器实例
        loader = factory.create("full_loader")

        # 验证实例类型
        assert isinstance(loader, FullDataLoader)

    def test_create_with_params(self):
        """测试带参数创建加载器"""
        factory = DataLoaderFactory()
        factory.register("batch_loader", BatchDataLoader)

        # 创建带参数的加载器
        loader = factory.create("batch_loader", batch_size=10)

        # 验证参数设置
        assert loader.batch_size == 10

    def test_unknown_loader(self):
        """测试请求未知加载器"""
        factory = DataLoaderFactory()

        # 应该抛出ValueError
        with pytest.raises(ValueError):
            factory.create("unknown_loader")
