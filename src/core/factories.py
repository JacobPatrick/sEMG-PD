from typing import Dict, Type, Any
from src.interfaces.data import DataLoader
from src.interfaces.feature import FeatureExtractor


class DataLoaderFactory:
    """数据加载器工厂类"""

    def __init__(self):
        self._loaders = {}

    def register(self, name: str, loader_class: Type[DataLoader]) -> None:
        """注册加载器类"""
        self._loaders[name] = loader_class

    def create(self, name: str, **kwargs) -> DataLoader:
        """创建加载器实例"""
        if name not in self._loaders:
            raise ValueError(f"未知的加载器类型: {name}")

        return self._loaders[name](**kwargs)


class FeatureExtractorFactory:
    """特征提取器工厂类"""

    def __init__(self):
        self._extractors: Dict[str, Type[FeatureExtractor]] = {}

    def register(
        self, name: str, extractor_class: Type[FeatureExtractor]
    ) -> None:
        """注册新的特征提取器类"""
        self._extractors[name] = extractor_class

    def create(self, name: str, **kwargs) -> FeatureExtractor:
        """创建特征提取器实例"""
        if name not in self._extractors:
            raise ValueError(f"未知的特征提取器类型: {name}")

        return self._extractors[name](**kwargs)
