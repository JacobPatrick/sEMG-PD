from typing import Dict, Type
from src.interfaces.data import DataLoader
from src.interfaces.preprocess import Preprocessor
from src.interfaces.feature import FeatureExtractor
from src.interfaces.split import Splitter
from interfaces.classification import Classification


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


class PreprocessorFactory:
    """预处理器工厂类"""

    def __init__(self):
        self._preprocessors: Dict[str, Type[Preprocessor]] = {}

    def register(
        self, name: str, preprocessor_class: Type[Preprocessor]
    ) -> None:
        """注册预处理器类"""
        self._preprocessors[name] = preprocessor_class

    def create(self, name: str, **kwargs) -> Preprocessor:
        """创建预处理器实例"""
        if name not in self._preprocessors:
            raise ValueError(f"未知的预处理器类型: {name}")

        return self._preprocessors[name](**kwargs)


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


class DataSplitterFactory:
    """数据分割器工厂类"""

    def __init__(self):
        self._splitters: Dict[str, Type[Splitter]] = {}

    def register(self, name: str, splitter_class: Type[Splitter]) -> None:
        """注册数据分割器类"""
        self._splitters[name] = splitter_class

    def create(self, name: str, **kwargs) -> Splitter:
        """创建数据分割器实例"""
        if name not in self._splitters:
            raise ValueError(f"未知的数据分割器类型: {name}")

        return self._splitters[name](**kwargs)


class ModelTrainerFactory:
    """分类模型训练工厂类"""

    def __init__(self):
        self._classifications: Dict[str, Type[Classification]] = {}

    def register(
        self, name: str, classifications_class: Type[Classification]
    ) -> None:
        """注册分类模型类"""
        self._classifications[name] = classifications_class

    def create(self, name: str, **kwargs) -> Classification:
        """创建分类模型实例"""
        if name not in self._classifications:
            raise ValueError(f"未知的分类模型类型: {name}")

        return self._classifications[name](**kwargs)


class ModelValidatorFactory:
    """分类模型验证工厂类"""

    pass
