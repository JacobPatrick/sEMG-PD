import pytest
from src.core.factories import FeatureExtractorFactory
from src.pipeline.feature.manual_feature_extractor import ManualFeatureExtractor
from src.pipeline.feature.deep_learning_feature_extractor import DeepLearningFeatureExtractor


class TestFeatureExtractorFactory:
    """特征提取器工厂测试"""

    def test_factory_registration(self):
        """测试工厂注册功能"""
        factory = FeatureExtractorFactory()
        factory.register("manual", ManualFeatureExtractor)
        factory.register("deep_learning", DeepLearningFeatureExtractor)

        assert len(factory._extractors) == 2
        assert "manual" in factory._extractors
        assert "deep_learning" in factory._extractors

    def test_create_extractor(self):
        """测试创建特征提取器实例"""
        factory = FeatureExtractorFactory()
        factory.register("manual", ManualFeatureExtractor)
        extractor = factory.create("manual")

        assert isinstance(extractor, ManualFeatureExtractor)

    def test_create_with_params(self):
        """测试带参数创建特征提取器"""
        raise NotImplementedError
    
    def test_unknown_extractor(self):
        """测试请求未知特征提取器"""
        factory = FeatureExtractorFactory()
        with pytest.raises(ValueError):
            factory.create("unknown_extractor")
            