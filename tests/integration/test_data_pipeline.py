from src.core.factories import DataLoaderFactory, FeatureExtractorFactory
from src.pipeline.data.full_data_loader import FullDataLoader
from src.pipeline.feature.manual_feature_extractor import ManualFeatureExtractor
from src.pipeline.pipeline import StandardMLPipeline


class TestDataPipeline:
    """数据加载流水线集成测试"""

    def test_pipeline_data_loading(self, mock_config, mocker):
        """测试流水线中的数据加载"""
        # 创建工厂并注册加载器
        factory = DataLoaderFactory()
        factory.register("full_loader", FullDataLoader)

        # 模拟其他工厂
        mock_preprocessor_factory = mocker.MagicMock()
        mock_feature_factory = mocker.MagicMock()
        mock_model_factory = mocker.MagicMock()
        mock_validator_factory = mocker.MagicMock()

        # 创建流水线，但只测试数据加载部分
        pipeline = StandardMLPipeline(
            data_loader_factory=factory,
            preprocessor_factory=mock_preprocessor_factory,
            feature_extractor_factory=mock_feature_factory,
            model_trainer_factory=mock_model_factory,
            model_validator_factory=mock_validator_factory,
        )

        # 执行数据加载
        data = pipeline.load_data(mock_config)

        # 验证数据结构
        assert "raw" in data
        assert "labels" in data
        assert "sub-1" in data["raw"]


class TestFeatureExtractorPipeline:
    """特征提取流水线集成测试"""

    def test_pipeline_feature_extraction(self, mock_config, mocker):
        """测试流水线中的特征提取"""
        factory = FeatureExtractorFactory()
        factory.register("manual", ManualFeatureExtractor)

        mock_data_loader_factory = mocker.MagicMock()
        mock_preprocessor_factory = mocker.MagicMock()
        mock_model_factory = mocker.MagicMock()
        mock_validator_factory = mocker.MagicMock()

        pipeline = StandardMLPipeline(
            data_loader_factory=mock_data_loader_factory,
            preprocessor_factory=mock_preprocessor_factory,
            feature_extractor_factory=factory,
            model_trainer_factory=mock_model_factory,
            model_validator_factory=mock_validator_factory,
        )

        features = pipeline.extract_features({}, mock_config)

        assert "features" in features
