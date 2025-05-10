import os, sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curr_path + "/..")

from src.config.config import load_config
from src.core.factories import (
    DataLoaderFactory,
    FeatureExtractorFactory,
    DataSplitterFactory,
    ModelTrainerFactory,
)
from src.pipeline.pipeline import StandardMLPipeline
from src.pipeline.data.new_data_loader import NewDataLoader
from src.pipeline.feature.manual_feature_extractor import (
    NewManualFeatureExtractor,
)
from src.pipeline.feature.minirocket_feature_extractor import (
    MiniRocketFeatureExtractor,
)
from src.pipeline.split.splitter import DataSplitter
from src.pipeline.split.train_test_splitter import TrainTestSplitter
from src.pipeline.classification.svc import SVM

# TODO: CNN classifier

from src.config.config import ExperimentConfig


def setup_factories():
    # 数据加载器工厂
    data_loader_factory = DataLoaderFactory()
    data_loader_factory.register("new_loader", NewDataLoader)

    # 特征提取器工厂
    feature_extractor_factory = FeatureExtractorFactory()
    feature_extractor_factory.register("manual", NewManualFeatureExtractor)
    feature_extractor_factory.register("minirocket", MiniRocketFeatureExtractor)

    # 数据分割器工厂
    data_splitter_factory = DataSplitterFactory()
    data_splitter_factory.register("train_test_split", TrainTestSplitter)
    data_splitter_factory.register("train_val_test_split", DataSplitter)

    # 模型训练工厂
    model_trainer_factory = ModelTrainerFactory()
    model_trainer_factory.register("svm", SVM)

    return {
        "data_loader_factory": data_loader_factory,
        "feature_extractor_factory": feature_extractor_factory,
        "data_splitter_factory": data_splitter_factory,
        "model_trainer_factory": model_trainer_factory,
    }


def load_config(config_type: str, yaml_path: str) -> ExperimentConfig | None:
    """加载配置的便捷函数"""
    if config_type == "experiment":
        config = ExperimentConfig.from_yaml(yaml_path)
        return config
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # 加载配置文件
    config = load_config("experiment", "src/config/ROCKETGait.yaml")

    # 初始化工厂
    factories = setup_factories()

    # 创建流水线
    pipeline = StandardMLPipeline(**factories)

    # 传入完整配置，运行流水线
    pipeline.run(config)
