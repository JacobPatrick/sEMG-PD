import os, sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curr_path + "/..")

from src.config.config import load_config
from src.core.factories import (
    DataLoaderFactory,
    PreprocessorFactory,
    FeatureExtractorFactory,
    DataSplitterFactory,
    ModelTrainerFactory,
)
from src.pipeline.pipeline import StandardMLPipeline
from src.pipeline.data.full_data_loader import FullDataLoader
from src.pipeline.data.data_loader import LazyDataLoader, BatchDataLoader
from src.pipeline.feature.manual_feature_extractor import ManualFeatureExtractor
from src.pipeline.feature.deep_learning_feature_extractor import (
    DeepLearningFeatureExtractor,
)
from src.pipeline.preprocess.pass_through_preprocessor import (
    PassThroughPreprocessor,
)
from src.pipeline.split.splitter import DataSplitter
from src.pipeline.classification.svc import SVM

# 创建并配置工厂
def setup_factories():
    # 数据加载器工厂
    data_loader_factory = DataLoaderFactory()
    data_loader_factory.register("full_loader", FullDataLoader)
    data_loader_factory.register("lazy_loader", LazyDataLoader)
    data_loader_factory.register("batch_loader", BatchDataLoader)

    # 预处理器工厂
    preprocessor_factory = PreprocessorFactory()
    preprocessor_factory.register(
        "pass_through", PassThroughPreprocessor
    )  # 暂时不对sEMG数据做任何预处理

    # 特征提取器工厂
    feature_factory = FeatureExtractorFactory()
    feature_factory.register("manual", ManualFeatureExtractor)
    feature_factory.register("deep_learning", DeepLearningFeatureExtractor)

    # 数据分割器工厂
    data_splitter_factory = DataSplitterFactory()
    # data_splitter_factory.register("train_test_split", TrainTestSplitter)
    data_splitter_factory.register("train_val_test_split", DataSplitter)

    # 模型训练工厂
    model_trainer_factory = ModelTrainerFactory()
    model_trainer_factory.register("svm", SVM)

    return {
        "data_loader_factory": data_loader_factory,
        "preprocessor_factory": preprocessor_factory,
        "feature_extractor_factory": feature_factory,
        "data_splitter_factory": data_splitter_factory,
        "model_trainer_factory": model_trainer_factory,
    }


if __name__ == "__main__":
    # 加载配置
    config = load_config(
        "experiment", "src/config/data_config/experiment1.yaml"
    )

    # 设置工厂
    factories = setup_factories()

    # 创建流水线
    pipeline = StandardMLPipeline(**factories)

    # 运行流水线，传入完整配置
    results = pipeline.run(config)

    # 处理结果
    print(f"实验结果: {results}")
