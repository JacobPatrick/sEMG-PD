from src.config.config import load_config
from src.core.factories import (
    DataLoaderFactory,
    PreprocessorFactory,
    FeatureExtractorFactory,
    ModelTrainerFactory,
    ModelValidatorFactory,
)
from src.pipeline.pipeline import StandardMLPipeline
from src.pipeline.data.full_data_loader import FullDataLoader
from src.pipeline.data.data_loader import LazyDataLoader, BatchDataLoader
from src.pipeline.feature.manual_feature_extractor import ManualFeatureExtractor
from src.pipeline.feature.deep_learning_feature_extractor import (
    DeepLearningFeatureExtractor,
)

# 其他导入...


# 创建并配置工厂
def setup_factories():
    # 数据加载器工厂
    data_loader_factory = DataLoaderFactory()
    data_loader_factory.register("full_loader", FullDataLoader)
    data_loader_factory.register("lazy_loader", LazyDataLoader)
    data_loader_factory.register("batch_loader", BatchDataLoader)

    # 特征提取器工厂
    feature_factory = FeatureExtractorFactory()
    feature_factory.register("manual", ManualFeatureExtractor)
    feature_factory.register("deep_learning", DeepLearningFeatureExtractor)

    # 类似地配置其他工厂...

    return {
        "data_loader_factory": data_loader_factory,
        "preprocessor_factory": preprocessor_factory,
        "feature_extractor_factory": feature_factory,
        "model_trainer_factory": model_trainer_factory,
        "model_validator_factory": model_validator_factory,
    }


def main():
    # 加载配置
    config = load_config(
        "exp_config", "src/config/data_config/experiment1.yaml"
    )

    # 设置工厂
    factories = setup_factories()

    # 创建流水线
    pipeline = StandardMLPipeline(**factories)

    # 运行流水线
    results = pipeline.run(config)

    # 处理结果
    print(f"实验结果: {results}")


if __name__ == "__main__":
    main()
