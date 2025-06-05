import os
import sys
import torch
from joblib import load

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curr_path + "/..")

from src.utils.model_evaluation import evaluate_classification
from src.core.factories import (
    DataLoaderFactory,
    FeatureExtractorFactory,
    ModelTrainerFactory,
)
from src.pipeline.data.new_data_loader import NewDataLoader
from src.pipeline.feature.manual_feature_extractor import (
    NewManualFeatureExtractor,
)
from src.pipeline.feature.minirocket_feature_extractor import (
    MiniRocketFeatureExtractor,
)
from src.pipeline.classification.svc import SVM
from src.pipeline.classification.cnn import CNN, CNN1DClassifier, ModelWrapper
from src.pipeline.classification.lda import LDAC
from src.pipeline.classification.rrc import RRC
from src.config.config import (
    ExperimentConfig,
    DataConfig,
    FeatureConfig,
    SplitConfig,
    TrainConfig,
)


class SoftVotingPipeline:
    def __init__(
        self,
        data_loader_factory,
        feature_extractor_factory,
        model_trainer_factory,
    ):
        """初始化流水线

        Args:
            data_loader_factory: 数据加载器工厂
            feature_extractor_factory: 特征提取器工厂
            model_trainer_factory: 模型训练器工厂
        """
        self.data_loader_factory = data_loader_factory
        self.feature_extractor_factory = feature_extractor_factory
        self.model_trainer_factory = model_trainer_factory

    def run(self, config: ExperimentConfig):
        """运行完整的机器学习流水线
        1. 加载数据集
        2. 特征提取
        3. 测试模型
        4. 存储实验结果

        Args:
            config: 配置参数字典

        Returns:
            包含交叉验证、测试结果和训练模型的字典
        """
        # 1. 加载数据集
        print("Loading data...")
        dataset = self.load_data(config.data)
        # 2. 特征提取
        print("Extracting features...")
        features = self.extract_features(dataset, config.feature)
        print("Loading models...")
        models = self.load_models(features, config.train)
        # 5. 测试模型，保存测试集结果
        print("Evaluating model...")
        self.test_model(models, features, config.train)
        print("Done!")

    def load_data(self, config):
        """加载数据集"""
        loader_type = config.data_loader
        loader = self.data_loader_factory.create(loader_type)
        return loader.load(config)

    def extract_features(
        self, dataset, config
    ):
        """特征提取"""
        extractor_type = config.feature_extractor
        extractor = self.feature_extractor_factory.create(extractor_type)
        return extractor.extract(dataset, config)
    
    def load_models(self, features, config):
        """加载模型"""
        # _, n_channels, n_features, n_windows = features[0].shape
        # model_path = "model/"
        # models = []
        # # 寻找所有名称匹配的模型文件
        # for filename in os.listdir(model_path):
        #     if filename.startswith(config.model_name) and filename.endswith(".pth"):
        #         # 先加载状态字典，检查结构
        #         model = torch.load(model_path + filename)
                
        #         # 创建适当的标签映射
        #         label_map = {i: i for i in range(5)}
        #         models.append(ModelWrapper(model=model, label_map=label_map))
        # return models
        models = []
        for filename in os.listdir(config.model_dir):
            if filename.startswith(config.model_name) and filename.endswith(".joblib"):
                model_path = os.path.join(config.model_dir, filename)
                model = load(model_path)
                models.append(model)
        return models
    
    def test_model(
        self,
        models,
        features,
        config,
    ):
        """测试模型"""
        tester_type = config.model_type
        tester = self.model_trainer_factory.create(tester_type, models=models)

        results = tester.predict(features)
        soft_results = tester.predict_proba(features)
        for i in range(len(models)):
            evaluation = evaluate_classification(
                results[0][:, i], results[1][:, i]
            )
            try:
                with open(
                    config.report_dir + f"/{config.model_name}_soft_voting_{i}.txt",
                    "w",
                ) as f:
                    f.write(
                        f"y=\n{results[0][:, i]}\n\ny_pred=\n{results[1][:, i]}\n\n{evaluation}\n\nsoft_y_pred=\n{soft_results[1][:, :, i]}"  # \n\nsoft_y_pred=\n{soft_results[1][:, :, i]}
                    )
            except Exception as e:
                print(f"Error saving test results: {e}")


def setup_factories():
    # 数据加载器工厂
    data_loader_factory = DataLoaderFactory()
    data_loader_factory.register("new_loader", NewDataLoader)

    # 特征提取器工厂
    feature_extractor_factory = FeatureExtractorFactory()
    feature_extractor_factory.register("manual", NewManualFeatureExtractor)
    feature_extractor_factory.register("minirocket", MiniRocketFeatureExtractor)

    # 模型训练工厂
    model_trainer_factory = ModelTrainerFactory()
    model_trainer_factory.register("svm", SVM)
    model_trainer_factory.register("cnn", CNN)
    model_trainer_factory.register("lda", LDAC)
    model_trainer_factory.register("rrc", RRC)

    return {
        "data_loader_factory": data_loader_factory,
        "feature_extractor_factory": feature_extractor_factory,
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
    config = load_config("experiment", "src/config/soft_voting.yaml")

    # 初始化工厂
    factories = setup_factories()

    # 创建流水线
    pipeline = SoftVotingPipeline(**factories)

    # 传入完整配置，运行流水线
    pipeline.run(config)
