import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from src.utils.save_model import save_model
from src.utils.model_evaluation import evaluate_classification
from src.config.config import (
    ExperimentConfig,
    DataConfig,
    FeatureConfig,
    SplitConfig,
    TrainConfig,
)


class MLPipeline(ABC):
    """传统机器学习流水线的基础模板类"""

    def __init__(self):
        pass

    def run(self, config: ExperimentConfig) -> Dict[str, Any]:
        """运行完整的机器学习流水线
        1. 加载数据集
        2. 特征提取
        3. train-test 划分
        4. 训练模型
        5. 测试模型
        6. 存储实验结果

        Args:
            config: 配置参数字典

        Returns:
            包含交叉验证、测试结果和训练模型的字典
        """
        # TODO: 定义算法骨架
        # 1. 加载数据集
        print("Loading data...")
        dataset = self.load_data(config.data)
        # 2. 特征提取
        print("Extracting features...")
        features = self.extract_features(dataset, config.feature)
        # 3. train-test 划分
        print("Splitting data...")
        splitted_data = self.split_data(features, config.split)
        # 4. 训练模型，保存训练集结果
        print("Training model...")
        models = self.train_model(splitted_data["train"], config.train)
        # 5. 测试模型，保存测试集结果
        print("Evaluating model...")
        self.test_model(models, splitted_data["test"], config.train)

        print("Done!")

    @abstractmethod
    def load_data(self, config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集"""
        pass

    @abstractmethod
    def extract_features(
        self, data: Tuple[np.ndarray, np.ndarray], config: FeatureConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """特征提取"""
        pass

    @abstractmethod
    def split_data(
        self, features: Tuple[np.ndarray, np.ndarray], config: SplitConfig
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """划分训练集和测试集"""
        pass

    @abstractmethod
    def train_model(
        self, features: Tuple[np.ndarray, np.ndarray], config: TrainConfig
    ) -> Any:
        """训练模型"""
        pass

    @abstractmethod
    def test_model(
        self,
        model: Any,
        features: Tuple[np.ndarray, np.ndarray],
        config: TrainConfig,
    ) -> None:
        """测试模型"""
        pass


class StandardMLPipeline(MLPipeline):
    """标准机器学习流水线实现"""

    def __init__(
        self,
        data_loader_factory,
        feature_extractor_factory,
        data_splitter_factory,
        model_trainer_factory,
    ):
        """初始化流水线

        Args:
            data_loader_factory: 数据加载器工厂
            feature_extractor_factory: 特征提取器工厂
            data_splitter_factory: 数据分割器工厂
            model_trainer_factory: 模型训练器工厂
        """
        self.data_loader_factory = data_loader_factory
        self.feature_extractor_factory = feature_extractor_factory
        self.data_splitter_factory = data_splitter_factory
        self.model_trainer_factory = model_trainer_factory

    def load_data(self, config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集"""
        loader_type = config.data_loader
        loader = self.data_loader_factory.create(loader_type)
        return loader.load(config)

    def extract_features(
        self, dataset: Tuple[np.ndarray, np.ndarray], config: FeatureConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """特征提取"""
        extractor_type = config.feature_extractor
        extractor = self.feature_extractor_factory.create(extractor_type)
        return extractor.extract(dataset, config)

    def split_data(
        self, dataset: Tuple[np.ndarray, np.ndarray], config: SplitConfig
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """划分训练集和测试集"""
        splitter_type = config.split_type
        splitter = self.data_splitter_factory.create(splitter_type)
        return splitter.split(dataset)

    def train_model(
        self, features: Tuple[np.ndarray, np.ndarray], config: TrainConfig
    ) -> Any:
        """训练模型"""
        trainer_type = config.model_type
        trainer = self.model_trainer_factory.create(trainer_type)
        models = trainer.fit(features)
        results = trainer.predict(features)

        # 保存训练集结果
        for i in range(len(models)):
            evaluation = evaluate_classification(
                results[0][:, i], results[1][:, i]
            )
            try:
                with open(
                    config.report_dir + f"/train_results_{i}.txt", "w"
                ) as f:
                    f.write(
                        f"y=\n{results[0][:, i]}\n\ny_pred=\n{results[1][:, i]}\n\n{evaluation}"
                    )
            except Exception as e:
                print(f"Error saving train results: {e}")
            finally:
                # 针对 CNN 分类模型
                if hasattr(models[i], "model"):
                    save_model(
                        models[i].model,
                        config.model_dir,
                        f"{config.model_name}_{i}.joblib",
                    )
                else:
                    save_model(
                        models[i],
                        config.model_dir,
                        f"{config.model_name}_{i}.joblib",
                    )

        return models

    def test_model(
        self,
        models: Any,
        features: Tuple[np.ndarray, np.ndarray],
        config: TrainConfig,
    ) -> None:
        """测试模型"""
        tester_type = config.model_type
        tester = self.model_trainer_factory.create(tester_type, models=models)

        results = tester.predict(features)
        for i in range(len(models)):
            evaluation = evaluate_classification(
                results[0][:, i], results[1][:, i]
            )
            try:
                with open(
                    config.report_dir + f"/test_results_{i}.txt", "w"
                ) as f:
                    f.write(
                        f"y=\n{results[0][:, i]}\n\ny_pred=\n{results[1][:, i]}\n\n{evaluation}"
                    )
            except Exception as e:
                print(f"Error saving test results: {e}")
