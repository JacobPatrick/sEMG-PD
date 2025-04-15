from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from src.utils.save_model import save_model

from src.config.config import (
    ExperimentConfig,
    DataConfig,
    PreprocessConfig,
    FeatureConfig,
    SplitConfig,
    TrainConfig,
    OutputConfig,
)


class MLPipeline(ABC):
    """机器学习流水线的基础模板类"""

    def __init__(self):
        pass

    def run(self, config: ExperimentConfig) -> Dict[str, Any]:
        """运行完整的机器学习流水线
        1. 加载数据
        2. 预处理数据
        3. 特征提取
        4. 训练模型
        5. 验证模型
        6. 存储实验结果

        Args:
            config: 配置参数字典

        Returns:
            包含交叉验证、测试结果和测试模型的字典
        """
        # 定义算法骨架
        data = self.load_data(config.data)
        preprocessed_data = self.preprocess_data(data, config.preprocess)
        features = self.extract_features(preprocessed_data, config.feature)
        splitted_data = self.split_data(features, config.split)
        cv_results = []
        for split in splitted_data["cv_splits"]:
            models = self.train_model(split["train"], config.train)
            results = self.validate_model(models, split["val"], config.train)
            cv_results.append(results)
        final_models = self.train_model(splitted_data["train"], config.train)
        test_results = self.validate_model(
            final_models, splitted_data["test"], config.train
        )
        results = dict(
            cv_results=cv_results,
            test_results=test_results,
            final_models=final_models,
        )
        self.save_results(results, config.output)
        return results

    @abstractmethod
    def load_data(self, config: DataConfig) -> Dict[str, Any]:
        """加载数据"""
        pass

    @abstractmethod
    def preprocess_data(
        self, data: Any, config: PreprocessConfig
    ) -> Dict[str, Any]:
        """预处理数据"""
        pass

    @abstractmethod
    def extract_features(
        self, data: Any, config: FeatureConfig
    ) -> Dict[str, Any]:
        """特征提取"""
        pass

    @abstractmethod
    def split_data(self, data: Any, config: SplitConfig) -> Dict[str, Any]:
        """数据分割"""
        pass

    @abstractmethod
    def train_model(self, features: Any, config: TrainConfig) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def validate_model(
        self, models: Any, features: Any, config: TrainConfig
    ) -> Dict[str, Any]:
        """验证模型"""
        pass

    @abstractmethod
    def save_results(
        self, results: Dict[str, Any], config: OutputConfig
    ) -> None:
        """保存结果"""
        pass


class StandardMLPipeline(MLPipeline):
    """标准机器学习流水线实现

    使用工厂模式获取各个处理阶段的具体策略实现
    """

    def __init__(
        self,
        data_loader_factory,
        preprocessor_factory,
        feature_extractor_factory,
        data_splitter_factory,
        model_trainer_factory,
    ):
        """初始化流水线

        Args:
            data_loader_factory: 数据加载器工厂
            preprocessor_factory: 预处理器工厂
            feature_extractor_factory: 特征提取器工厂
            data_splitter_factory: 数据分割器工厂
            model_trainer_factory: 模型训练器工厂
        """
        self.data_loader_factory = data_loader_factory
        self.preprocessor_factory = preprocessor_factory
        self.feature_extractor_factory = feature_extractor_factory
        self.data_splitter_factory = data_splitter_factory
        self.model_trainer_factory = model_trainer_factory

    def load_data(self, config: DataConfig) -> Dict[str, Any]:
        """加载数据"""
        loader_type = config.data_loader
        loader = self.data_loader_factory.create(loader_type)
        return loader.load(config)

    def preprocess_data(
        self, data: Dict[str, Any], config: PreprocessConfig
    ) -> Dict[str, Any]:
        """数据预处理"""
        preprocessor_type = config.preprocess_type
        preprocessor = self.preprocessor_factory.create(preprocessor_type)
        return preprocessor.process(data, config)

    def extract_features(
        self, data: Dict[str, Any], config: FeatureConfig
    ) -> Dict[str, Any]:
        """特征提取"""
        extractor_type = config.feature_extractor
        extractor = self.feature_extractor_factory.create(extractor_type)
        return extractor.extract(data, config)

    def split_data(
        self, data: Dict[str, Any], config: SplitConfig
    ) -> Dict[str, Any]:
        """数据分割"""
        split_type = config.split_type
        splitter = self.data_splitter_factory.create(split_type)
        return splitter.train_val_test_split(data)

    def train_model(self, features: Tuple, config: TrainConfig) -> List[Any]:
        """模型训练"""
        trainer_type = config.model_type
        trainer = self.model_trainer_factory.create(trainer_type)
        return trainer.fit(features)

    def validate_model(
        self, models: Any, features: Any, config: TrainConfig
    ) -> Any:
        """模型验证"""
        trainer_type = config.model_type
        trainer = self.model_trainer_factory.create(trainer_type, models=models)
        return trainer.predict(features)

    def save_results(
        self, results: Dict[str, Any], config: OutputConfig
    ) -> None:
        """保存结果"""

        # 保存在测试集上训练得到的模型
        model = results["final_models"]
        save_model(model, config.model_dir, "test.joblib")

        # 保存交叉验证结果
        cv_results = results["cv_results"]
        try:
            with open(config.report_dir + "/cv_results.txt", "w") as f:
                for i, result in enumerate(cv_results):
                    f.write(f"Fold {i+1} results: {result}\n")
        except Exception as e:
            print(f"Error saving CV results: {e}")

        # 保存测试集结果
        test_results = results["test_results"]
        try:
            with open(config.report_dir + "/test_results.txt", "w") as f:
                f.write(f"Test results: {test_results}")
        except Exception as e:
            print(f"Error saving test dataset results: {e}")
