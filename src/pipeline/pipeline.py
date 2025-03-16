from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class MLPipeline(ABC):
    """机器学习流水线的基础模板类"""

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
            包含结果的字典
        """
        # 定义算法骨架
        data = self.load_data(config)
        preprocessed_data = self.preprocess_data(data, config)
        features = self.extract_features(preprocessed_data, config)
        model = self.train_model(features, config)
        results = self.validate_model(model, features, config)
        self.save_results(results, config)
        return results

    @abstractmethod
    def load_data(self, config: Dict[str, Any]) -> Any:
        """加载数据"""
        pass

    @abstractmethod
    def preprocess_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """预处理数据"""
        pass

    @abstractmethod
    def extract_features(self, data: Any, config: Dict[str, Any]) -> Any:
        """特征提取"""
        pass

    @abstractmethod
    def train_model(self, features: Any, config: Dict[str, Any]) -> Any:
        """训练模型"""
        pass

    @abstractmethod
    def validate_model(
        self, model: Any, features: Any, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证模型"""
        pass

    @abstractmethod
    def save_results(
        self, results: Dict[str, Any], config: Dict[str, Any]
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
        model_trainer_factory,
        model_validator_factory,
    ):
        """初始化流水线

        Args:
            data_loader_factory: 数据加载器工厂
            preprocessor_factory: 预处理器工厂
            feature_extractor_factory: 特征提取器工厂
            model_trainer_factory: 模型训练器工厂
            model_validator_factory: 模型验证器工厂
        """
        self.data_loader_factory = data_loader_factory
        self.preprocessor_factory = preprocessor_factory
        self.feature_extractor_factory = feature_extractor_factory
        self.model_trainer_factory = model_trainer_factory
        self.model_validator_factory = model_validator_factory

    def load_data(self, config) -> Any:
        """使用数据加载器策略加载数据"""
        loader_type = config.data_loader
        loader = self.data_loader_factory.create(loader_type)
        return loader.load(config)

    def preprocess_data(self, data: Any, config) -> Any:
        """使用预处理器策略预处理数据"""
        preprocessor_type = config.preprocessor
        preprocessor = self.preprocessor_factory.create(preprocessor_type)
        return preprocessor.process(data, config)

    def extract_features(self, data: Any, config) -> Any:
        """使用特征提取器策略提取特征"""
        extractor_type = config.feature_extractor
        extractor = self.feature_extractor_factory.create(extractor_type)
        return extractor.extract(data, config)

    def train_model(self, features: Any, config) -> Any:
        """使用模型训练器策略训练模型"""
        trainer_type = config.model_trainer
        trainer = self.model_trainer_factory.create(trainer_type)
        return trainer.train(features, config)

    def validate_model(
        self, model: Any, features: Any, config
    ) -> Dict[str, Any]:
        """使用模型验证器策略验证模型"""
        validator_type = config.model_validator
        validator = self.model_validator_factory.create(validator_type)
        return validator.validate(model, features, config)

    def save_results(self, results: Dict[str, Any], config) -> None:
        """保存结果到指定位置"""
        # 保存结果的实现，可能包括模型、报告等
        output_dir = config.output_dir
        # 保存逻辑...
