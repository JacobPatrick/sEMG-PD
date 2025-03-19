from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """数据加载配置"""
    data_loader: str = "full_loader"


@dataclass
class PreprocessConfig:
    """预处理配置"""
    preprocess_type: str = "pass_through"


@dataclass
class FeatureConfig:
    """特征提取配置"""
    feature_extractor: str = "manual"
    window_size: float = 0.2
    overlap: float = 0.1
    sampling_rate: int = 2000
    features: List[str] = field(default_factory=lambda: ["mav", "rms", "zc", "mnf", "mdf"])


@dataclass
class TrainConfig:
    """模型训练配置"""
    model_type: str = "svm"
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidateConfig:
    """模型验证配置"""
    validator_type: str = "cross_validation"
    validator_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    data: DataConfig
    preprocess: PreprocessConfig
    feature: FeatureConfig
    train: TrainConfig
    validate: ValidateConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        # 创建各个配置部分
        data_config = DataConfig(**config_dict.get('data', {}))
        preprocess_config = PreprocessConfig(**config_dict.get('preprocess', {}))
        feature_config = FeatureConfig(**config_dict.get('feature', {}))
        train_config = TrainConfig(**config_dict.get('train', {}))
        validate_config = ValidateConfig(**config_dict.get('validate', {}))
        
        return cls(
            data=data_config,
            preprocess=preprocess_config,
            feature=feature_config,
            train=train_config,
            validate=validate_config
        )

    def validate_config(self) -> bool:
        """验证配置的有效性"""
        # 验证数据加载器类型
        valid_loaders = ["full_loader", "lazy_loader", "batch_loader"]
        if self.data.data_loader not in valid_loaders:
            raise ValueError(f"Invalid data loader type. Must be one of {valid_loaders}")
        
        # 验证特征提取器类型
        valid_extractors = ["manual", "deep_learning"]
        if self.feature.feature_extractor not in valid_extractors:
            raise ValueError(f"Invalid feature extractor type. Must be one of {valid_extractors}")
        
        # 验证窗口参数
        if self.feature.window_size <= 0 or self.feature.overlap < 0:
            raise ValueError("Window size must be positive and overlap must be non-negative")
        if self.feature.overlap >= self.feature.window_size:
            raise ValueError("Overlap must be smaller than window size")
            
        # 验证采样率
        if self.feature.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
            
        return True


def load_config(config_type: str, yaml_path: str) -> ExperimentConfig:
    """加载配置的便捷函数"""
    if config_type == "experiment":
        config = ExperimentConfig.from_yaml(yaml_path)
        config.validate_config()
        return config
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # 测试配置加载
    config = load_config("experiment", "src/config/data_config/experiment1.yaml")
    print(f"Data loader: {config.data.data_loader}")
    print(f"Feature extractor: {config.feature.feature_extractor}")
    print(f"Window size: {config.feature.window_size}")
