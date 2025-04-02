import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class DataConfig:
    """数据加载配置"""

    data_dir: str = "raw/"
    labels_file: str = "labels.csv"
    data_loader: str = "full_loader"  # 默认全量加载


@dataclass
class PreprocessConfig:
    """预处理配置"""

    preprocess_type: str = "pass_through"  # 默认无预处理


@dataclass
class FeatureConfig:
    """特征提取配置"""

    feature_extractor: str = "manual"  # 默认提取手工特征
    window_size: float = 0.2
    overlap: float = 0.1
    sampling_rate: int = 2000
    features: List[str] = field(
        default_factory=lambda: ["mav", "rms", "zc", "mnf", "mdf"]
    )


@dataclass
class SplitConfig:
    """数据分割配置"""

    split_type: str = "train_test_split"  # 默认使用简单测试/验证分割
    split_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """模型训练配置"""

    model_type: str = "svm"  # 默认使用SVM模型
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidateConfig:
    """模型验证配置"""

    validator_type: str = "cross_validation"
    validator_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """结果输出配置"""

    report_dir: str = "reports"
    model_dir: str = "model"


@dataclass
class ExperimentConfig:
    """完整实验配置"""

    data: DataConfig
    preprocess: PreprocessConfig
    feature: FeatureConfig
    split: SplitConfig
    train: TrainConfig
    validate: ValidateConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 创建各个配置部分
        data_config = DataConfig(**config_dict.get('data', {}))
        preprocess_config = PreprocessConfig(
            **config_dict.get('preprocess', {})
        )
        feature_config = FeatureConfig(**config_dict.get('feature', {}))
        split_config = SplitConfig(**config_dict.get('split', {}))
        train_config = TrainConfig(**config_dict.get('train', {}))
        validate_config = ValidateConfig(**config_dict.get('validate', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))

        return cls(
            data=data_config,
            preprocess=preprocess_config,
            feature=feature_config,
            split=split_config,
            train=train_config,
            validate=validate_config,
            output=output_config,
        )

    def validate_config(self) -> bool:
        """验证配置的有效性"""
        # 数据加载器
        valid_loaders = ["full_loader", "lazy_loader", "batch_loader"]
        if self.data.data_loader not in valid_loaders:
            raise ValueError(
                f"Invalid data loader type. Must be one of {valid_loaders}"
            )

        # 特征提取器
        valid_extractors = ["manual", "deep_learning"]
        if self.feature.feature_extractor not in valid_extractors:
            raise ValueError(
                f"Invalid feature extractor type. Must be one of {valid_extractors}"
            )

        # 特征提取器窗口参数
        if self.feature.window_size <= 0 or self.feature.overlap < 0:
            raise ValueError(
                "Window size must be positive and overlap must be non-negative"
            )
        if self.feature.overlap >= self.feature.window_size:
            raise ValueError("Overlap must be smaller than window size")

        # 传感器采样率
        if self.feature.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")

        # 数据集分割器
        splitters = ["train_test_split", "train_val_test_split"]
        if self.split.split_type not in splitters:
            raise ValueError(
                "Invalid split type. Must be 'train_test_split' or 'train_val_test_split'"
            )

        # 回归模型类型与参数

        return True


def load_config(config_type: str, yaml_path: str) -> ExperimentConfig | None:
    """加载配置的便捷函数"""
    if config_type == "experiment":
        config = ExperimentConfig.from_yaml(yaml_path)
        try:
            config.validate_config()
        except ValueError as e:
            print(f"Configuration validation failed: {e}")
        else:
            return config
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # 测试配置加载
    config = load_config(
        "experiment", "src/config/data_config/experiment1.yaml"
    )
    print(f"Data loader: {config.data.data_loader}")
    print(f"Feature extractor: {config.feature.feature_extractor}")
    print(f"Window size: {config.feature.window_size}")
