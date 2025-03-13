from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import yaml
from pathlib import Path


class ConfigBase(ABC):
    """配置基类，定义配置类的基本接口"""

    @classmethod
    @abstractmethod
    def __post_init__(self):
        """配置初始化后处理"""
        pass

    @abstractmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML文件加载配置"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """验证配置是否有效"""
        pass


@dataclass
class DataConfig(ConfigBase):
    """数据处理相关配置"""

    raw_data_dir: str = "raw/"
    skiprows: int = 2
    skipcols: List[int] = field(default_factory=list)
    header: int = 3
    dtype: Dict[int, str] = field(default_factory=lambda: {2: 'str'})
    sensor_config: List[str] = field(default_factory=list)
    data_modal: Dict[str, List[int]] = field(init=False)

    def __post_init__(self):
        """配置初始化后处理"""
        self.skiprows += 1
        self.skipcols = [col - 1 for col in self.skipcols]

        data_modal = []
        for i, sensor in enumerate(self.sensor_config):
            if sensor == 'emg':
                data_modal.append('emg')
            elif sensor == 'imu':
                data_modal.extend(
                    [
                        'acc',
                        'acc',
                        'acc',
                        'gyro',
                        'gyro',
                        'gyro',
                        'mag',
                        'mag',
                        'mag',
                    ]
                )
        self.data_modal = {'emg': [], 'acc': [], 'gyro': [], 'mag': []}
        for i, modal in enumerate(data_modal):
            self.data_modal[modal].append(i + 1)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DataConfig':
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证数据配置的有效性"""
        if not Path(self.raw_data_dir).exists():
            raise ValueError(
                f"Data directory {self.raw_data_dir} does not exist"
            )
        if self.skiprows < 0 or self.header < 0:
            raise ValueError("skiprows and header must be non-negative")
        return True


@dataclass
class FeatureConfig(ConfigBase):
    """特征提取相关配置"""

    # TODO: 特征提取相关配置
    pass


@dataclass
class ModelConfig(ConfigBase):
    """模型训练相关配置"""

    model_type: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    train_params: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = field(default=42)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证模型配置的有效性"""
        valid_model_types = ['svm', 'rf', 'lstm']  # 可扩展的模型类型列表
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model type. Must be one of {valid_model_types}"
            )
        return True


class ConfigFactory:
    """配置工厂类，用于创建不同类型的配置对象"""

    _config_types = {'data': DataConfig, 'model': ModelConfig}

    @classmethod
    def create_config(cls, config_type: str, yaml_path: str) -> ConfigBase:
        """
        创建配置对象
        :param config_type: 配置类型 ('data' 或 'model')
        :param yaml_path: YAML配置文件路径
        :return: 配置对象
        """
        if config_type not in cls._config_types:
            raise ValueError(f"Unknown config type: {config_type}")

        config_class = cls._config_types[config_type]
        config = config_class.from_yaml(yaml_path)
        config.validate()
        return config


# 使用示例
def load_config(config_type: str, yaml_path: str) -> ConfigBase:
    """加载配置的便捷函数"""
    return ConfigFactory.create_config(config_type, yaml_path)


if __name__ == "__main__":
    data_config = load_config('data', 'src/config/data_config/new_raw.yaml')
    print(data_config)
