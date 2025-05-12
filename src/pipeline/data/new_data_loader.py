from src.interfaces.data import DataLoader
import numpy as np
from typing import Tuple


class NewDataLoader(DataLoader):
    """新数据加载器：直接加载 np.ndarray 格式的数据"""

    def load(self, config) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据集

        Args:
            config: 配置参数，至少包含:
                - data_dir: 原始数据根目录

        Returns:
            包含数据和标签的元组 (data, labels)
        """
        # 获取数据目录
        data_dir = config.data_dir if hasattr(config, "data_dir") else "data/"
        data_name = config.data_name if hasattr(config, "data_name") else ""

        # 加载数据和标签
        dataset = np.load(data_dir + data_name)
        data = dataset["data"]
        labels = dataset["labels"]

        return data, labels
