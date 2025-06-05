import numpy as np
from src.interfaces.feature import FeatureExtractor
from src.config.config import FeatureConfig
from src.pipeline.feature.minirocket_multivariate import *
from typing import Tuple, Dict, List, Any


class MiniRocketFeatureExtractor(FeatureExtractor):
    def extract(
        self, dataset: Tuple[np.ndarray, np.ndarray], config: FeatureConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """特征提取"""
        data = dataset[0]
        labels = dataset[1]

        # 验证输入数据格式
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.dtype != np.float32:
            raise ValueError("Input data must be a numpy array of float32")

        # 训练模型
        (
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            biases,
        ) = fit(data, num_features=1176, max_dilations_per_kernel=16)

        # 数据转换
        features = transform(
            data,
            (
                num_channels_per_combination,
                channel_indices,
                dilations,
                num_features_per_dilation,
                biases,
            ),
        )
        print(f"features shape: {features.shape}")

        return features, labels
