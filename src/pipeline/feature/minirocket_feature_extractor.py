import numpy as np
from src.interfaces.feature import FeatureExtractor
from src.config.config import FeatureConfig
from src.pipeline.feature.minirocket_multivariate import *


class MiniRocketFeatureExtractor(FeatureExtractor):
    def extract(self, X: np.ndarray, config: FeatureConfig) -> np.ndarray:
        """特征提取"""

        # 验证输入数据格式
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if X.dtype != np.float32:
            raise ValueError("Input data must be a numpy array of float32")

        # 训练模型
        (
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            biases,
        ) = fit(X, num_features=5000, max_dilations_per_kernel=16)

        # 数据转换
        features = transform(
            X,
            (
                num_channels_per_combination,
                channel_indices,
                dilations,
                num_features_per_dilation,
                biases,
            ),
        )

        return features
