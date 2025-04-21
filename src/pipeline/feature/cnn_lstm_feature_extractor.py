from typing import Dict, Any
from src.interfaces.feature import FeatureExtractor
from src.config.config import FeatureConfig


class CnnLstmFeatureExtractor(FeatureExtractor):
    """深度学习特征提取器实现"""

    def extract(self, data: Any, config: FeatureConfig) -> Dict[str, Any]:
        """使用 CNN + LSTM 模型提取特征"""
        # TODO: 具体实现
        features_data = {}
        return features_data
