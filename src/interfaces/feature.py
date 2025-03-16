from abc import ABC, abstractmethod
from typing import Dict, Any


class FeatureExtractor(ABC):
    """特征提取器接口"""

    @abstractmethod
    def extract(self, data: Any, config: Dict[str, Any]) -> Any:
        """从数据中提取特征"""
        pass
