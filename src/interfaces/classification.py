from abc import ABC, abstractmethod
from typing import Tuple, List, Any


class Classification(ABC):
    """回归模型接口"""

    @abstractmethod
    def __init__(self, models) -> None:
        """初始化模型"""
        pass

    @abstractmethod
    def fit(self, data: Tuple) -> List[Any]:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, data: Tuple) -> Any:
        """预测"""
        pass
