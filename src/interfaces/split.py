from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class Splitter(ABC):
    """数据集分割器接口"""

    @abstractmethod
    def train_test_split(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分割训练集/测试集"""
        pass

    @abstractmethod
    def train_val_test_split(self, data: Any) -> Dict[str, Tuple]:
        """在训练集内进一步获取交叉验证的分割"""
        pass
