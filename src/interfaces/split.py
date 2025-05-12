from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class Splitter(ABC):
    """数据集分割器接口"""

    @abstractmethod
    def split(self, data: Tuple) -> Dict[str, Tuple[Any]]:
        """划分数据集"""
        pass
