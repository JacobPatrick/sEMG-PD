from abc import ABC, abstractmethod
from typing import Dict, Any


class DataLoader(ABC):
    """数据加载器接口"""

    @abstractmethod
    def load(self, config: Dict[str, Any]) -> Any:
        """加载数据"""
        pass
