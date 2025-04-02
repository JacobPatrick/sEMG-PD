from abc import ABC, abstractmethod
from typing import Dict, Any


class Preprocessor(ABC):
    """预处理器接口"""

    @abstractmethod
    def process(self, data: Any, config: Dict[str, Any]) -> Any:
        """
        对数据进行预处理

        Args:
            data: 输入数据
            config: 配置参数

        Returns:
            处理后的数据
        """
        pass
