from typing import Dict, Any
from interfaces.preprocess import Preprocessor


class PassThroughPreprocessor(Preprocessor):
    """直接传递数据的预处理器，不做任何处理"""

    def process(self, data: Any, config: Dict[str, Any]) -> Any:
        """
        直接返回输入数据
        
        Args:
            data: 输入数据
            config: 配置参数
            
        Returns:
            原始输入数据
        """
        return data 