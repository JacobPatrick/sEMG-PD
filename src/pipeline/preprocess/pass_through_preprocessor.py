import json
from typing import Dict, List, Any
from interfaces.preprocess import Preprocessor


class PassThroughPreprocessor(Preprocessor):
    """直接传递实验数据，仅调整标签数据格式的预处理器"""

    def process(self, data: Any, config: Dict[str, Any]) -> Any:
        """
        直接返回输入数据

        Args:
            data: 输入数据
            config: 配置参数

        Returns:
            原始输入数据
        """
        # 调整标签数据格式，提取 UPDRS-III 量表得分并转为列表
        for subject_id, subject_labels in data["labels"].items():
            updrs_scores = json.loads(subject_labels["UPDRS-III"])
            data["labels"][subject_id] = _dict_values_to_list(updrs_scores)

        return data


def _dict_values_to_list(data: Dict[str, Any]) -> List[Any]:
    """将多层字典的值转为按顺序排列的列表"""

    updrs_scores_list = []

    def dict_values_to_list_recursive(data):
        """递归遍历字典，将所有值依次存入列表"""
        if not isinstance(data, dict):
            updrs_scores_list.append(data)
            return

        for _, value in data.items():
            dict_values_to_list_recursive(value)

    dict_values_to_list_recursive(data)

    return updrs_scores_list
