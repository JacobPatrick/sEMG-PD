from typing import Any, List, Dict, Tuple
from sklearn.model_selection import train_test_split, KFold
from src.interfaces.split import Splitter
import numpy as np


class DataSplitter(Splitter):
    """数据集分割器"""

    def train_test_split(self, data: Any) -> Dict[str, Tuple]:
        """简单的训练集/测试集分割"""
        # 将字典格式的数据转为特征矩阵和标签矩阵
        X, y = self._prepare_data(data, dimension=4)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return dict(train=(X_train, y_train), test=(X_test, y_test))

    def train_val_test_split(self, data: Any) -> Dict[str, Tuple]:
        """训练集/验证集/测试集分割"""
        # 1. 划分训练集和测试集
        train_test_split_data = self.train_test_split(data)
        train_data = train_test_split_data["train"]
        test_data = train_test_split_data["test"]

        # 2. 从训练集中划分验证集
        cv_splits = self._get_cv_splits(train_data, n_splits=5)

        return dict(train=train_data ,test=test_data, cv_splits=cv_splits)

    def _get_cv_splits(
        self, data: Tuple, n_splits: int = 5
    ) -> List[Dict[str, Tuple]]:
        """获取交叉验证的数据分割"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []
        for train_idx, val_idx in kf.split(data[1]):
            X_train, y_train = data[0][train_idx], data[1][train_idx]
            X_val, y_val = data[0][val_idx], data[1][val_idx]
            splits.append(dict(train=(X_train, y_train), val=(X_val, y_val)))
        return splits

    def _prepare_data(
        self, data: Dict[str, Any], dimension: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备数据

        Returns:
            X: shape (n_samples, n_features) 的特征矩阵
            y: shape (n_samples, n_dimensions) 的标签矩阵
        """
        # 区分特征字典和标签字典
        features_dict = data["features"]
        labels_dict = data["labels"]

        # 将特征字典转为特征矩阵
        features_list = []
        for _, subject_features in features_dict.items():
            feature_matrix = self._dict_to_mat(subject_features, dimension)
            feature_matrix_flattened = feature_matrix.reshape(1, -1)
            features_list.append(feature_matrix_flattened)

        # 所有样本特征拼接成特征矩阵
        X = np.vstack(features_list)

        # 提取标签向量
        y_list = []
        for _, subject_label in labels_dict.items():
            if isinstance(subject_label, (int, float)):
                # 单维度标签，转换为向量
                y_list.append([subject_label])
            else:
                # 多维度标签，已经是向量形式
                y_list.append(subject_label)

        y = np.vstack(y_list)

        return X, y

    def _dict_to_mat(
        self, dict_data: Dict[str, Any], dimension: int
    ) -> np.ndarray:
        """将多层字典转换为多维矩阵

        Args:
            dict_data: 多层嵌套字典
            dimension: 字典的嵌套深度（假设同一层所有子字典的嵌套深度相同）

        Returns:
            numpy数组
        """
        # 1. 为每一层创建键到索引的映射
        key_to_index = [{} for _ in range(dimension)]
        max_indices = [0] * dimension

        def map_keys_to_indices(current_dict, current_depth):
            """递归遍历字典，建立键到索引的映射"""
            # TODO: 字典深度不全相同？
            if isinstance(current_dict, dict):
                if current_depth == dimension - 1:
                    for key in current_dict.keys():
                        if key not in key_to_index[current_depth]:
                            key_to_index[current_depth][key] = max_indices[
                                current_depth
                            ]
                            max_indices[current_depth] += 1
                    return

                for key, sub_dict in current_dict.items():
                    if key not in key_to_index[current_depth]:
                        key_to_index[current_depth][key] = max_indices[
                            current_depth
                        ]
                        max_indices[current_depth] += 1
                    map_keys_to_indices(sub_dict, current_depth + 1)

        # 建立映射关系
        map_keys_to_indices(dict_data, 0)

        # 2. 创建矩阵并填充数据
        matrix = np.zeros(tuple(max_indices))

        def fill_matrix(
            current_dict: Dict, current_depth: int, index_list: List[Any]
        ):
            """递归填充矩阵"""
            # TODO: 字典深度不全相同？
            if isinstance(current_dict, dict):
                if current_depth == dimension - 1:
                    for key, value in current_dict.items():
                        idx = key_to_index[current_depth][key]
                        index_list[current_depth] = idx
                        matrix[tuple(index_list)] = value
                    return

                for key, sub_dict in current_dict.items():
                    idx = key_to_index[current_depth][key]
                    index_list[current_depth] = idx
                    fill_matrix(sub_dict, current_depth + 1, index_list)

        fill_matrix(dict_data, 0, [0] * dimension)

        return matrix
