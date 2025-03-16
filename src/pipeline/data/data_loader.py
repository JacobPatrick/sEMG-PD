import os
import pandas as pd
from typing import Dict, Any, List, Generator
from src.interfaces.data import DataLoader


# TODO: to be implemented
class LazyDataLoader(DataLoader):
    """懒加载器 - 按需加载数据"""

    def load(self, config) -> Dict[str, Any]:
        """返回数据生成器而非全部数据"""
        data_paths = self._get_data_paths(config)
        metadata = self._load_metadata(config)

        return {
            'data_generator': self._data_generator(data_paths, config),
            'metadata': metadata,
        }

    def _data_generator(self, paths: List[str], config) -> Generator:
        """生成器函数，按需加载每个文件"""
        for path in paths:
            # 加载单个文件
            data = pd.read_csv(path)
            # 可能的初步处理
            yield data

    def _get_data_paths(self, config):
        # 根据配置获取所有数据文件路径
        # ...
        raise NotImplementedError

    def _load_metadata(self, config):
        # 加载量表得分等元数据
        # ...
        raise NotImplementedError


# TODO: to be implemented
class BatchDataLoader(DataLoader):
    """分批数据加载器 - 按受试者或实验分批加载"""

    def __init__(self, batch_size: int = 1):
        """
        Args:
            batch_size: 每批加载的受试者数量
        """
        self.batch_size = batch_size

    def load(self, config) -> Dict[str, Any]:
        """返回分批加载的数据生成器"""
        subject_ids = self._get_subject_ids(config)
        metadata = self._load_metadata(config)

        return {
            'batch_generator': self._batch_generator(subject_ids, config),
            'metadata': metadata,
        }

    def _batch_generator(self, subject_ids, config):
        """按批次生成数据"""
        for i in range(0, len(subject_ids), self.batch_size):
            batch_ids = subject_ids[i : i + self.batch_size]
            batch_data = {}

            for subject_id in batch_ids:
                subject_data = self._load_subject_data(subject_id, config)
                batch_data[subject_id] = subject_data

            yield batch_data


# class CrossValidationDataLoader(DataLoader):
#     """交叉验证数据加载器 - 用于模型验证"""

#     def __init__(self, n_splits: int = 5):
#         """
#         Args:
#             n_splits: 交叉验证的折数
#         """
#         self.n_splits = n_splits

#     def load(self, config):
#         """返回交叉验证数据生成器"""
#         data = super().load(config)

#         # 分割数据集
#         if config.get('split_data', False):
#             train_ids, val_ids, test_ids = self._split_subject_ids(
#                 list(data['labels'].keys()),
#                 config.train_ratio,
#                 config.val_ratio
#             )

#             data['splits'] = {
#                 'train': train_ids,
#                 'val': val_ids,
#                 'test': test_ids
#             }

#         return data
