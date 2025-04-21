from src.interfaces.data import DataLoader
import pandas as pd
import os
import glob
import json
from typing import Dict, Tuple, List, Any


class FullDataLoader(DataLoader):
    """全量数据加载器 - 一次性加载所有数据"""

    def load(self, config) -> Dict[str, Any]:
        """
        加载所有受试者的所有实验数据和标签

        Args:
            config: 配置参数，至少包含:
                   - data_dir: 原始数据根目录
                   - labels_file: 标签文件路径

        Returns:
            包含所有数据的字典，格式为:
            {
                "raw": {
                    "sub-1": {
                        "sit": pd.DataFrame(...),
                        "motion1": pd.DataFrame(...),
                        ...
                    },
                    "sub-2": {...},
                    ...
                },
                "labels": {
                    "sub-1": [...],
                    "sub-2": [...],
                    ...
                }
            }
        """
        # 获取数据目录
        data_dir = (
            config.data_dir if hasattr(config, "data_dir") else "raw/grad-proj/"
        )

        # 获取所有受试者目录
        subject_dirs = self._get_subject_dirs(data_dir)

        # 加载所有受试者的实验数据
        raw_data = self._load_all_subjects_data(subject_dirs, config.data_range)

        # 加载标签数据
        labels_file = (
            config.labels_file
            if hasattr(config, "labels_file")
            else os.path.join(data_dir, "labels.csv")
        )
        labels_data = self._load_labels(data_dir, labels_file)

        # 返回完整数据字典
        return {"raw": raw_data, "labels": labels_data}

    def _get_subject_dirs(self, data_dir: str) -> List[str]:
        """获取所有受试者目录路径"""
        # 受试者目录为"sub-{id}"
        return [
            d
            for d in glob.glob(os.path.join(data_dir, "sub-*"))
            if os.path.isdir(d)
        ]

    def _load_all_subjects_data(
        self, subject_dirs: List[str], data_range: Tuple | None = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """加载所有受试者的所有实验数据"""
        all_subjects_data = {}

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            subject_data = self._load_subject_data(subject_dir, data_range)
            all_subjects_data[subject_id] = subject_data

        return all_subjects_data

    def _load_subject_data(
        self, subject_dir: str, data_range: Tuple | None
    ) -> Dict[str, pd.DataFrame]:
        """加载单个受试者的所有实验数据"""
        subject_data = {}

        csv_files = glob.glob(os.path.join(subject_dir, "*.csv"))

        for csv_file in csv_files:
            experiment_name = os.path.splitext(os.path.basename(csv_file))[0]
            df = pd.read_csv(csv_file, encoding="utf-8")
            if data_range:
                start, end = data_range
                subject_data[experiment_name] = df.iloc[start:end, :]
            else:
                subject_data[experiment_name] = df

        return subject_data

    def _load_labels(self, data_dir: str, labels_file: str) -> Dict[str, Dict]:
        """加载标签数据"""
        if not os.path.exists(os.path.join(data_dir, labels_file)):
            print(f"警告: 标签文件 {labels_file} 不存在")
            return {}

        labels_df = pd.read_csv(
            os.path.join(data_dir, labels_file), encoding="utf-8"
        )
        labels_data = {}

        for _, row in labels_df.iterrows():
            subject_id = f"sub-{row['ID']}"
            labels_data[subject_id] = row.to_dict()
            updrs_scores = json.loads(labels_data[subject_id]["UPDRS-III"])
            labels_data[subject_id] = _dict_values_to_list(updrs_scores)

        return labels_data


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
