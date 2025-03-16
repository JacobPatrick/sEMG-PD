from src.interfaces.data import DataLoader
import pandas as pd
import os
import glob
from typing import Dict, Any, List


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
                    "sub-1": {...},
                    "sub-2": {...},
                    ...
                }
            }
        """
        # 获取数据目录
        data_dir = config.data_dir if hasattr(config, "data_dir") else "raw"

        # 获取所有受试者目录
        subject_dirs = self._get_subject_dirs(data_dir)

        # 加载所有受试者的实验数据
        raw_data = self._load_all_subjects_data(subject_dirs)

        # 加载标签数据
        labels_file = (
            config.labels_file
            if hasattr(config, "labels_file")
            else os.path.join(data_dir, "labels.csv")
        )
        labels_data = self._load_labels(labels_file)

        # 返回完整数据字典
        return {"raw": raw_data, "labels": labels_data}

    def _get_subject_dirs(self, data_dir: str) -> List[str]:
        """获取所有受试者目录路径"""
        # 假设受试者目录以"sub-"开头
        return [
            d
            for d in glob.glob(os.path.join(data_dir, "sub-*"))
            if os.path.isdir(d)
        ]

    def _load_all_subjects_data(
        self, subject_dirs: List[str]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """加载所有受试者的所有实验数据"""
        all_subjects_data = {}

        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            subject_data = self._load_subject_data(subject_dir)
            all_subjects_data[subject_id] = subject_data

        return all_subjects_data

    def _load_subject_data(self, subject_dir: str) -> Dict[str, pd.DataFrame]:
        """加载单个受试者的所有实验数据"""
        subject_data = {}

        csv_files = glob.glob(os.path.join(subject_dir, "*.csv"))

        for csv_file in csv_files:
            experiment_name = os.path.splitext(os.path.basename(csv_file))[0]
            df = pd.read_csv(csv_file)
            subject_data[experiment_name] = df

        return subject_data

    def _load_labels(self, labels_file: str) -> Dict[str, Dict]:
        """加载标签数据"""
        if not os.path.exists(labels_file):
            print(f"警告: 标签文件 {labels_file} 不存在")
            return {}

        labels_df = pd.read_csv(labels_file)
        labels_data = {}

        for _, row in labels_df.iterrows():
            subject_id = f"sub-{row['ID']}"
            labels_data[subject_id] = row.to_dict()

        return labels_data
