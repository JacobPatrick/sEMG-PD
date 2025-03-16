import os
import pandas as pd
import pytest
import numpy as np
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """创建测试数据目录"""
    data_dir = Path("tests/test_data")

    # 创建目录结构
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # 创建受试者目录
    for i in range(1, 3):
        subject_dir = data_dir / f"sub-{i}"
        subject_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


@pytest.fixture(scope="session")
def mock_timeseries_data(test_data_dir):
    """创建模拟时序数据"""
    # 创建随机时序数据
    for i in range(1, 3):
        subject_dir = test_data_dir / f"sub-{i}"

        # 第一个受试者有sit和motion1
        if i == 1:
            experiments = ["sit", "motion1"]
        # 第二个受试者有sit和gait
        else:
            experiments = ["sit", "gait"]

        # 为每个实验创建CSV文件
        for exp in experiments:
            # 创建带有多个通道的时序数据
            df = pd.DataFrame(
                {
                    "time": np.arange(0, 1, 0.01),
                    "channel1": np.random.randn(100),
                    "channel2": np.random.randn(100),
                    "channel3": np.random.randn(100),
                    "channel4": np.random.randn(100),
                    "channel5": np.random.randn(100),
                    "channel6": np.random.randn(100),
                    "channel7": np.random.randn(100),
                    "channel8": np.random.randn(100),
                }
            )

            # 保存到CSV
            df.to_csv(subject_dir / f"{exp}.csv", index=False)

    return test_data_dir


@pytest.fixture(scope="session")
def mock_labels_data(test_data_dir):
    """创建模拟标签数据"""
    # 创建标签数据
    labels_df = pd.DataFrame(
        {
            "ID": [1, 2],
            "age": [65, 72],
            "gender": ["M", "F"],
            "score": [42.5, 35.8],
        }
    )

    # 保存到CSV
    labels_file = test_data_dir / "labels.csv"
    labels_df.to_csv(labels_file, index=False)

    return labels_file


@pytest.fixture(scope="session")
def mock_config(test_data_dir, mock_timeseries_data, mock_labels_data):
    """创建模拟配置对象"""

    # 创建一个配置对象（可以是简单的命名空间或字典对象）
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return Config(
        data_dir=str(test_data_dir),
        labels_file=str(mock_labels_data),
        data_loader="full_loader",
    )
