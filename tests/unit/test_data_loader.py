import os
import pandas as pd
from src.pipeline.data.full_data_loader import FullDataLoader
from src.interfaces.data import DataLoader


class TestFullDataLoader:
    """FullDataLoader单元测试"""

    def test_loader_initialization(self):
        """测试加载器初始化"""
        loader = FullDataLoader()
        assert isinstance(loader, DataLoader)
        assert isinstance(loader, FullDataLoader)

    def test_get_subject_dirs(self, mock_config):
        """测试获取受试者目录"""
        loader = FullDataLoader()
        dirs = loader._get_subject_dirs(mock_config.data_dir)

        # 应该找到两个受试者目录
        assert len(dirs) == 2
        assert os.path.basename(dirs[0]) in ["sub-1", "sub-2"]
        assert os.path.basename(dirs[1]) in ["sub-1", "sub-2"]

    def test_load_subject_data(self, mock_config):
        """测试加载单个受试者数据"""
        loader = FullDataLoader()

        # 测试第一个受试者数据
        subject_dir = os.path.join(mock_config.data_dir, "sub-1")
        data = loader._load_subject_data(subject_dir)

        # 检查数据结构
        assert "sit" in data
        assert "motion1" in data
        assert isinstance(data["sit"], pd.DataFrame)
        assert "channel1" in data["sit"].columns

    def test_load_all_subjects_data(self, mock_config):
        """测试加载所有受试者数据"""
        loader = FullDataLoader()
        dirs = loader._get_subject_dirs(mock_config.data_dir)
        data = loader._load_all_subjects_data(dirs)

        # 检查数据结构
        assert "sub-1" in data
        assert "sub-2" in data
        assert "sit" in data["sub-1"]
        assert "gait" in data["sub-2"]

    def test_load_labels(self, mock_labels_data):
        """测试加载标签数据"""
        loader = FullDataLoader()
        labels = loader._load_labels(mock_labels_data)

        # 检查标签数据
        assert "sub-1" in labels
        assert "sub-2" in labels
        assert "age" in labels["sub-1"]
        assert labels["sub-1"]["age"] == 65

    def test_load_complete(self, mock_config):
        """测试完整加载流程"""
        loader = FullDataLoader()
        data = loader.load(mock_config)

        # 检查返回数据结构
        assert "raw" in data
        assert "labels" in data
        assert "sub-1" in data["raw"]
        assert "sub-2" in data["labels"]

    def test_missing_labels_file(self, mock_config):
        """测试标签文件缺失情况"""
        # 修改配置，指向不存在的文件
        mock_config.labels_file = "non_existent_file.csv"

        loader = FullDataLoader()
        data = loader.load(mock_config)

        # 应该有警告日志
        # assert "警告" in caplog.text
        # 标签字典应该为空
        assert data["labels"] == {}

    def test_custom_config_parameters(self):
        """测试自定义配置参数"""

        # 创建一个具有最小配置的对象
        class MinimalConfig:
            pass

        min_config = MinimalConfig()

        # 测试默认值处理
        loader = FullDataLoader()
        data = loader.load(min_config)

        # 应该使用默认值
        assert "raw" in data
        assert "labels" in data
