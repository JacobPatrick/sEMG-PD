import os
import pandas as pd
from src.pipeline.feature.manual_feature_extractor import ManualFeatureExtractor
from src.interfaces.feature import FeatureExtractor
import numpy as np


class TestManualFeatureExtractor:
    """手工特征提取器测试"""

    def test_extractor_initialization(self):
        """测试特征提取器初始化"""
        extractor = ManualFeatureExtractor()
        assert isinstance(extractor, FeatureExtractor)
        assert isinstance(extractor, ManualFeatureExtractor)

    def test_features_data_structure(self, mock_preprocessed_data, mock_experiment_config):
        """测试特征数据结构"""
        extractor = ManualFeatureExtractor()
        
        # 验证输入数据的结构
        assert "raw" in mock_preprocessed_data
        assert "sub-1" in mock_preprocessed_data["raw"]
        assert "sit" in mock_preprocessed_data["raw"]["sub-1"]
        df = mock_preprocessed_data["raw"]["sub-1"]["sit"]
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        
        # 提取特征
        features = extractor.extract(mock_preprocessed_data, mock_experiment_config)
        
        # 验证输出数据结构
        assert isinstance(features, dict)
        assert "features" in features
        assert "sub-1" in features["features"]
        assert "sit" in features["features"]["sub-1"]
        assert "windows" in features["features"]["sub-1"]["sit"]
        assert "window_0" in features["features"]["sub-1"]["sit"]["windows"]
        assert "channel1" in features["features"]["sub-1"]["sit"]["windows"]["window_0"]
        
        # 验证特征值
        window_features = features["features"]["sub-1"]["sit"]["windows"]["window_0"]["channel1"]
        assert "mav" in window_features
        assert "rms" in window_features
        assert isinstance(window_features["mav"], float)

    def test_time_domain_features(self):
        """测试时域特征提取"""
        extractor = ManualFeatureExtractor()
        # 使用已知信号测试
        window_data = np.sin(2 * np.pi * 10 * np.arange(0, 0.1, 0.0005))
        features = extractor._extract_window_features(window_data, 2000)
        
        # 可以精确验证特征值
        assert np.isclose(features["mav"], 0.6366, rtol=1e-4)
        assert np.isclose(features["rms"], 0.7071, rtol=1e-4)

    def test_frequency_domain_features(self):
        """测试频域特征提取"""
        extractor = ManualFeatureExtractor()
        # 使用纯正弦波，便于验证频域特征
        t = np.arange(0, 1, 0.0005)
        signal = np.sin(2 * np.pi * 10 * t)  # 10Hz正弦波
        features = extractor._extract_window_features(signal, 2000)
        
        assert np.isclose(features["mnf"], 10.0, rtol=1e-2)
