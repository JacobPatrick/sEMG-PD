from typing import Dict, Any
import pandas as pd
import numpy as np

def select_features(features_data: Dict[str, Any], feature_set_config: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置选择特征子集
    
    Args:
        features_data: 提取的所有特征
        feature_set_config: {
            "channels": ["channel_1", "channel_3"],  # 选择特定通道
            "features": ["mean", "std", "mav"],      # 选择特定特征
            "experiments": ["sit", "motion1"],       # 选择特定实验
        }
    """
    selected = {}
    for subject_id, subject_data in features_data["features"].items():
        selected[subject_id] = {}
        for exp_name in feature_set_config["experiments"]:
            if exp_name not in subject_data:
                continue
                
            selected[subject_id][exp_name] = {
                "windows": {}
            }
            for window_id, window_data in subject_data[exp_name]["windows"].items():
                selected[subject_id][exp_name]["windows"][window_id] = {
                    channel: {
                        feat: value 
                        for feat, value in feats.items()
                        if feat in feature_set_config["features"]
                    }
                    for channel, feats in window_data.items()
                    if channel in feature_set_config["channels"]
                }
    
    return selected


def prepare_feature_sets(features_data: Dict[str, Any], ablation_config: Dict[str, Any]) -> Dict[str, Any]:
    """准备不同的特征集合"""
    feature_sets = {}
    for set_name, config in ablation_config["feature_sets"].items():
        feature_sets[set_name] = select_features(features_data, config)
    return feature_sets
