from typing import Dict, Any
from experiments.utils.feature_selection import (
    select_features,
    prepare_feature_sets,
)
from src.pipeline.pipeline import StandardMLPipeline


class FeatureAblationStudy:
    """特征消融实验"""

    def __init__(self, pipeline: StandardMLPipeline, config: Dict[str, Any]):
        self.pipeline = pipeline
        self.config = config

    def run(
        self, features_data: Dict[str, Any], ablation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行消融实验"""
        results = {}
        feature_sets = prepare_feature_sets(features_data, ablation_config)

        for set_name, feature_set in feature_sets.items():
            # 使用不同特征集训练和评估模型
            results[set_name] = self._evaluate_feature_set(feature_set)

        return results
