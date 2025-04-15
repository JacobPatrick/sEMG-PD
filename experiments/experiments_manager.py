# not implemented
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, KFold
from src.config.config import ExperimentConfig


@dataclass
class ExperimentResult:
    """实验结果类"""

    model_name: str
    params: Dict[str, Any]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    test_score: float


class ExperimentManager:
    def __init__(
        self,
        pipeline_factory,
        test_size: float = 0.2,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.pipeline_factory = pipeline_factory
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state
        self.results: List[ExperimentResult] = []

    def run_experiment(
        self, model_name: str, config: ExperimentConfig, data: Any
    ) -> ExperimentResult:
        """运行单个实验（包含交叉验证）"""
        # 分割测试集
        dev_data, test_data = self._split_test_data(data)

        # 进行交叉验证
        cv_scores = []
        kf = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        for fold, (train_idx, val_idx) in enumerate(kf.split(dev_data)):
            # 获取当前fold的训练集和验证集
            train_data = self._get_subset_data(dev_data, train_idx)
            val_data = self._get_subset_data(dev_data, val_idx)

            # 创建新的pipeline实例
            pipeline = self.pipeline_factory.create(model_name)

            # 训练和验证
            fold_score = self._run_fold(pipeline, config, train_data, val_data)
            cv_scores.append(fold_score)

        # 在完整开发集上训练最终模型，在测试集上评估
        final_pipeline = self.pipeline_factory.create(model_name)
        test_score = self._run_final_evaluation(
            final_pipeline, config, dev_data, test_data
        )

        # 记录结果
        result = ExperimentResult(
            model_name=model_name,
            params=config.to_dict(),
            cv_scores=cv_scores,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            test_score=test_score,
        )
        self.results.append(result)
        return result

    def run_grid_search(
        self,
        model_name: str,
        param_grid: Dict[str, List[Any]],
        base_config: ExperimentConfig,
        data: Any,
    ) -> List[ExperimentResult]:
        """网格搜索超参数"""
        results = []
        for params in self._generate_param_combinations(param_grid):
            config = self._update_config(base_config, params)
            result = self.run_experiment(model_name, config, data)
            results.append(result)
        return results

    def save_results(self, output_dir: str):
        """保存实验结果"""
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump([result.__dict__ for result in self.results], f, indent=2)
