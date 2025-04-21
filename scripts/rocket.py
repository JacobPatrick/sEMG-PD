import os, sys
import numpy as np
from sklearn.model_selection import train_test_split

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curr_path + "/..")

from src.core.factories import (
    DataLoaderFactory,
    FeatureExtractorFactory,
    ModelTrainerFactory,
)

from src.pipeline.data.full_data_loader import FullDataLoader
from src.pipeline.feature.minirocket_feature_extractor import (
    MiniRocketFeatureExtractor,
)
from src.pipeline.classification.svc import SVM

from src.config.config import (
    ExperimentConfig,
    DataConfig,
    FeatureConfig,
    TrainConfig,
    OutputConfig,
)

# from src.utils.save_model import save_model
from src.utils.visualization import plot_confusion_matrix
from src.utils.model_evaluation import evaluate_classification


class ROCKETGaitPipeline:
    def __init__(
        self,
        data_loader_factory,
        feature_extractor_factory,
        model_trainer_factory,
    ):
        self.data_loader_factory = data_loader_factory
        self.feature_extractor_factory = feature_extractor_factory
        self.model_trainer_factory = model_trainer_factory

    def run(self, config: ExperimentConfig):
        print("导入数据...")
        data = self.load_data(config.data)
        print("处理原始数据格式...")
        # 筛选出步态相关的原始数据和标签
        gait_raw_list, gait_label_list = [], []

        for subject_id, subject_data in data["raw"].items():

            gait_label_list.append(data["labels"][subject_id][17])

            for exp_name, exp_df in subject_data.items():
                if exp_name == "gait":
                    exp_df = exp_df.iloc[
                        0:30000, 1:
                    ]  # 原始数据长度对齐，并去除时间列
                    gait_raw_list.append(np.array(exp_df).T)
                    continue

        # 原始数据和标签拼接成 X, y 格式
        X = np.stack(gait_raw_list, axis=0)  # (11, 8, 30000)
        X = X.astype(np.float32)  # 转为 float32 类型以适应 rocket
        y = np.array(gait_label_list)  # (11,)

        # 特征提取
        print("提取特征...")
        X = self.extract_features(X, config.feature)

        # 分割训练集和测试集
        # TODO: 样本数量最少的类别只有 1 个样本，需要考虑如何处理
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,  # stratify=gait_label
        )

        # 训练模型
        print("训练模型...")
        train_results = []
        models = self.train_model((X_train, y_train), config.train)
        # results = self.validate_model(models, (X_train, y_train), config.train)
        # train_results.append(results)
        y_train, pred_train = self.validate_model(
            models, (X_train, y_train), config.train
        )
        results = evaluate_classification(y_train, pred_train)
        train_results.append(results)
        plot_confusion_matrix(
            y_train,
            pred_train,
            [0, 1, 2, 3, 4],
            "训练集混淆矩阵",
            save_path="reports/20250421-minirocket/train_confusion_matrix.png",
        )

        # 测试模型
        print("测试模型...")
        test_results = []
        # results = self.validate_model(models, (X_test, y_test), config.train)
        # test_results.append(results)
        y_test, pred_test = self.validate_model(
            models, (X_test, y_test), config.train
        )
        results = evaluate_classification(y_test, pred_test)
        test_results.append(results)
        plot_confusion_matrix(
            y_test,
            pred_test,
            [0, 1, 2, 3, 4],
            "测试集混淆矩阵",
            save_path="reports/20250421-minirocket/test_confusion_matrix.png",
        )

        # 保存结果
        results = {
            "train_results": train_results,
            "test_results": test_results,
            "final_models": models,
        }
        # self.save_results(results, config.output)

        return results

    def load_data(self, config: DataConfig):
        loader_type = config.data_loader
        loader = self.data_loader_factory.create(loader_type)
        return loader.load(config)

    def extract_features(self, data, config: FeatureConfig):
        """特征提取"""
        extractor_type = config.feature_extractor
        extractor = self.feature_extractor_factory.create(extractor_type)
        return extractor.extract(data, config)

    def train_model(self, features, config: TrainConfig):
        """模型训练"""
        trainer_type = config.model_type
        trainer = self.model_trainer_factory.create(trainer_type)
        return trainer.fit(features)

    def validate_model(self, models, features, config: TrainConfig):
        """模型验证"""
        trainer_type = config.model_type
        trainer = self.model_trainer_factory.create(trainer_type, models=models)
        return trainer.predict(features)

    def save_results(self, results, config: OutputConfig) -> None:
        """保存结果"""

        # # 保存在测试集上训练得到的模型
        # model = results["final_models"]
        # save_model(model, config.model_dir, "ROCKETGait.joblib")

        # 保存交叉验证结果
        cv_results = results["cv_results"]
        try:
            with open(config.report_dir + "/cv_results.txt", "w") as f:
                for i, result in enumerate(cv_results):
                    f.write(f"Fold {i+1} results: {result}\n")
        except Exception as e:
            print(f"Error saving CV results: {e}")

        # 保存测试集结果
        test_results = results["test_results"]
        try:
            with open(config.report_dir + "/test_results.txt", "w") as f:
                f.write(f"Test results: {test_results}")
        except Exception as e:
            print(f"Error saving test dataset results: {e}")


def setup_factories():
    # 数据加载器工厂
    data_loader_factory = DataLoaderFactory()
    data_loader_factory.register("full_loader", FullDataLoader)

    # 特征提取器工厂
    feature_extractor_factory = FeatureExtractorFactory()
    feature_extractor_factory.register("minirocket", MiniRocketFeatureExtractor)

    # 模型训练工厂
    model_trainer_factory = ModelTrainerFactory()
    model_trainer_factory.register("svm", SVM)

    return {
        "data_loader_factory": data_loader_factory,
        "feature_extractor_factory": feature_extractor_factory,
        "model_trainer_factory": model_trainer_factory,
    }


def load_config(config_type: str, yaml_path: str) -> ExperimentConfig | None:
    """加载配置的便捷函数"""
    if config_type == "experiment":
        config = ExperimentConfig.from_yaml(yaml_path)
        return config
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    # 加载配置文件
    config = load_config("experiment", "src/config/ROCKETGait.yaml")

    # 初始化工厂
    factories = setup_factories()

    # 创建流水线
    pipeline = ROCKETGaitPipeline(**factories)

    # 运行流水线，传入完整配置
    results = pipeline.run(config)

    print(results)
