import numpy as np
from typing import Tuple, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from src.interfaces.classification import Classification
from src.utils.model_evaluation import evaluate_classification
from src.utils.visualization import plot_confusion_matrix
from src.utils.calc import calc_mean_and_std


class LDAC(Classification):
    def __init__(self, models=[]):
        self.models = models

    def fit(self, data: Tuple) -> List:
        X, y = data
        self.models = []  # 清空模型列表以避免重复训练

        try:
            for i in range(y.shape[1]):  # 遍历每个输出维度
                unique_labels = np.unique(y[:, i])
                if unique_labels.size < 2:  # 如果标签只有一个类别，则跳过
                    self.models.append(FixedOutputLDAC(unique_labels[0]))
                    continue
                model = self._train_with_cv(
                    X,
                    y[:, i],
                    n_components=min(50, X.shape[1], len(unique_labels) - 1),
                )
                self.models.append(model)
        except IndexError:
            unique_labels = np.unique(y)
            if unique_labels.size < 2:  # 如果标签只有一个类别，则跳过
                self.models.append(FixedOutputLDAC(unique_labels[0]))
            model = self._train_with_cv(
                X, y, n_components=min(50, X.shape[1], len(unique_labels) - 1)
            )
            self.models.append(model)

        return self.models

    def _train_with_cv(self, X, y, n_components, n_splits=5):
        """使用交叉验证训练LDA模型

        参数:
            X: 特征数据
            y: 标签数据
            n_components: LDA组件数量
            n_splits: 交叉验证折数

        返回:
            训练好的最佳LDA模型
        """
        from sklearn.base import clone

        # 初始化交叉验证分割器
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 初始化用于存储最佳模型的变量
        best_score = 0.0
        best_model = None

        folds_scores = []

        print(f"开始{n_splits}折交叉验证训练...")

        # 执行交叉验证
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 创建并训练模型
            model = LDA(n_components=n_components)
            model.fit(X_train, y_train)

            # 评估模型
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)

            print(f"折 {fold_idx + 1}/{n_splits} - 准确率: {accuracy:.4f}")

            folds_scores.append(accuracy)

            # 更新最佳模型
            if accuracy > best_score:
                best_score = accuracy
                best_model = clone(model)  # 创建模型的深拷贝
                best_model.fit(X_train, y_train)  # 使用当前折的数据重新训练

        if best_model is None:
            # 如果所有折的得分都是0，使用完整数据集训练一个模型
            best_model = LDA(n_components=n_components)
            best_model.fit(X, y)

        print(f"交叉验证完成，最佳验证准确率: {best_score:.4f}")

        mean, std = calc_mean_and_std(folds_scores)
        print(f"所有折准确率的均值: {mean:.4f}, 标准差: {std:.4f}")

        # 使用全部数据重新训练最佳模型配置
        final_model = clone(best_model)
        final_model.fit(X, y)
        return final_model

    def predict_proba(self, data: Tuple):
        X, y = data
        proba_predictions = []
        for idx, model in enumerate(self.models):
            pred_proba = model.predict_proba(X)
            complete_proba = np.zeros((y.shape[0], 5))
            # 补齐 5 种类型
            y_unique = np.unique(y[:, idx])
            for i in range(5):
                if i in y_unique:
                    complete_proba[:, i] = pred_proba[
                        :, np.where(y_unique == i)[0][0]
                    ]
                else:
                    complete_proba[:, i] = 0.0
            proba_predictions.append(complete_proba)

        return y, np.array(proba_predictions).T

    def predict(self, data: Tuple):
        X, y = data
        predictions = []
        for idx, model in enumerate(self.models):
            predictions.append(model.predict(X))

            # 绘制混淆矩阵
            plot_confusion_matrix(
                label_true=y[:, idx],
                label_pred=predictions[-1],
                classes=[0, 1, 2, 3, 4],
                save_path=f"reports/fig/final/gait_lda_test_confusion_matrix_{idx}.png",
            )

        return y, np.array(predictions).T


class FixedOutputLDAC:
    def __init__(self, fixed_output):
        self.fixed_output = fixed_output

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, 5))
        probas[:, self.fixed_output] = 1.0
        return probas

    def predict(self, X):
        return np.full((X.shape[0],), self.fixed_output)
