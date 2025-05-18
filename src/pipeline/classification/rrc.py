import numpy as np
from typing import Tuple, List
from sklearn.linear_model import RidgeClassifierCV
from src.interfaces.classification import Classification


class RRC(Classification):
    """岭回归分类器实现"""

    def __init__(self, models=[]):
        self.models = models  # 为测试集加载已经训练好的模型

    def fit(self, data: Tuple) -> List:
        X, y = data
        self.models = []  # 清空模型列表以避免重复训练

        try:
            for i in range(y.shape[1]):  # 遍历每个输出维度
                unique_labels = np.unique(y[:, i])
                if unique_labels.size < 2:  # 如果标签只有一个类别，则跳过
                    self.models.append(FixedOutputRRC(unique_labels[0]))
                    continue

                # 创建并训练岭回归分类器，使用交叉验证自动选择alpha值
                model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
                model.fit(X, y[:, i])
                self.models.append(model)

        except IndexError:  # 处理一维标签的情况
            unique_labels = np.unique(y)
            if unique_labels.size < 2:  # 如果标签只有一个类别
                self.models.append(FixedOutputRRC(unique_labels[0]))
            else:
                # 创建并训练岭回归分类器
                model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
                model.fit(X, y)
                self.models.append(model)

        return self.models

    def predict_proba(self, data: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """返回概率预测结果

        注意：RidgeClassifier没有内置的predict_proba方法，这里通过决策函数值转换为概率
        """
        X, y = data
        proba_predictions = []

        for model in self.models:
            if isinstance(model, FixedOutputRRC):  # 处理固定输出的情况
                proba_pred = model.predict_proba(X)
            else:
                # RidgeClassifier没有直接的predict_proba方法，需要从decision_function转换
                decision_values = model.decision_function(X)
                # 将决策函数值转换为概率（多类情况）
                if decision_values.ndim > 1:
                    # 使用softmax将决策值转换为概率
                    exp_scores = np.exp(
                        decision_values
                        - np.max(decision_values, axis=1, keepdims=True)
                    )
                    proba_pred = exp_scores / np.sum(
                        exp_scores, axis=1, keepdims=True
                    )
                else:
                    # 二分类情况
                    proba_pred = np.zeros((decision_values.shape[0], 2))
                    proba_pred[:, 1] = 1 / (1 + np.exp(-decision_values))
                    proba_pred[:, 0] = 1 - proba_pred[:, 1]

            proba_predictions.append(proba_pred)

        return y, np.array(proba_predictions).T

    def predict(self, data: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """返回预测类别"""
        X, y = data
        predictions = []

        for model in self.models:
            predictions.append(model.predict(X))

        return y, np.array(predictions).T


class FixedOutputRRC:
    """固定输出分类器，用于处理只有一个类别的情况"""

    def __init__(self, fixed_output):
        self.fixed_output = fixed_output

    def fit(self, X, y):
        """空实现，保持接口一致性"""
        return self

    def predict_proba(self, X):
        """返回概率预测结果"""
        n_samples = X.shape[0]
        # 动态确定概率数组大小，确保self.fixed_output是有效索引
        num_classes = max(5, self.fixed_output + 1)
        probas = np.zeros((n_samples, num_classes))
        probas[:, self.fixed_output] = 1.0
        return probas

    def predict(self, X):
        """对所有输入预测同一个类别"""
        return np.full((X.shape[0],), self.fixed_output)
