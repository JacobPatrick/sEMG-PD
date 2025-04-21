import numpy as np
from tsai.models.ROCKET import *
from tsai.models.MINIROCKET import *
from tsai.models.MultiRocketPlus import *
from tsai.all import *
from typing import List, Tuple
from src.interfaces.classification import Classification

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV


class ROCKET(Classification):
    """ROCKET分类模型实现"""

    def __init__(self, models=[]):
        self.models = models  # 每个输出维度对应一个ROCKET模型

    def fit(self, data: Tuple) -> List:
        """训练模型

        Args:
            data: 特征和标签数据，格式为 (X, y)，其中 X 是特征矩阵，y 是标签矩阵
        """
        X, y = data
        self.models = []

        fname = 'Rocket'
        cls = RocketClassifier()
        cls.fit(X, y)
        self.models.append(cls)
        cls.save(fname)
        del cls

        # for i in range(y.shape[1]):
        #     if np.unique(y[:, i]).size < 2:  # 如果标签只有一个类别，则跳过
        #         self.models.append(FixedOutputClassifier(y[:, i][0]))
        #         continue

        #     fname = f'Rocket-{i}'
        #     cls = RocketClassifier()
        #     cls.fit(X, y)
        #     self.models.append(cls)
        #     cls.save(fname)
        #     del cls

        return self.models

    def predict(self, data: Tuple) -> np.array:
        """预测量表中每个维度的得分"""
        X, y = data
        predictions = []
        confusion_matrix = np.zeros((5, 5))

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

            for true_label, pred_label in zip(y, pred):
                confusion_matrix[true_label, pred_label] += 1

        return np.array(confusion_matrix)


class CustomMiniROCKETClassifier(
    BaseEstimator, ClassifierMixin, TransformerMixin
):
    """自定义 MiniROCKETClassifier 分类器包装器，以满足 sklearn 的接口要求"""

    def __init__(self):
        self.minirocket = MiniRocketClassifier()

    def fit(self, X, y):
        self.minirocket.fit(X, y)
        return self

    def predict(self, X):
        return self.minirocket.predict(X)

    def transform(self, X):
        return X


class MiniROCKET(Classification):
    """miniROCKET分类模型实现"""

    def __init__(self, models=[]):
        self.models = models  # 每个输出维度对应一个ROCKET模型

    def fit(self, data: Tuple) -> List:
        """训练模型

        Args:
            data: 特征和标签数据，格式为 (X, y)，其中 X 是特征矩阵，y 是标签矩阵
        """
        X, y = data
        self.models = []

        # FIXME: 疑似 tsai 与 sklearn 接口不兼容
        fname = 'Rocket'
        cls = MiniRocketClassifier()
        cls.fit(X, y)
        self.models.append(cls)
        cls.save(fname)
        del cls

        # for i in range(y.shape[1]):
        #     if np.unique(y[:, i]).size < 2:  # 如果标签只有一个类别，则跳过
        #         self.models.append(FixedOutputClassifier(y[:, i][0]))
        #         continue

        #     fname = f'Rocket-{i}'
        #     cls = RocketClassifier()
        #     cls.fit(X, y)
        #     self.models.append(cls)
        #     cls.save(fname)
        #     del cls

        return self.models

    def predict(self, data: Tuple) -> np.array:
        """预测量表中每个维度的得分"""
        X, y = data
        predictions = []
        confusion_matrix = np.zeros((5, 5))

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

            for true_label, pred_label in zip(y, pred):
                confusion_matrix[true_label, pred_label] += 1

        return np.array(confusion_matrix)


class MultiROCKET(Classification):
    """multiROCKET分类模型实现"""

    def __init__(self, models=[]):
        self.models = models  # 每个输出维度对应一个ROCKET模型

    def fit(self, data: Tuple) -> List:
        """训练模型

        Args:
            data: 特征和标签数据，格式为 (X, y)，其中 X 是特征矩阵，y 是标签矩阵
        """
        X, y = data
        self.models = []

        # TODO: 更换成 Multi-ROCKET 模型
        fname = 'Rocket'
        cls = RocketClassifier()
        cls.fit(X, y)
        self.models.append(cls)
        cls.save(fname)
        del cls

        # for i in range(y.shape[1]):
        #     if np.unique(y[:, i]).size < 2:  # 如果标签只有一个类别，则跳过
        #         self.models.append(FixedOutputClassifier(y[:, i][0]))
        #         continue

        #     fname = f'Rocket-{i}'
        #     cls = RocketClassifier()
        #     cls.fit(X, y)
        #     self.models.append(cls)
        #     cls.save(fname)
        #     del cls

        return self.models

    def predict(self, data: Tuple) -> np.array:
        """预测量表中每个维度的得分"""
        X, y = data
        predictions = []
        confusion_matrix = np.zeros((5, 5))

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

            for true_label, pred_label in zip(y, pred):
                confusion_matrix[true_label, pred_label] += 1

        return np.array(confusion_matrix)


class FixedOutputClassifier:
    """固定输出分类器"""

    def __init__(self, fixed_output):
        self.fixed_output = fixed_output

    def predict(self, X):
        return np.full(X.shape[0], self.fixed_output)
