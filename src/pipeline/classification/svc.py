import numpy as np
from typing import Tuple, List
from sklearn.svm import SVC
from src.interfaces.classification import Classification


class SVM(Classification):
    """SVM分类模型实现"""

    def __init__(self, models=[]):
        self.models = models  # 每个输出维度对应一个SVC模型

    def fit(self, data: Tuple) -> List[SVC]:
        """训练模型

        Args:
            data: 特征和标签数据，格式为 (X, y)，其中 X 是特征矩阵，y 是标签矩阵
        """

        X, y = data

        # 为每个输出维度训练一个SVC模型
        self.models = []
        for i in range(y.shape[1]):  # 遍历每个输出维度
            if np.unique(y[:, i]).size < 2:  # 如果标签只有一个类别，则跳过
                self.models.append(FixedOutputSVC(y[:, i][0]))
                continue
            model = SVC(kernel='rbf', C=1.0, gamma=0.1)
            model.fit(X, y[:, i])
            self.models.append(model)

        return self.models

    def predict(self, data: Tuple) -> np.ndarray:
        """预测量表中每个维度的得分

        Args:
            data: 特征和标签数据，格式为 (X, y)，其中 X 是特征矩阵，y 是标签矩阵

        Returns:
            predictions: shape (n_samples, n_dimensions) 的预测结果
        """
        X, y = data
        predictions = []
        confusion_matrices = []

        # 使用每个模型预测对应维度的输出，并计算混淆矩阵
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

            # 计算混淆矩阵
            confusion_matrix = np.zeros((5, 5))
            for true_label, pred_label in zip(y, pred):
                confusion_matrix[true_label, pred_label] += 1
            
            confusion_matrices.append(confusion_matrix)

        # 计算平均混淆矩阵
        average_confusion_matrix = np.mean(confusion_matrices, axis=0)

        # 将各维度的预测结果组合
        # return np.column_stack(predictions)
        return average_confusion_matrix


class FixedOutputSVC(SVC):
    """输出给定值的SVC模型，应对标签仅有单一类别的情况"""

    def __init__(self, fixed_output, **kwargs):
        super().__init__(**kwargs)
        self.fixed_output = fixed_output

    def predict(self, X):
        return np.full((X.shape[0],), self.fixed_output)
