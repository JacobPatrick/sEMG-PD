import numpy as np
from typing import Tuple, List
from sklearn.svm import SVC
from src.interfaces.classification import Classification


class SVM(Classification):
    """SVM分类模型实现"""

    def __init__(self, models=[]):
        self.models = models  # 为测试集加载已经训练好的模型

    def fit(self, data: Tuple) -> List[SVC]:
        X, y = data

        try:
            for i in range(y.shape[1]):  # 遍历每个输出维度
                if np.unique(y[:, i]).size < 2:  # 如果标签只有一个类别，则跳过
                    self.models.append(FixedOutputSVC(y[:, i][0]))
                    continue
                model = SVC(kernel='rbf', C=0.8, gamma=0.1)
                model.fit(X, y[:, i])
                self.models.append(model)
        except IndexError:
            if np.unique(y).size < 2:  # 如果标签只有一个类别，则跳过
                self.models.append(FixedOutputSVC(y[0]))

            model = SVC(kernel='linear')
            model.fit(X, y)
            self.models.append(model)

    def predict(self, data: Tuple):
        X, y = data
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        return y, np.array(predictions).T


class FixedOutputSVC(SVC):
    """输出给定值的SVC模型，应对标签仅有单一类别的情况"""

    def __init__(self, fixed_output, **kwargs):
        super().__init__(**kwargs)
        self.fixed_output = fixed_output

    def predict(self, X):
        return np.full((X.shape[0],), self.fixed_output)
