import numpy as np
from typing import Tuple, List
from sklearn.svm import SVC

# from sklearn.calibration import CalibratedClassifierCV
from src.interfaces.classification import Classification
from src.utils.visualization import plot_confusion_matrix


class SVM(Classification):
    """SVM分类模型实现"""

    def __init__(self, models=[]):
        self.models = models  # 为测试集加载已经训练好的模型

    def fit(self, data: Tuple) -> List[SVC]:
        X, y = data

        if y.ndim == 1:
            if np.unique(y).size < 2:  # 如果标签只有一个类别，则跳过
                self.models.append(FixedOutputSVC(y[0]))
            else:
                model = SVC(kernel='linear', probability=True)
                model.fit(X, y)
                self.models.append(model)

        else:
            for i in range(y.shape[1]):  # 遍历每个输出维度
                if np.unique(y[:, i]).size < 2:  # 如果标签只有一个类别，则跳过
                    self.models.append(FixedOutputSVC(y[:, i][0]))
                    continue
                model = SVC(kernel='linear', probability=True)
                model.fit(X, y[:, i])
                self.models.append(model)

        return self.models

    def predict_proba(self, data: Tuple):
        X, y = data
        proba_predictions = []

        for model in self.models:
            proba_predictions.append(model.predict_proba(X))

        return y, np.array(proba_predictions).T

    def predict(self, data: Tuple):
        X, y = data
        predictions = []
        for idx, model in enumerate(self.models):
            predictions.append(model.predict(X))
            print(y[:, idx])
            print(predictions[-1])

            plot_confusion_matrix(
                label_true=y[:, idx],
                label_pred=predictions[-1],
                classes=[0, 1, 2, 3, 4],
                title=f"reports/fig/final/gait_svm_linear_test_confusion_matrix_{idx}.png",
            )

        return y, np.array(predictions).T


class FixedOutputSVC:
    """输出给定值的SVC模型，应对标签仅有单一类别的情况"""

    def __init__(self, fixed_output):
        self.fixed_output = fixed_output

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        # 生成只有一个类的概率分布
        probas = np.zeros((n_samples, 5))
        probas[:, self.fixed_output] = 1.0
        return probas

    def predict(self, X):
        return np.full((X.shape[0],), self.fixed_output)
