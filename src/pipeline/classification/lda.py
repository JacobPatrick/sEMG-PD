import numpy as np
from typing import Tuple, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from src.interfaces.classification import Classification


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
                model = LDA(n_components=min(50, X.shape[1], len(unique_labels) - 1))
                model.fit(X, y[:, i])
                self.models.append(model)
        except IndexError:
            unique_labels = np.unique(y)
            if unique_labels.size < 2:  # 如果标签只有一个类别，则跳过
                self.models.append(FixedOutputLDAC(unique_labels[0]))
            model = LDA(n_components=min(50, X.shape[1], len(unique_labels) - 1))
            model.fit(X, y)
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
        for model in self.models:
            predictions.append(model.predict(X))
        
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
