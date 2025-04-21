from sklearn.metrics import *
from typing import List, Any


def evaluate_classification(y_true: List[int], y_pred: List[int]) -> Any:
    """
    评估模型性能
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
