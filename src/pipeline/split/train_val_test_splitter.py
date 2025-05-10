import numpy as np
from sklearn.model_selection import train_test_split
from src.interfaces.split import Splitter
from typing import Tuple, Dict


class TrainValTestSplitter(Splitter):
    """训练集/验证集/测试集分割器"""

    def split(self, data: Tuple) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        X, y = data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        return dict(
            train=(X_train, y_train), val=(X_val, y_val), test=(X_test, y_test)
        )
