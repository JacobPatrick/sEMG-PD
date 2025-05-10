import numpy as np
from sklearn.model_selection import train_test_split
from src.interfaces.split import Splitter
from typing import Tuple, Dict


class TrainTestSplitter(Splitter):
    """训练集/测试集分割器"""

    def split(self, data: Tuple) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return dict(train=(X_train, y_train), test=(X_test, y_test))
