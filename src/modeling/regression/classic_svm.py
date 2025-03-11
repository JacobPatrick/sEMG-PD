"""
@author: Jacob Patrick

@date: 2025-01-18

@email: jacob_patrick@163.com

@description: A classic SVM model for predicting PDwP UPDRS scores
"""

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
