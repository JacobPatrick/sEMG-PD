"""
@author: Jacob Patrick

@date: 2025-01-17

@email: jacob_patrick@163.com

@description: A script for reading raw data
"""
# import numpy as np
import pandas as pd

# from matplotlib import pyplot as plt

def read_data(file_path):
	"""
	read raw data from csv file
	"""
	dtype_dict = {2: str}	# 设置第 3 列的数据类型为字符串
	df = pd.read_csv(file_path, skiprows=3, header=3, dtype=dtype_dict)
	columns = [0] + list(range(3, df.shape[1]))
	selected_df = df.iloc[:, columns]
	data = selected_df.iloc[1:, :].values

	return data
