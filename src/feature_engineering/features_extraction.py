"""
@author: Jacob Patrick

@date: 2025-01-15

@email: jacob_patrick@163.com

@description: A helper class for hand-crafted feature extraction
"""

import numpy as np
# import pandas as pd

from scipy import stats
# import sampen
# import math

# import sys

from src.data_processing.read_data import read_data
from src.data_processing.pre_process import PreProcessor
from src.utils.load_config import load_json_config


class HandCraftedFeaturesExtractor:
	"""
	A helper class for hand-crafted feature extraction
	"""
	def __init__(self, vector):
		self.vec = vector
	
	def extractFeatures(self, features_config):
		"""
		Extract hand-crafted features from the input data
		"""
		features = []
		for c in range(len(self.vec)):
			for feature_name, params in features_config.items():
				if hasattr(self, feature_name):
					method = getattr(self, feature_name)
					if params:
						feature_value = method(c, **params)
					else:
						feature_value = method(c)
					features.append(feature_value)
		return np.array(features).reshape(-1, 1)	# 返回特征列向量

	################################
	##### Time Domain Features #####
	################################

	def getMAV(self):
		"""
		mean absolute value
		"""
		return np.mean(np.abs(self.vec))
	
	def getIEMG(self):
		"""
		integrated EMG
		"""
		return np.sum(np.abs(self.vec))

	def getRMS(self):
		"""
		root mean square
		"""
		return np.sqrt(np.mean(np.square(self.vec)))
	
	def getZC(self, threshold=0.1):
		"""
		zero crossing
		"""
		return len(zero_crossings = np.where(
				  (np.sign(self.vec[1:]) * np.sign(self.vec[:-1]) < 0) &
				  (np.abs(self.vec[1:]) >= threshold) &
				  (np.abs(self.vec[:-1]) >= threshold)
			  )[0])
	
	def getVAR(self):
		"""
		variance
		"""
		return np.var(self.vec)
	
	def getSSC(self):
		"""
		slope sign change
		"""
		return len(np.where(np.diff(np.sign(np.diff(self.vec))) != 0)[0])
	
	def getWAMP(self, threshold=0.1):
		"""
		willison amplitude
		"""
		return np.sum(np.abs(np.diff(self.vec)) >= threshold)
	
	def getSSI(self):
		"""
		signed square sum
		"""
		return np.sum(np.square(self.vec))

	def getKURT(self):
		"""
		kurtosis
		"""
		return stats.kurtosis(self.vec)
	
	def getWL(self):
		"""
		waveform length
		"""
		return np.sum(np.abs(np.diff(self.vec)))
	
	#####################################
	##### Frequency Domain Features #####
	#####################################

	def getTP(self):
		"""
		total power
		"""
		return np.sum(np.square(np.abs(np.fft.fft(self.vec))))

	def getMP(self):
		"""
		mean power
		"""
		return np.mean(np.square(np.abs(np.fft.fft(self.vec))))
	
	def getMNF(self, fs=2000):
		"""
		mean frequency
		"""
		N = len(self.vec)
		fft_res = np.fft.fft(self.vec)
		freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2 + 1]
		powers = np.abs(fft_res[:N // 2 + 1]) ** 2
		powers /= np.sum(powers)

		avg_freq = np.dot(freqs, powers)
		return avg_freq
	
	def getMDF(self, fs=2000):
		"""
		median frequency
		"""
		N = len(self.vec)
		fft_res = np.fft.fft(self.vec)
		freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2 + 1]
		powers = np.abs(fft_res[:N // 2 + 1]) ** 2
		powers /= np.sum(powers)

		cum_powers = np.cumsum(powers)
		mid_id = np.argmax(cum_powers >= 0.5)

		if mid_id == 0:
			return freqs[0]
		
		cum_prev = cum_powers[mid_id - 1]
		delta = 0.5 - cum_prev
		fraction = delta / powers[mid_id]
		mid_freq = freqs[mid_id - 1] + fraction * (freqs[mid_id] - freqs[mid_id - 1])
		return mid_freq
		
	##########################
	##### Other Features #####
	##########################

	def getAR(self):
		"""
		autoregression coefficient
		"""
		pass


if __name__ == "__main__":
	# load configration
	config = load_json_config('config.json')

	raw_data_path = config['data']['raw_path']
	processed_data_path = config['data']['processed_path']
	data_config = config['data_config']
	features_config = config['hand_crafted_features']

	# read raw data and pre-process
	raw_data = read_data(raw_data_path)
	processed_emg = PreProcessor(raw_data, data_config)
	filtered_emg = processed_emg.filter_emg()
	segmented_emg = processed_emg.segment_emg()

	feature_mat = np.zeros((len(segmented_emg) * segmented_emg[0].shape[1], 1))

	# extract features
	for i in range(len(segmented_emg)):
		for j in range(segmented_emg[0].shape[1]):
			fe = HandCraftedFeaturesExtractor(segmented_emg[j][:, i])
			feature_vec = fe.extractFeatures(features_config)
			feature_mat[j + i * segmented_emg[0].shape[1]] = feature_vec	# 针对传统机器学习模型，展平特征矩阵
	
	feature_mat = feature_mat.reshape(-1, 1)	# 转为列向量

	# 数据标签（评分）
