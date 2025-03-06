"""
@author: Jacob Patrick

@date: 2025-01-16

@email: jacob_patrick@163.com

@description: Main script for the project
"""
import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.read_data import read_data
from src.data_processing.pre_process import PreProcessor
from src.data_processing.pre_process import segment
from src.utils.plot_signal import SignalPlotter
from src.feature_engineering.features_extraction import HandCraftedFeaturesExtractor
from src.feature_engineering.basic_param import TremorParamExtractor
from src.modeling.unsupervised.cluster import Clustering
# from src.modeling.model_training import train_model
# from src.modeling.model_validation import cross_validate_model
from src.utils.load_config import load_json_config

# load configration
config = load_json_config('config/config.json')

RAW_DATA_PATH = config['path']['data']['raw_path']
PROCESSED_DATA_PATH = config['path']['data']['processed_path']
DATA_CONFIG = config['data_config']['cfg-1']
TEST_FIG_PATH = config['path']['output']['fig']['test']

FEATURES_CONFIG = config['hand_crafted_features']

MODEL_TYPE = config['model']['type']
MODEL_PARAMS = config['model']['params']

# # read raw data and pre-process
# raw_data = read_data(raw_data_path)
# processed_emg = PreProcessor(raw_data, data_config)

# # get raw EMG data
# raw_emg = processed_emg.get_raw_emg()

# # filter the EMG data
# filtered_emg = processed_emg.filter_emg()

# # read raw data and pre-process
# raw_data = read_data(raw_data_path)
# processed_emg = PreProcessor(raw_data, data_config)
# filtered_emg = processed_emg.filter_emg()
# segmented_emg = processed_emg.segment_emg()

# # 这一段之后看看怎么放回到 feature_extraction.py 里去
# # extract features
# feature_lst_0 = []
# for i in range(segmented_emg[0].shape[1]):

#     feature_lst_1 = []
#     for j in range(len(segmented_emg)):
#         fe = HandCraftedFeaturesExtractor(segmented_emg[j][:, i])
#         feature_vec = fe.extractFeatures(features_config)
#         feature_lst_1.append(feature_vec)

#     feature_mat_0 = np.stack(feature_lst_1, axis=0)
#     feature_lst_0.append(feature_mat_0)

# feature_mat = np.stack(feature_lst_0, axis=0)	# 形成形状为 (num_channels, num_segments, num_features) 的特征矩阵
# flattened_features = feature_mat.flatten()	# 展平特征矩阵

# print(len(segmented_emg))
# print(segmented_emg[0].shape)
# print(feature_mat.shape)

MODE = 'TIME_PLOT'

# 定义病人编号、动作标签和疗前疗后标签
patient_ids = range(2, 4)	# range(1, 7)
treatment_labels = ['pre', 'post']
# action_labels = ['sit', 'motion1', 'motion2', 'motion3', 'motion4', 'gait']
action_labels = ['sit', 'motion', 'gait']

# 用于存储所有展平的特征矩阵
all_flattened_features = []

# 用于存储所有已分割数据的向量
all_segmented_emg = []	#TODO: 将这部分特征提取的代码移入 Clustering 类中，外部最多保留分段采样的代码

# 按患者-动作-疗前疗后的顺序，遍历所有需要的数据文件
for patient_id in patient_ids:
	for treatment_label in treatment_labels:
		for action_label in action_labels:
			# 匹配文件名
			file_name = f"{patient_id}-{treatment_label}-{action_label}.csv"
			file_path = os.path.join(RAW_DATA_PATH, '20250204', file_name)

			try:
				# read raw data and pre-process
				raw_data = read_data(file_path)
				processed_emg = PreProcessor(raw_data, DATA_CONFIG).filter_emg()
				if MODE == 'EXTRACT_FEATURES':
					# segment the EMG data
					# segmented_emg = processed_emg.segment_emg()
					
					# 分段采样，自此列表中的数据按患者-动作-疗前疗后-分段采样的顺序排列
					sampled_emg = processed_emg.sample_emg()
					for i in range(len(sampled_emg)):
						segmented_emg = segment(sampled_emg[i])
						all_segmented_emg.append(segmented_emg)

						# extract features
						feature_lst_0 = []
						for i in range(segmented_emg[0].shape[1]):
							feature_lst_1 = []
							for j in range(len(segmented_emg)):
								fe = HandCraftedFeaturesExtractor(segmented_emg[j][:, i])
								feature_vec = fe.extractFeatures(FEATURES_CONFIG)
								feature_lst_1.append(feature_vec)

							feature_mat_0 = np.stack(feature_lst_1, axis=0)
							feature_lst_0.append(feature_mat_0)

						feature_mat = np.stack(feature_lst_0, axis=0)  # 形成形状为 (n_channels, n_segments, n_features) 的特征矩阵
						flattened_features = feature_mat.flatten()  # 展平特征矩阵

						# 将展平的特征矩阵添加到列表中
						all_flattened_features.append(flattened_features)

				elif MODE == 'TIME_PLOT':
					# plot the time figure
					path = os.path.join(TEST_FIG_PATH, f"{patient_id}_{treatment_label}_{action_label}_time.png")
					SignalPlotter(processed_emg, start=0, end=10).plot_time_domain(fig_path=path)
					
				elif MODE == 'FREQ_PLOT':
					# plot the frequency figure
					path = os.path.join(TEST_FIG_PATH, f"{patient_id}_{treatment_label}_{action_label}_freq.png")
					SignalPlotter(processed_emg, start=0, end=10).plot_freq_domain(fig_path=path)

				elif MODE == 'PSD_PLOT':
					# plot the PSD figure
					path = os.path.join(TEST_FIG_PATH, f"{patient_id}_{treatment_label}_{action_label}_psd.png")
					TremorParamExtractor(processed_emg).plotPsd(fig_path=path)

				elif MODE == 'PSD_FREQ':
					# get the tremor frequency
					tremor_freq = TremorParamExtractor(processed_emg).getTremorFreq()
					print(tremor_freq)

				elif MODE == 'ENV_TIME_PLOT':
					# plot the envelope figure in time domain
					path = os.path.join(TEST_FIG_PATH, f"{patient_id}_{treatment_label}_{action_label}_env_time.png")
					env = TremorParamExtractor(processed_emg).getEnvelope()
					SignalPlotter(env, start=0, end=10).plot_time_domain(fig_path=path)

				elif MODE == 'ENV_FREQ_PLOT':
					# plot the envelope figure in frequency domain
					path = os.path.join(TEST_FIG_PATH, f"{patient_id}_{treatment_label}_{action_label}_env_freq.png")
					env = TremorParamExtractor(processed_emg).getEnvelope()
					SignalPlotter(env, start=0, end=10).plot_freq_domain(fig_path=path)

				elif MODE == 'ENV_PSD_FREQ':
					# get the envelope frequency
					env = TremorParamExtractor(processed_emg).getEnvelope()
					env_psd_freq = TremorParamExtractor(env).getTremorFreq()
					print(f"{patient_id}-{treatment_label}-{action_label} envelope psd frequency:\n", env_psd_freq)

				elif MODE == 'POWER_RATIO':
					# get the envelope frequency
					env = TremorParamExtractor(processed_emg).getEnvelope()
					env_psd_freq = TremorParamExtractor(env).getTremorFreq()

					# get the power ratio
					power_ratio = TremorParamExtractor(processed_emg).getPowerRatio(env_psd_freq=env_psd_freq)
					print(f"{patient_id}-{treatment_label}-{action_label} power ratio:\n", power_ratio)

				elif MODE == 'HALF_PEAK_BANDWIDTH':
					# get the envelope frequency
					env = TremorParamExtractor(processed_emg).getEnvelope()

					# get the half peak bandwidth
					hfb = TremorParamExtractor(env).getHalfPeakBandwidth()
					print(f"{patient_id}-{treatment_label}-{action_label} half peak bandwidth:\n", hfb)

				else:
					raise ValueError(f"Invalid mode: {MODE}")

			# 如果没有对应文件，则跳过
			except FileNotFoundError:
				print(f"文件 {file_path} 未找到，跳过该文件。")

# fs = 2000

# patient_ids = range(1, 2)	# range(1, 7)
# treatment_labels = ['pre', 'mid', 'post']
# action_labels = ['sit']

# # 用于存储单个患者肌电数据的列表
# all_emg_data = []

# for patient_id in patient_ids:
# 	for treatment_label in treatment_labels:
# 		for action_label in action_labels:
# 			# 匹配文件名
# 			file_name = f"{patient_id}-{treatment_label}-{action_label}.csv"
# 			file_path = os.path.join(RAW_DATA_PATH, file_name)

# 			try:
# 				# read raw data and pre-process
# 				raw_data = read_data(file_path)
# 				processed_emg = PreProcessor(raw_data, DATA_CONFIG).filter_emg()

# 				# 采样10-20秒上肢数据
# 				processed_emg = processed_emg[10*fs:13*fs, :4]
# 				all_emg_data.append(processed_emg)

# 			except FileNotFoundError:
# 				print(f"文件 {file_path} 未找到，跳过该文件。")
# 				continue

# RMS = [np.sqrt(np.mean(np.square(emg_data))) for emg_data in all_emg_data[1].T]
# print('RMS: ', RMS)

# # 画图
# fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, squeeze=False, figsize=(12, 8))
# set_ylim = [[-20, 20], [-25, 25], [-5, 5], [-2, 2]]
# set_color = ['#72B063', '#719AAC', '#E29135']

# for j in range(2):
# 	for i in range(4):
# 		axes[i, j].set_ylim(set_ylim[i])
# 		axes[i, 0].set_ylabel(['ECR-L', 'FCR-L', 'ECR-R', 'FCR-R'][i])
# 		axes[i, j].plot(all_emg_data[j][:, i], color=set_color[j], linewidth=0.5)
# 		# axes[-1, j].set_xlabel('Time')

# plt.subplots_adjust(hspace=0)
# plt.show()

# # 可以将所有展平的特征矩阵转换为一个 (n_samples, n_features) 形状的大特征矩阵
# all_flattened_features_mat = np.vstack(all_flattened_features)

# 添加数据标签

# # 提取特征
# X, y = extract_features(data)

# # 训练模型
# model = train_model(X, y)

# # 交叉验证模型
# cv_score = cross_validate_model(model, X, y)
# print(f'Cross-validation score: {cv_score}')
