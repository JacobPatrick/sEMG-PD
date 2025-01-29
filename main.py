"""
@author: Jacob Patrick

@date: 2025-01-16

@email: jacob_patrick@163.com

@description: Main script for the project
"""
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.read_data import read_data
from src.data_processing.pre_process import PreProcessor
from src.feature_engineering.features_extraction import HandCraftedFeaturesExtractor
# from src.modeling.model_training import train_model
# from src.modeling.model_validation import cross_validate_model
from src.utils.load_config import load_json_config

# load configration
config = load_json_config('config/config.json')

raw_data_path = config['data']['raw_path']
processed_data_path = config['data']['processed_path']
data_config = config['data_config']

features_config = config['hand_crafted_features']

model_type = config['model']['type']
model_params = config['model']['params']

# read raw data and pre-process
raw_data = read_data(raw_data_path)
processed_emg = PreProcessor(raw_data, data_config)

# get raw EMG data
raw_emg = processed_emg.get_raw_emg()

# filter the EMG data
filtered_emg = processed_emg.filter_emg()

# Take any data within 5 second for plotting
duration = 5
sample_rate = 2000
start = np.random.randint(0, filtered_emg.shape[0] - duration * sample_rate)
end = start + duration * sample_rate

sample_num, channel_num = filtered_emg.shape
fig, axes = plt.subplots(nrows=channel_num, sharex=True, squeeze=False, figsize=(20, channel_num * 1.5))

ylabel = ['ECR-L', 'FCR-L', 'ECR-R', 'FCR-R', 'TA-L', 'GA-L', 'TA-R', 'GA-R']

# plot time domain signal
# x = np.arange(sample_num) / sample_rate
# for i in range(channel_num):
#     axes[i, 0].set_ylim(-60, 60)
#     axes[i, 0].plot(x[start:end], raw_emg[start:end, i])    # raw EMG data
#     axes[i, 0].plot(x[start:end], filtered_emg[start:end, i])   # filtered EMG data
#     axes[i, 0].set_ylabel(ylabel[i])

#     if i != channel_num - 1:
#         axes[i, 0].tick_params(labelbottom=False)

# axes[-1, 0].set_xlabel('Time')
# plt.suptitle('EMG Channels')

# plt.subplots_adjust(hspace=0)
# plt.show()

# plot frequency domain signal
n = raw_emg.shape[0]
emg_f = np.fft.fft(raw_emg[:, 0])
# emg_f = np.fft.fft(filtered_emg[:, 0])
emg_f_abs = np.abs(emg_f)

freq = np.fft.fftfreq(n, 1/sample_rate)

half_n = n // 2
half_freq = freq[:half_n]
half_emg_f = emg_f_abs[:half_n]

half_emg_f[0] = emg_f_abs[0]

for i in range(channel_num):
    axes[i, 0].plot(half_freq, half_emg_f)
    axes[i, 0].set_ylabel(ylabel[i])

    if i!= channel_num - 1:
        axes[i, 0].tick_params(labelbottom=False)

axes[-1, 0].set_xlabel('Frequency')
plt.suptitle('EMG Channels')
plt.subplots_adjust(hspace=0)
plt.show()

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

feature_mat = feature_mat.reshape(-1, 1)	# 转为列向量3

# # 提取特征
# X, y = extract_features(data)

# # 训练模型
# model = train_model(X, y)

# # 交叉验证模型
# cv_score = cross_validate_model(model, X, y)
# print(f'Cross-validation score: {cv_score}')
