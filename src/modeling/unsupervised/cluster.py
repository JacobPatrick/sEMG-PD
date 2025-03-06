"""
@author: Jacob Patrick

@date: 2025-01-30

@email: jacob_patrick@163.com

@description: 使用 K-means 聚类分析区分治疗前后的肌电信号样本
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# from sklearn.decomposition import PCA
from src.feature_engineering.features_extraction import (
    HandCraftedFeaturesExtractor,
)


os.environ['OMP_NUM_THREADS'] = '1'


class Clustering:
    """
    cluster = Clustering()
    cluster.k_means(all_flattened_features_mat)
    """

    def __init__(self):
        pass

    def k_means(self, all_features, k=None):
        # step1: 数据加载与整理
        # extract features
        # feature_lst_0 = []
        # for i in range(segmented_emg[0].shape[1]):
        #     feature_lst_1 = []
        #     for j in range(len(segmented_emg)):
        #         fe = HandCraftedFeaturesExtractor(segmented_emg[j][:, i])
        #         feature_vec = fe.extractFeatures(features_config)
        #         feature_lst_1.append(feature_vec)

        #     feature_mat_0 = np.stack(feature_lst_1, axis=0)
        #     feature_lst_0.append(feature_mat_0)

        # feature_mat = np.stack(feature_lst_0, axis=0)  # 形成形状为 (n_channels, n_segments, n_features) 的特征矩阵
        # flattened_features = feature_mat.flatten()  # 展平特征矩阵
        # labels = [i for i in range(6) for _ in range(6)] # 一个患者，疗前疗后各三组动作，每组动作采6个样本
        labels = [i for i in range(2) for _ in range(6)]  # 只对静坐样本聚类

        # step2: 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(all_features)

        # step3: 根据 elbow-method 选择最优 k 值
        if k is None:
            optimal_k = find_optimal_k(scaled_data=scaled_data, k_range=10)
        elif isinstance(k, int):
            optimal_k = k
        else:
            raise ValueError("k must be an integer")

        # step4: K - means聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # step5: 结果分析
        cluster_results = pd.DataFrame(
            {'Treatment Status': labels, 'Cluster': cluster_labels}
        )

        # 查看每个簇中治疗前和治疗后的样本数量
        cluster_summary = (
            cluster_results.groupby(['Cluster', 'Treatment Status'])
            .size()
            .unstack()
        )
        print(cluster_summary)

        # # PCA 降维高维特征
        # pca = PCA(n_components=2)
        # reduced_data = pca.fit_transform(scaled_data)

        # # 绘制聚类结果图
        # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis')
        # plt.show()


def find_optimal_k(scaled_data, k_range):
    # 计算不同 k 值下的惯性
    inertia = []
    k_range_ = range(1, k_range + 1)
    for k in k_range_:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # 计算惯性的二阶差分
    diff = np.diff(inertia)
    second_diff = np.diff(diff)

    # 找到二阶差分绝对值最大的位置，对应最优 k 值
    optimal_k = k_range_[np.argmax(np.abs(second_diff)) + 1]
    print(f"Optimal k: {optimal_k}")

    return optimal_k


if __name__ == '__main__':
    pass
    # # 定义病人编号、动作标签和疗前疗后标签
    # patient_ids = range(1, 2)	# range(1, 7)
    # action_labels = ['sit']
    # treatment_labels = ['pre', 'post']

    # # 用于存储所有展平的特征矩阵
    # all_flattened_features = []

    # # 按患者-动作-疗前疗后的顺序，遍历所有需要的数据文件
    # for patient_id in patient_ids:
    #     for action_label in action_labels:
    #         for treatment_label in treatment_labels:
    #             # 构建文件名
    #             file_name = f"{patient_id}-{action_label}-{treatment_label}.csv"
    #             file_path = os.path.join(raw_data_path, file_name)

    #             try:
    #                 # read raw data and pre-process
    #                 raw_data = read_data(file_path)
    #                 processed_emg = PreProcessor(raw_data, data_config)

    #                 # segment the EMG data
    #                 # segmented_emg = processed_emg.segment_emg()

    #                 # samle the EMG data
    #                 sampled_emg = processed_emg.sample_emg()
    #                 for i in range(len(sampled_emg)):
    #                     segmented_emg = segment(sampled_emg[i])

    #                     # extract features
    #                     feature_lst_0 = []
    #                     for i in range(segmented_emg[0].shape[1]):
    #                         feature_lst_1 = []
    #                         for j in range(len(segmented_emg)):
    #                             fe = HandCraftedFeaturesExtractor(segmented_emg[j][:, i])
    #                             feature_vec = fe.extractFeatures(features_config)
    #                             feature_lst_1.append(feature_vec)

    #                         feature_mat_0 = np.stack(feature_lst_1, axis=0)
    #                         feature_lst_0.append(feature_mat_0)

    #                     feature_mat = np.stack(feature_lst_0, axis=0)  # 形成形状为 (n_channels, n_segments, n_features) 的特征矩阵
    #                     flattened_features = feature_mat.flatten()  # 展平特征矩阵

    #                     # 将展平的特征矩阵添加到列表中
    #                     all_flattened_features.append(flattened_features)

    #             except FileNotFoundError:
    #                 print(f"文件 {file_path} 未找到，跳过该文件。")

    # # 可以将所有展平的特征矩阵转换为一个 (n_samples, n_features) 形状的大特征矩阵
    # all_flattened_features_mat = np.vstack(all_flattened_features)
