from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from src.interfaces.feature import FeatureExtractor
from src.config.config import FeatureConfig


class ManualFeatureExtractor(FeatureExtractor):
    """手工特征提取器实现"""

    def __init__(self):
        # 定义所有可用的特征提取函数
        self.time_domain_features = {
            "mav": self._get_mav,
            "iemg": self._get_iemg,
            "rms": self._get_rms,
            "zc": self._get_zc,
            "var": self._get_var,
            "ssc": self._get_ssc,
            "wamp": self._get_wamp,
            "ssi": self._get_ssi,
            "kurt": self._get_kurt,
            "wl": self._get_wl,
        }

        self.freq_domain_features = {
            "tp": self._get_tp,
            "mp": self._get_mp,
            "mnf": self._get_mnf,
            "mdf": self._get_mdf,
        }

        self.model_based_features = {
            "ar": self._get_ar,
        }

    def extract(
        self, data: Dict[str, Any], config: FeatureConfig
    ) -> Dict[str, Any]:
        """实现手工特征提取逻辑

        Args:
            data: 预处理后的数据，格式为:
                {
                    "raw": {
                        "sub-1": {
                            "sit": pd.DataFrame(...),
                            ...
                        },
                        ...
                    },
                    "labels": {...}
                }
            config: 配置参数，包含:
                - window_size: 窗口大小(秒)
                - overlap: 重叠大小(秒)
                - sampling_rate: 采样率
                - features: 要提取的特征列表

        Returns:
            features_data: 特征数据，格式为:
                {
                    "features": {
                        "sub-1": {
                            "sit": {
                                "window_0": {
                                    "channel1": {...},
                                    ...
                                },
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    "labels": {...}
                }
        """
        features_data = {"features": {}, "labels": data["labels"]}

        # 验证输入数据结构
        if "raw" not in data:
            raise ValueError("Input data must contain 'raw' key")
        if "labels" not in data:
            raise ValueError("Input data must contain 'labels' key")

        # 获取配置参数
        window_size = config.window_size
        overlap = config.overlap
        sampling_rate = config.sampling_rate

        # 计算窗口大小（采样数）
        window_samples = int(window_size * sampling_rate)
        overlap_samples = int(overlap * sampling_rate)
        stride = window_samples - overlap_samples

        # 对每个受试者的每个实验进行特征提取
        for subject_id, subject_data in data["raw"].items():
            features_data["features"][subject_id] = {}

            for exp_name, exp_data in subject_data.items():
                # 验证数据帧结构
                if not isinstance(exp_data, pd.DataFrame):
                    raise ValueError(
                        f"Experiment data for {subject_id}/{exp_name} must be a DataFrame"
                    )

                # 初始化实验数据结构
                # exp_features = {
                #     "windows": {},
                #     "metadata": {
                #         "window_size": window_size,
                #         "overlap": overlap,
                #         "sampling_rate": sampling_rate,
                #         "n_channels": len(exp_data.columns) - 1,  # 减去时间列
                #         "n_features": len(self.time_domain_features)
                #         + len(self.freq_domain_features),
                #     },
                # }
                exp_features = {}

                try:
                    # 获取信号数据（去除时间列）
                    signals = exp_data.drop('time', axis=1)
                except KeyError as e:
                    print(f"Error dropping 'time' column: {e}")
                    print("Available columns:", exp_data.columns.tolist())
                    raise

                # 对数据进行分窗并提取特征
                n_samples = len(signals)
                window_id = 0

                for start in range(0, n_samples - window_samples + 1, stride):
                    window_features = {}
                    end = start + window_samples

                    # 对每个通道提取特征
                    for channel in signals.columns:
                        window_data = signals[channel].values[start:end]
                        window_features[channel] = (
                            self._extract_window_features(
                                window_data, sampling_rate
                            )
                        )

                    # exp_features["windows"][
                    #     f"window_{window_id}"
                    # ] = window_features
                    exp_features[f"window_{window_id}"] = window_features
                    window_id += 1

                # exp_features["metadata"]["n_windows"] = window_id
                features_data["features"][subject_id][exp_name] = exp_features

        return features_data

    def _extract_window_features(
        self, window_data: np.ndarray, sampling_rate: int
    ) -> Dict[str, float]:
        """提取单个窗口的所有特征"""
        features = {}

        # 设置特征提取的上下文
        self.vec = window_data
        self.fs = sampling_rate

        # 提取时域特征
        for name, func in self.time_domain_features.items():
            features[name] = func()

        # 提取频域特征
        for name, func in self.freq_domain_features.items():
            features[name] = func()

        # 提取基于模型的特征
        # for name, func in self.model_based_features.items():
        #     features[name] = func()

        return features

    """时域特征"""

    def _get_mav(self) -> float:
        """平均绝对值"""
        return np.mean(np.abs(self.vec))

    def _get_iemg(self) -> float:
        """积分肌电"""
        return np.sum(np.abs(self.vec))

    def _get_rms(self) -> float:
        """均方根"""
        return np.sqrt(np.mean(np.square(self.vec)))

    def _get_zc(self, threshold: float = 0.1) -> float:
        """过零点数"""
        zero_crossings = np.where(
            (np.sign(self.vec[1:]) * np.sign(self.vec[:-1]) < 0)
            & (np.abs(self.vec[1:]) >= threshold)
            & (np.abs(self.vec[:-1]) >= threshold)
        )[0]
        return len(zero_crossings)

    def _get_var(self) -> float:
        """方差"""
        return np.var(self.vec)

    def _get_ssc(self) -> float:
        """斜率符号变化"""
        return len(np.where(np.diff(np.sign(np.diff(self.vec))) != 0)[0])

    def _get_wamp(self, threshold: float = 0.1) -> float:
        """Willison幅值"""
        return np.sum(np.abs(np.diff(self.vec)) >= threshold)

    def _get_ssi(self) -> float:
        """简单平方积分"""
        return np.sum(np.square(self.vec))

    def _get_kurt(self) -> float:
        """峰度"""
        return stats.kurtosis(self.vec)

    def _get_wl(self) -> float:
        """波形长度"""
        return np.sum(np.abs(np.diff(self.vec)))

    """频域特征"""

    def _get_tp(self) -> float:
        """总功率"""
        return np.sum(np.square(np.abs(np.fft.fft(self.vec))))

    def _get_mp(self) -> float:
        """平均功率"""
        return np.mean(np.square(np.abs(np.fft.fft(self.vec))))

    def _get_mnf(self) -> float:
        """平均频率"""
        N = len(self.vec)
        fft_res = np.fft.fft(self.vec)
        freqs = np.fft.fftfreq(N, 1 / self.fs)[: N // 2 + 1]
        powers = np.abs(fft_res[: N // 2 + 1]) ** 2
        powers /= np.sum(powers)

        return np.dot(freqs, powers)

    def _get_mdf(self) -> float:
        """中值频率"""
        N = len(self.vec)
        fft_res = np.fft.fft(self.vec)
        freqs = np.fft.fftfreq(N, 1 / self.fs)[: N // 2 + 1]
        powers = np.abs(fft_res[: N // 2 + 1]) ** 2
        powers /= np.sum(powers)

        cum_powers = np.cumsum(powers)
        mid_id = np.argmax(cum_powers >= 0.5)

        if mid_id == 0:
            return freqs[0]

        cum_prev = cum_powers[mid_id - 1]
        delta = 0.5 - cum_prev
        fraction = delta / powers[mid_id]
        mid_freq = freqs[mid_id - 1] + fraction * (
            freqs[mid_id] - freqs[mid_id - 1]
        )

        return mid_freq

    """基于模型的特征"""

    # TODO: 修复错误
    def _get_ar(self) -> float:
        """AR模型系数"""
        return np.linalg.eig(np.cov(self.vec))[0][0]
