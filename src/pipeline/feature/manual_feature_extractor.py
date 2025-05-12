import numpy as np
from scipy import stats
from src.interfaces.feature import FeatureExtractor
from src.config.config import FeatureConfig
from typing import Tuple, Dict, List, Any


class NewManualFeatureExtractor(FeatureExtractor):
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

    def extract(
        self, dataset: Tuple[np.ndarray, np.ndarray], config: FeatureConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = dataset[0]
        labels = dataset[1]

        # 获取配置参数
        window_size = config.window_size
        overlap = config.overlap
        sampling_rate = config.sampling_rate
        feature_names = config.features

        # 计算窗口大小（采样数）
        window_samples = int(window_size * sampling_rate)
        overlap_samples = int(overlap * sampling_rate)
        stride = window_samples - overlap_samples

        n_samples, n_channels, ts_length = data.shape

        n_windows = max(int((ts_length - overlap_samples) // stride), 1)

        n_features = len(feature_names)
        features = np.zeros((n_samples, n_channels, n_features, n_windows))

        for i in range(n_samples):
            for j in range(n_channels):
                channel_data = data[i, j]
                channel_features = []
                for w in range(n_windows):
                    start_idx = w * stride
                    end_idx = start_idx + window_samples
                    if end_idx > ts_length:
                        break
                    window = channel_data[start_idx:end_idx]
                    channel_features.append(
                        self._extract_window_features(
                            window, feature_names, sampling_rate
                        )
                    )

                features[i, j] = np.array(channel_features).T

            # 对所有通道中提取的单个特征归一化
            for k in range(n_features):
                min_value = np.min(features[i, :, k])
                max_value = np.max(features[i, :, k])
                features[i, :, k] = (features[i, :, k] - min_value) / (
                    max_value - min_value
                )

        return features, labels

    def _extract_window_features(
        self, window: np.ndarray, features: List[str], sampling_rate: int
    ) -> np.ndarray:
        """从单个窗口中提取特征"""
        feature_vector = []

        # 设置特征提取的上下文
        self.vec = window
        self.fs = sampling_rate

        # 提取时域特征
        for name, func in self.time_domain_features.items():
            if features and name not in features:
                continue
            feature_vector.append(func())

        # 提取频域特征
        for name, func in self.freq_domain_features.items():
            if features and name not in features:
                continue
            feature_vector.append(func())

        return np.array(feature_vector)

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

    # FIXME: 归一化后部分数值过于接近导致逢赌计算出现精度损失
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
