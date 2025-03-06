'''
@author: Jacob Patrick

@date: 2025-02-21

@email: jacob_patrick@163.com

@description: Extract basic parameters from tremor signals
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class TremorParamExtractor:
    """
    从肌电信号中提取震颤相关的参数
    """
    def __init__(self, data, fs=2000):
        self.data = data
        self.fs = fs
        self.length = data.shape[0]
        self.n_channels = data.shape[1]

    def getPeak(self):
        """
        计算信号的峰值
        """
        peaks = []
        for ch in range(self.n_channels):
            peak, _ = signal.find_peaks(self.data[:, ch])
            peaks.append(peak)

        return np.array(peaks).T

    # 这个参数暂时看起来也不太靠谱
    def getHalfPeakBandwidth(self):
        """
        计算信号的半峰带宽
        """
        hfb = []
        for ch in range(self.n_channels):
            freqs, psd_ch = signal.welch(self.data[:, ch], fs=self.fs)
            half_peak_power = np.max(psd_ch) / 2
            indices = np.where(psd_ch >= half_peak_power)[0]
            try:
                hfb_ch = freqs[indices[-1]] - freqs[indices[0]]
                hfb.append(hfb_ch)
            except:
                hfb.append(0)
        
        return np.array(hfb).T

    def getEnvelope(self):
        """
        计算信号的包络
        """
        env = []
        for ch in range(self.n_channels):
            envelope = np.abs(signal.hilbert(self.data[:, ch]))
            env.append(envelope - np.mean(envelope))
        
        return np.array(env).T

    def plotPsd(self, fig_path):
        """
        绘制信号的功率谱密度图
        """
        _, axes = plt.subplots(nrows=self.n_channels, sharex=True, squeeze=False, figsize=(20, self.n_channels * 1.5))
        
        for i in range(self.n_channels):
            freqs, psd = signal.welch(self.data[:, i], fs=self.fs)
            axes[i, 0].plot(freqs, psd)
            axes[i, 0].set_ylabel(['ECR-L', 'FCR-L', 'ECR-R', 'FCR-R', 'TA-L', 'GA-L', 'TA-R', 'GA-R'][i])

            if i != self.n_channels - 1:
                axes[i, 0].tick_params(labelbottom=False)

        axes[-1, 0].set_xlabel('Frequency')
        plt.suptitle('EMG Channels')
        plt.subplots_adjust(hspace=0)
        plt.savefig(fig_path)
    
    def getPowerRatio(self, env_psd_freq, window_size=20000, overlap=10000):
        """
        计算功率比
        Args:
            window_size: 窗口大小,默认20000个采样点(10s)
            overlap: 重叠长度,默认10000个采样点(5s)
        Returns:
            power_ratio: 震颤频段信号(峰值频率+-0.25Hz)与低频信号(0-2Hz)的功率比
        """
        power_ratio = []

        for ch in range(self.n_channels):
            # 分段处理信号
            segments = self._segment_signal(self.data[:, ch], window_size, overlap)

            # 对每段信号进行频谱分析
            power_ratio_list = []
            for seg in segments:
                # 计算功率谱密度
                freqs, psd = signal.welch(seg, fs=self.fs, nperseg=len(seg))

                # 提取峰值频率附近范围内的功率
                peak_freq = env_psd_freq[ch]
                mask = (freqs >= (peak_freq - 0.25)) & (freqs <= (peak_freq + 0.25))
                peak_freq_power = np.sum(psd[mask])

                # 提取0-2Hz范围内的功率
                mask = (freqs > 0) & (freqs <= 2)
                low_freq_power = np.sum(psd[mask])

                # 计算功率比
                power_ratio_seg = (peak_freq_power / 0.5) / (low_freq_power / 2)
                power_ratio_list.append(power_ratio_seg)

            # 取平均数作为该通道的功率比
            power_ratio.append(np.mean(power_ratio_list))

        return np.array(power_ratio)
    
    def getTremorFreq(self, window_size=20000, overlap=10000):
        """
        提取震颤频率
        Args:
            window_size: 窗口大小,默认20000个采样点(10s)
            overlap: 重叠长度,默认10000个采样点(5s)
        Returns:
            dominant_freqs: 各通道的主频率
        """
        dominant_freqs = []
        
        for ch in range(self.n_channels):
            # 分段处理信号
            segments = self._segment_signal(self.data[:, ch], window_size, overlap)
            
            # 对每段信号进行频谱分析
            freqs_list = []
            for seg in segments:
                # 计算功率谱密度
                freqs, psd = signal.welch(seg, fs=self.fs, nperseg=len(seg))

                # 提取3-7Hz范围内的峰值频率
                mask = (freqs >= 3) & (freqs <= 7)
                tremor_freq = freqs[mask][np.argmax(psd[mask])]
                freqs_list.append(tremor_freq)
            
            # 取中位数作为该通道的震颤频率
            dominant_freqs.append(np.median(freqs_list))
            
        return np.array(dominant_freqs)
    
    def _segment_signal(self, signal, window_size, overlap):
        step = window_size - overlap
        n_segments = (len(signal) - overlap) // step
        segments = []
        
        for i in range(n_segments):
            start = i * step
            end = start + window_size
            segments.append(signal[start:end])
            
        return segments
