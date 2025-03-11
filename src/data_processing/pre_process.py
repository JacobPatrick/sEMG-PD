"""
@author: Jacob Patrick

@date: 2025-03-06

@email: lumivoxflow@gmail.com

@description: A helper class for sEMG and IMU signal pre-processing
"""

from scipy.signal import iirnotch, butter, filtfilt


class PreProcessor:
    """
    A helper class for sEMG and IMU signal pre-processing
    """

    def __init__(self, data):
        self.data = data
        self.time = data[:, 0]

    def filter_emg(self, fs=2000, lowcut=10.0, highcut=500.0, order=4, notch_freq=50.0, Q=30.0):
        """
        filter the sEMG signal
        """
        if self.data is None:
            raise ValueError("sEMG data is not loaded")
        
        b_notch, a_notch = iirnotch(notch_freq / (fs / 2), Q)
        b_butter, a_butter = butter(
            order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band'
        )

        for i in range(self.data.shape[1]):
            self.data[:, i] = filtfilt(
                b_butter, a_butter, self.data[:, i]
            )
            self.data[:, i] = filtfilt(
                b_notch, a_notch, self.data[:, i]
            )

        return self.data
    
    def filter_imu(self, fs=200, highcut=50.0):
        """
        filter the IMU signal
        """
        if self.data is None:
            raise ValueError("ACC data is not loaded")

        b_butter, a_butter = butter(4, highcut / (fs / 2), btype='low')
        for i in range(self.data.shape[1]):
            self.data[:, i] = filtfilt(
                b_butter, a_butter, self.data[:, i]
            )

        return self.data
    
def segment_signal(signal, window_size, overlap_ratio):
    """
    segment single channel signal into windows
    :param signal: the single channel signal to be segmented
    :param window_size: the size of each window
    :param overlap_ratio: the overlap ratio between adjacent windows
    :return: a list of segments
    """
    step_size = int(window_size * (1 - overlap_ratio))
    segments = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        segments.append(signal[start : end])

    return segments
