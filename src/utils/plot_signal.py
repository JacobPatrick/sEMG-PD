import numpy as np
import matplotlib.pyplot as plt


class SignalPlotter:
    def __init__(self, signal, fs=2000, random=False, **kwargs):
        self.signal = signal
        self.fs = fs
        self.random = random

        if self.random:
            required_params = ['duration']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter: {param}")
                setattr(self, param, kwargs[param])

            self.start = np.random.randint(
                0, self.signal.shape[0] - self.duration * self.fs
            )
            self.end = self.start + self.duration * self.fs
        else:
            required_params = ['start', 'end']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter: {param}")

            self.start = kwargs['start'] * fs
            self.end = kwargs['end'] * fs

        self.sample_num, self.channel_num = self.signal.shape

        self.ylabel = [
            'ECR-L',
            'FCR-L',
            'ECR-R',
            'FCR-R',
            'TA-L',
            'GA-L',
            'TA-R',
            'GA-R',
        ]

    def plot_time_domain(self, fig_path):
        _, axes = plt.subplots(
            nrows=self.channel_num,
            sharex=True,
            squeeze=False,
            figsize=(20, self.channel_num * 1.5),
        )
        x = np.arange(self.sample_num) / self.fs

        if self.end > self.signal.shape[0]:
            self.end = self.signal.shape[0]

        for i in range(self.channel_num):
            axes[i, 0].plot(
                x[self.start : self.end], self.signal[self.start : self.end, i]
            )
            axes[i, 0].set_ylabel(self.ylabel[i])

            if i != self.channel_num - 1:
                axes[i, 0].tick_params(labelbottom=False)

        axes[-1, 0].set_xlabel('Time')
        plt.suptitle('EMG Channels')

        plt.subplots_adjust(hspace=0)
        # plt.show()
        plt.savefig(fig_path)

    def plot_freq_domain(self, fig_path):
        _, axes = plt.subplots(
            nrows=self.channel_num,
            sharex=True,
            squeeze=False,
            figsize=(20, self.channel_num * 1.5),
        )
        n = self.signal.shape[0]
        for i in range(self.channel_num):
            emg_f = np.fft.fft(self.signal[:, i])
            emg_f_abs = np.abs(emg_f)

            freq = np.fft.fftfreq(n, 1 / self.fs)

            half_n = n // 2
            half_freq = freq[:half_n]
            half_emg_f = emg_f_abs[:half_n]

            half_emg_f[0] = emg_f_abs[0]

            axes[i, 0].plot(half_freq, half_emg_f)
            axes[i, 0].set_ylabel(self.ylabel[i])

            if i != self.channel_num - 1:
                axes[i, 0].tick_params(labelbottom=False)

        axes[-1, 0].set_xlabel('Frequency')
        plt.suptitle('EMG Channels')
        plt.subplots_adjust(hspace=0)
        # plt.show()
        plt.savefig(fig_path)
