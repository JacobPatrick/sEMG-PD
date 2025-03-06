"""
@author: Jacob Patrick

@date: 2025-01-16

@email: jacob_patrick@163.com

@description: A helper class for semg signal pre-processing
"""
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt


class PreProcessor:
	"""
	A helper class for semg signal pre-processing
	"""
	def __init__(self, data, data_config):
		# 8 channels
		self.data = data
		self.time = data[:, 0]
		self.model = {}

		for data_type, lines in data_config.items():
			if data_type in ['emg', 'acc', 'gyro', 'mag']:
				self.model[data_type] = data[:, lines]
			else:
				raise ValueError(f"Unknown data type: {data_type}")

	def get_raw_emg(self):
		"""
		get the raw emg data
		"""
		return self.model['emg']

	def filter_emg(self, fs=2000, lowcut=10.0, highcut=500.0, order=4, notch_freq=50.0, Q=30.0):
		"""
		filter the emg signal
		"""
		if self.model['emg'] is None:
			raise ValueError("EMG data is not loaded")
		
		b_notch, a_notch = iirnotch(notch_freq / (fs / 2), Q)
		b_butter, a_butter = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')

		for i in range(self.model['emg'].shape[1]):
			self.model['emg'][:, i] = filtfilt(b_butter, a_butter, self.model['emg'][:, i])
			self.model['emg'][:, i] = filtfilt(b_notch, a_notch, self.model['emg'][:, i])

		return self.model['emg']
	
	def filter_acc(self, fs=200, highcut=50.0):
		"""
		filter the acc signal
		"""
		if self.model['acc'] is None:
			raise ValueError("ACC data is not loaded")

		b_butter, a_butter = butter(4, highcut / (fs / 2), btype='low')
		for i in range(self.model['acc'].shape[1]):
			self.model['acc'][:, i] = filtfilt(b_butter, a_butter, self.model['acc'][:, i])
		
		return self.model['acc']
	
	def segment_emg(self, window_size=200, overlap=0.5):
		"""
		segment the EMG data into windows
		"""
		if self.model['emg'] is None:
			raise ValueError("EMG data is not loaded")
			return

		step_size = int(window_size * (1 - overlap))
		segments = []
		for i in range(0, self.model['emg'].shape[0] - window_size + 1, step_size):
			segment = self.model['emg'][i:i + window_size, :]
			segments.append(segment)

		return segments
	
	def sample_emg(self, fs=2000, duration=5, num_samples=6):
		"""
		sample the EMG data
		"""
		if self.model['emg'] is None:
			raise ValueError("EMG data is not loaded")
			return
		
		samples = []
		# 随机采样，其实是不对的，尤其是运动阶段，应该采样信号能量集中的时间段
		for _ in range(num_samples):
			start = np.random.randint(0, self.model['emg'].shape[0] - duration * fs)
			end = start + duration * fs
			sample = self.model['emg'][start:end, :]
			samples.append(sample)
		
		return samples
	

def segment(data, window_size=200, overlap=0.5):
	"""
	segment the EMG data into windows
	"""
	if data is None:
		raise ValueError("EMG data is not loaded")
	
	step_size = int(window_size * (1 - overlap))
	segments = []
	for i in range(0, data.shape[0] - window_size + 1, step_size):
		segment = data[i:i + window_size, :]
		segments.append(segment)

	return segments


# if __name__ == "__main__":
# 	# load configration
# 	config = load_json_config('config.json')

# 	raw_data_path = config['data']['raw_path']
# 	processed_data_path = config['data']['processed_path']
# 	data_config = config['data_config']

# 	# read raw data and pre-process
# 	processed_emg = PreProcessor(read_data, data_config)

# 	# filter the EMG data
# 	filtered_emg = processed_emg.filter_emg()

# 	# Take any data within 1 second for plotting
# 	duration = 1
# 	sample_rate = 2000
# 	start = np.random.randint(0, filtered_emg.shape[0] - duration * sample_rate)
# 	end = start + duration * sample_rate

# 	channel_num = filtered_emg.shape[1]
# 	fig, axes = plt.subplots(nrows=channel_num, sharex=True, squeeze=False, figsize=(10, channel_num))

# 	x = filtered_emg.index
# 	for i in range(channel_num):
# 		axes[i, 0].plot(x, filtered_emg.iloc[start:end, i])
# 		axes[i, 0].set_ylabel(f'Channel {i+1}')

# 		if i != channel_num - 1:
# 			axes[i, 0].tick_params(labelbottom=False)

# 	axes[-1, 0].set_xlabel('Time')
# 	plt.suptitle('EMG Channels')

# 	plt.subplots_adjust(hspace=0)
# 	plt.show()
