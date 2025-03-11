"""
@author: Jacob Patrick

@date: 2025-03-06

@email: lumivoxflow@gmail.com

@description: Main script for the project
"""

import os
import numpy as np

from src.data_processing.data_loader import load_config, load_raw_data, get_raw_modal_data
from src.data_processing.pre_process import PreProcessor, segment_signal
from src.utils.plot_signal import SignalPlotter
from src.feature_engineering.features_extraction import (
    HandCraftedFeaturesExtractor,
)
from src.feature_engineering.basic_param import TremorParamExtractor

data_config = load_config('config/data_config.yml')
model_config = load_config('config/model_config.yml')
features_config = load_config('config/features_config.yml')
training_config = load_config('config/training_config.yml')


