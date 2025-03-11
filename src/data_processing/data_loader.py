"""
@author: Jacob Patrick

@date: 2025-03-06

@email: lumivoxflow@gmail.com

@description: A script to load and get assigned modal data from raw data
"""

import os, yaml
import pandas as pd


def load_config(cfg_path):
    """
    load config file
    :param cfg_path: config file path
    :return: config
    """
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg
    except FileNotFoundError:
        print(f"Error: The file {cfg_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file: {e}")
    return None

def load_raw_data(file_name):
    """
    load data from csv file
    :param file_path: data file path
    :return: data
    """
    # get config info
    data_cfg = load_config('config/data_config.yml')
    data_dir = data_cfg['raw_data_cfg']['raw_data_dir']
    skiprows = data_cfg['raw_data_cfg']['skiprows']
    skipcols = data_cfg['raw_data_cfg']['skipcols']
    header = data_cfg['raw_data_cfg']['header']
    dtype = data_cfg['raw_data_cfg']['dtype']

    try:
        # read data
        data_path = os.path.join(data_dir, file_name)
        data_frame = pd.read_csv(
            data_path, skiprows=skiprows, header=header, dtype=dtype
        )
        all_columns = list(range(data_frame.shape[1]))
        valid_columns = [col for col in all_columns if col not in skipcols]
        data_frame = data_frame.iloc[:, valid_columns]
        data = data_frame.iloc[1:, :].values

        return data

    except FileNotFoundError:
        print(f"File {data_path} not found. Skip this file.")


def get_raw_modal_data(data, modal, new=True):
    """
    get raw data of assigned modal with time
    :param data: raw data
    :param modal: assigned modal
    :return: raw data of assigned modal with time
    """
    # get config info
    data_cfg = load_config('config/data_config.yml')
    modal_cols = data_cfg['raw_data_cfg']['data_modal']['new' if new else 'old']

    # get raw data of assigned modal
    if modal not in ['emg', 'acc', 'gyro', 'mag']:
        raise ValueError(f"Unknown sensor modality: {modal}. Please choose among 'emg', 'acc', 'gyro', 'mag'.")
    
    for data_modal, cols in modal_cols.items():
        if data_modal == modal:
            # insert time column
            cols.insert(0, 0)

            return data[:, cols]
        

def _test_cfg():
    data_cfg = load_config('config/data_config.yml')
    print(data_cfg)


if __name__ == '__main__':
    _test_cfg()