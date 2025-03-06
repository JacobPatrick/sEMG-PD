"""
@author: Jacob Patrick

@date: 2025-03-06

@email: jacob_patrick@163.com

@description: A script for data loader
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


def load_raw_data(file_name, cfg_path='config/data_config.yml'):
    """
    load data from csv file
    :param file_path: data file path
    :return: data
    """
    # get config info
    data_cfg = load_config(cfg_path)
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

    except FileNotFoundError as e:
        print(f"文件 {data_path} 未找到，跳过该文件。")
