import yaml


def load_config(cfg_path):
    """
    load config file
    :param cfg_path: config file path
    :return: config
    """
    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    except FileNotFoundError:
        print(f"Error: The file {cfg_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file: {e}")
    return None
