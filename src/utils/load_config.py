# import yaml
import json

# def load_yaml_config(file_path):
#     with open(file_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

def load_json_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config