import json
import numpy as np


# 将np.array类型数据转换为可被json解码的类型
def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)


def save_dict_to_json(dictionary, file_path):
    """将字典保存为JSON文件"""

    try:
        with open(file_path, 'w') as f:
            json.dump(
                dictionary, f, indent=2, default=convert_to_json_serializable
            )
        print(f"字典已保存到 {file_path}")
    except Exception as e:
        print(f"保存字典时出错: {e}")
