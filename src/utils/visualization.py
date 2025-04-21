import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List


def plot_confusion_matrix(
    label_true: List[int],
    label_pred: List[int],
    classes: List[str],
    title: str,
    save_path: str = None,
    dpi: int = 300,
) -> None:
    """
    绘制混淆矩阵，显示所有指定类别，并处理缺失数据

    Args:
        label_true: 真实样本标签序列
        label_pred: 预测样本标签序列
        classes: 类别列表，对应标签名
        title: 图标题
        save_path: 保存路径，默认为None，不保存
        dpi: 图像分辨率，默认300

    Returns:
        None
    """
    full_labels = list(range(len(classes)))

    try:
        cm = confusion_matrix(
            y_true=label_true,
            y_pred=label_pred,
            labels=full_labels,
            normalize='true',
        )
    except ValueError as e:
        # 对于全零行，手动归一化
        cm_raw = confusion_matrix(
            y_true=label_true, y_pred=label_pred, labels=full_labels
        )
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = cm_raw / row_sums
        cm = np.nan_to_num(cm)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("预测得分")
    plt.ylabel("实际得分")
    plt.xticks(range(classes.__len__()), classes, rotation=45)
    plt.yticks(range(classes.__len__()), classes)

    plt.tight_layout()

    plt.colorbar()

    thresh = cm.max() / 2.0
    for i in range(classes.__len__()):
        for j in range(classes.__len__()):
            color = (1, 1, 1) if cm[j, i] > thresh else (0, 0, 0)
            value = float(f"{cm[j, i]:.2f}")
            plt.text(i, j, value, ha="center", va="center", color=color)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    else:
        plt.show()
