import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import List


colors = {
    "red1": "#B72230",
    "red2": "#DC6D57",
    "blue1": "#317CB7",
    "blue2": "#6DADD1",
}


def plot_confusion_matrix(
    label_true: List[int],
    label_pred: List[int],
    classes: List[str],
    title: str = None,
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

    # try:
    #     cm = confusion_matrix(
    #         y_true=label_true,
    #         y_pred=label_pred,
    #         labels=full_labels,
    #         normalize='true',
    #     )
    # except ValueError as e:
    #     # 对于全零行，手动归一化
    #     cm_raw = confusion_matrix(
    #         y_true=label_true, y_pred=label_pred, labels=full_labels
    #     )
    #     row_sums = cm_raw.sum(axis=1, keepdims=True)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         cm = cm_raw / row_sums
    #     cm = np.nan_to_num(cm)

    # plt.rcParams["font.sans-serif"] = ["FangSong"]
    # plt.rcParams["font.size"] = 12
    # plt.rcParams["axes.unicode_minus"] = False

    # plt.figure(figsize=(3, 3))

    # plt.imshow(cm, cmap="Blues")
    # # plt.title(title)
    # plt.xlabel("预测得分")
    # plt.ylabel("真实得分")
    # plt.xticks(range(classes.__len__()), classes, rotation=45)
    # plt.yticks(range(classes.__len__()), classes)

    # plt.tight_layout()

    # # plt.colorbar()

    # thresh = cm.max() / 2.0
    # for i in range(classes.__len__()):
    #     for j in range(classes.__len__()):
    #         color = (1, 1, 1) if cm[j, i] > thresh else (0, 0, 0)
    #         value = float(f"{cm[j, i]:.2f}")
    #         plt.text(i, j, value, ha="center", va="center", color=color)

    # if save_path:
    #     plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    # else:
    #     plt.show()

    # plt.close()

    # 计算原始混淆矩阵（不归一化）
    cm_raw = confusion_matrix(
        y_true=label_true, y_pred=label_pred, labels=full_labels
    )
    
    # 计算总样本数
    total_samples = np.sum(cm_raw)
    
    # 计算每个单元格的全局百分比
    cm_percent = cm_raw / total_samples
    
    # 行归一化（每行代表真实类别，归一化后表示该真实类别被预测为各类别的比例）
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = cm_raw / row_sums
    cm = np.nan_to_num(cm)
    
    # 注意：由于需要交换轴，这里需要转置混淆矩阵
    # 转置前：行是真实值，列是预测值
    # 转置后：行是预测值，列是真实值（符合你要求的x轴为真值，y轴为预测值）
    cm = cm.T
    cm_percent = cm_percent.T
    
    plt.rcParams["font.sans-serif"] = ["FangSong"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(4, 4))

    plt.imshow(cm, cmap="Blues")
    if title:
        plt.title(title)
    plt.xlabel("真实得分")  # 交换后x轴为真实标签
    plt.ylabel("预测得分")  # 交换后y轴为预测标签
    plt.xticks(range(classes.__len__()), classes)
    plt.yticks(range(classes.__len__()), classes)

    plt.tight_layout()

    thresh = cm.max() / 2.0
    for i in range(classes.__len__()):  # i是真实类别（现在是x轴）
        for j in range(classes.__len__()):  # j是预测类别（现在是y轴）
            # 确定文本颜色
            color = (1, 1, 1) if cm[j, i] > thresh else (0, 0, 0)
            
            # 获取归一化值（预测为j类别的i类别样本比例）
            norm_value = float(f"{cm[j, i]:.2f}")
            
            # 获取全局百分比（占总样本的百分比）
            percent = cm_percent[j, i] * 100
            
            # 组合文本显示
            text = f"{norm_value:.2f}\n({percent:.1f}%)"
            
            # 在单元格中显示文本
            plt.text(i, j, text, ha="center", va="center", color=color, fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    else:
        plt.show()

    plt.close()


def plot_training_curve(model, save_path=None, title=None, figsize=(3, 3)):
    """绘制训练曲线，左侧y轴显示loss，右侧y轴显示accuracy

    参数:
        model: 训练好的模型，必须实现get_training_history()方法
        save_path: 图像保存路径，如果为None则不保存
        title: 图像标题，默认为None
        figsize: 图像大小，默认为(5, 4)
    """
    history = model.get_training_history()
    if not history:
        print("没有可用的训练历史记录")
        return None

    plt.rcParams["font.sans-serif"] = ["FangSong"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.unicode_minus"] = False

    # 创建具有较大右边距的图形和坐标轴
    fig = plt.figure(figsize=figsize)
    # 调整子图边距，增加右边距以容纳准确率为1的情况
    ax1 = fig.add_subplot(111)

    # 左侧Y轴 - 损失
    max_loss = max(max(history['train_loss']), max(history['val_loss']))
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失函数值')
    ax1.set_ylim(0, max_loss * 1.05)
    ax1.plot(
        history['train_loss'],
        color=colors['blue1'],
        linestyle='-',
        label='训练损失',
    )
    ax1.plot(
        history['val_loss'],
        color=colors['red1'],
        linestyle='-',
        label='验证损失',
    )
    ax1.tick_params(axis='y')

    # 右侧Y轴 - 准确率
    ax2 = ax1.twinx()
    # 准确率范围保持0-1，但确保刻度标签能够显示
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks(np.arange(0, 1.2, 0.2))
    ax2.set_ylabel('准确率')
    ax2.plot(
        history['train_acc'],
        color=colors["blue1"],
        linestyle='--',
        label='训练准确率',
    )
    ax2.plot(
        history['val_acc'],
        color=colors['red1'],
        linestyle='--',
        label='验证准确率',
    )
    ax2.tick_params(axis='y')

    # 确保两个轴的图例都显示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # 添加标题
    if title:
        plt.title(title)

    # 调整布局，增加顶部的空间
    plt.subplots_adjust(top=0.9)

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"训练曲线已保存至 {save_path}")

    return fig
