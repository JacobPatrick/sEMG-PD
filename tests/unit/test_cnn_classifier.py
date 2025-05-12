import os
import sys
import numpy as np
import torch
import pytest

# 添加源代码路径
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
)

from src.pipeline.classification.cnn import CNN, FixedOutputCNN


def test_cnn_multi_task_classification():
    """
    测试 CNN 类处理多分类任务的能力

    生成一个形状为 (n_samples, n_channels, n_features, n_windows) 的特征张量
    和一个形状为 (n_samples, n_labels) 的标签张量
    """
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)

    # 配置参数
    n_samples = 50  # 样本数量
    n_channels = 8  # 通道数量（例如EMG传感器通道）
    n_features = 5  # 每个通道的特征数量（例如手工特征）
    n_windows = 10  # 窗口数量
    n_labels = 3  # 标签任务数量

    # 创建随机特征张量 - 确保与CNN预期的输入格式匹配
    X = np.random.rand(n_samples, n_channels * n_features, n_windows).astype(
        np.float32
    )

    # 创建多个分类任务的标签张量
    # 第一个任务：二分类（0或1）
    y1 = np.random.randint(0, 2, size=(n_samples,))

    # 第二个任务：三分类（0, 1或2）
    y2 = np.random.randint(0, 3, size=(n_samples,))

    # 第三个任务：只有一个类别（全部为0）
    y3 = np.zeros((n_samples,), dtype=int)

    # 组合为一个标签矩阵
    y = np.column_stack((y1, y2, y3))

    # 初始化CNN分类器
    cnn = CNN()

    # 训练模型，设置较小的轮数以加快测试速度
    models = cnn.fit((X, y))

    # 断言检查
    assert (
        len(models) == n_labels
    ), f"应该创建 {n_labels} 个模型，但实际创建了 {len(models)} 个"

    # 检查第一个模型（二分类任务）
    assert not isinstance(
        models[0], FixedOutputCNN
    ), "第一个模型不应该是FixedOutputCNN"

    # 检查第二个模型（三分类任务）
    assert not isinstance(
        models[1], FixedOutputCNN
    ), "第二个模型不应该是FixedOutputCNN"

    # 检查第三个模型（单类别任务）
    assert isinstance(
        models[2], FixedOutputCNN
    ), "第三个模型应该是FixedOutputCNN"
    assert models[2].fixed_output == 0, "第三个模型的固定输出应该是0"

    # 使用模型进行预测
    X_test = np.random.rand(10, n_channels * n_features, n_windows).astype(
        np.float32
    )

    # 对每个任务进行预测
    for i, model in enumerate(models):
        preds = model.predict(X_test)
        assert len(preds) == 10, f"预测的样本数应该是10，但得到了{len(preds)}"

        # 对于第三个任务，所有预测结果应该都是0
        if i == 2:
            assert np.all(preds == 0), "第三个任务的所有预测结果应该都是0"

    print("CNN 多任务分类测试通过!")


def test_single_label_classification():
    """测试CNN类处理单标签分类任务的能力"""
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)

    # 配置参数
    n_samples = 30  # 样本数量
    n_channels = 4  # 通道数量
    n_features = 3  # 每个通道的特征数量
    n_windows = 8  # 窗口数量

    # 创建随机特征张量
    X = np.random.rand(n_samples, n_channels * n_features, n_windows).astype(
        np.float32
    )

    # 创建单个分类任务的标签
    y = np.random.randint(0, 3, size=(n_samples,))

    # 初始化CNN分类器
    cnn = CNN()

    # 训练模型
    models = cnn.fit((X, y))

    # 断言检查
    assert len(models) == 1, "应该创建1个模型"
    assert not isinstance(
        models[0], FixedOutputCNN
    ), "模型不应该是FixedOutputCNN"

    # 使用模型进行预测
    X_test = np.random.rand(5, n_channels * n_features, n_windows).astype(
        np.float32
    )
    preds = models[0].predict(X_test)

    assert len(preds) == 5, f"预测的样本数应该是5，但得到了{len(preds)}"

    print("CNN 单标签分类测试通过!")


if __name__ == "__main__":
    # 运行测试
    print("开始测试CNN分类器...")
    test_cnn_multi_task_classification()
    test_single_label_classification()
    print("所有测试通过!")
