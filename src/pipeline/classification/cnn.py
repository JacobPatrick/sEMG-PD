import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from src.interfaces.classification import Classification
from typing import Tuple, List


class CNN(Classification):
    def __init__(self, models=[]):
        self.models = models  # 每个输出维度对应一个CNN模型

    def fit(self, data: Tuple) -> List:
        X, y = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_samples, n_channels, n_features, n_windows = X.shape

        try:
            self.label_map_list = [{} for _ in range(y.shape[1])]
        except IndexError:  # 处理一维标签的情况
            self.label_map_list = [{}]

        # 为每个输出维度训练一个CNN模型
        self.models = []
        try:
            for i in range(y.shape[1]):  # 遍历每个输出维度
                unique_labels = np.unique(y[:, i])
                if len(unique_labels) < 2:  # 如果标签只有一个类别，则跳过
                    fixed_model = FixedOutputCNN(fixed_output=y[:, i][0])
                    self.models.append(ModelWrapper(model=fixed_model))
                    continue

                # (使用CNN分类器时)为不连续的标签创建一个从 0 开始的映射
                label_map = {
                    label: idx for idx, label in enumerate(unique_labels)
                }
                y_mapped = np.array(
                    [label_map[int(label)] for label in y[:, i]]
                )

                # 创建模型并训练
                cnn_model = CNN1DClassifier(
                    input_size=n_channels * n_features,
                    num_classes=len(unique_labels),
                    num_windows=n_windows,
                ).to(device)

                # 训练模型，含交叉验证
                cnn_model.fit_cv(
                    X, y_mapped, batch_size=32, num_epochs=50, lr=0.0001
                )
                self.models.append(ModelWrapper(model=cnn_model, label_map=label_map))

        except IndexError:  # 处理一维标签的情况
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:  # 如果标签只有一个类别，则使用固定输出
                fixed_model = FixedOutputCNN(fixed_output=y)
                self.models.append(ModelWrapper(model=fixed_model))
            else:
                # (使用CNN分类器时)为不连续的标签创建一个从 0 开始的映射
                label_map = {
                    label: idx for idx, label in enumerate(unique_labels)
                }
                y_mapped = np.array([label_map[label] for label in y])

                # 创建模型并训练
                cnn_model = CNN1DClassifier(
                    input_size=n_samples * n_features,
                    num_classes=len(unique_labels),
                    num_windows=n_windows,
                ).to(device)

                # 训练模型（含交叉验证）
                cnn_model.fit_cv(
                    X, y_mapped, batch_size=32, num_epochs=50, lr=0.0005
                )
                self.models.append(ModelWrapper(model=cnn_model, label_map=label_map))

        return self.models

    def predict_proba(self, data: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        X, y = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 对每个模型进行预测
        proba_predictions = []
        for model_wrapper in self.models:
            if isinstance(model_wrapper.model, FixedOutputCNN):  # 处理固定输出模型
                proba_pred = model_wrapper.predict_proba(X)
                proba_predictions.append(proba_pred)
            else:
                # 将输入转换为PyTorch张量并移动到设备上
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                proba_pred = model_wrapper.predict_proba(X_tensor)
                
                # 标签反向映射
                if model_wrapper.label_map:
                    # 确定最大标签值以创建足够大的数组
                    max_label = max(model_wrapper.label_map.keys())
                    proba_pred_true_label = np.zeros((proba_pred.shape[0], max_label + 1))
                    for label, idx in model_wrapper.label_map.items():
                        proba_pred_true_label[:, label] = proba_pred[:, idx]
                    proba_predictions.append(proba_pred_true_label)
                else:
                    proba_predictions.append(proba_pred)

        return y, np.array(proba_predictions).T  # 预测结果张量转置以匹配原始数据形状

    def predict(self, data: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        X, y = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 对每个模型进行预测
        predictions = []
        for model_wrapper in self.models:
            if isinstance(model_wrapper.model, FixedOutputCNN):  # 处理固定输出模型
                pred = model_wrapper.predict(X)
                predictions.append(pred)
            else:
                # 将输入转换为PyTorch张量并移动到设备上
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                pred = model_wrapper.predict(X_tensor)
                
                # 标签反向映射
                if model_wrapper.label_map:
                    reverse_map = {idx: label for label, idx in model_wrapper.label_map.items()}
                    pred = np.array([reverse_map[label] for label in pred])
                
                predictions.append(pred)

        return y, np.array(predictions).T  # 预测结果张量转置以匹配原始数据形状


class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes, num_windows):
        super(CNN1DClassifier, self).__init__()
        self.num_windows = num_windows
        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_size, self.num_windows)
            out = self.features(dummy)
            flat_size = out.shape[1] * out.shape[2]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))  # 合并手工特征的通道和特征维度
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

    def _to_tensor(self, X):
        """自动将输入转换为 tensor"""

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            pass
        else:
            raise ValueError("Input must be a torch.Tensor or numpy.ndarray")
        return X

    def fit_cv(self, X, y, batch_size=32, num_epochs=20, lr=0.001, n_splits=5):
        """含交叉验证的模型训练

        参数:
            X: 特征数据，可以是numpy数组或PyTorch张量
            y: 标签数据，可以是numpy数组或PyTorch张量
            batch_size: 批量大小
            num_epochs: 训练轮数
            lr: 学习率
            n_splits: 交叉验证折数
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # 保存初始权重以便在每个折上重置
        initial_weights = copy.deepcopy(self.state_dict())

        # 将数据转换为 PyTorch 张量
        X_tensor = self._to_tensor(X)
        y_tensor = (
            torch.tensor(y, dtype=torch.long)
            if isinstance(y, np.ndarray)
            else y
        )

        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor)

        # 定义分层交叉验证
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # criterion = nn.CrossEntropyLoss()
        criterion = OrdinalCrossEntropyLoss(alpha=0.5, num_classes=self.classifier[-1].out_features)
        best_acc = 0.0
        best_weights = None

        # 执行交叉验证
        for fold_idx, (train_ids, val_ids) in enumerate(
            skf.split(X_tensor.cpu().numpy(), y_tensor.cpu().numpy())
        ):
            print(f'FOLD {fold_idx + 1}')
            print('--------------------------------')

            # 在每个折开始前重置模型权重到初始状态
            self.load_state_dict(copy.deepcopy(initial_weights))

            # 创建训练和验证数据加载器
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)

            train_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler
            )
            val_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=val_subsampler
            )

            # 重新初始化优化器
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

            # 训练这一折
            fold_best_acc = 0.0
            fold_best_weights = None

            # 定义学习率调度器
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, patience=3
            # )

            # 定义早停规则
            patience_counter = 0
            patience_limit = 20  # 先放宽一点

            for epoch in range(num_epochs):
                self.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)   # 梯度裁剪（暂时不用）  
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                epoch_loss = running_loss / len(train_subsampler)
                epoch_acc = correct / total

                print(
                    f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}',
                    end='',
                )

                # 在验证集上评估
                self.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self(inputs)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                val_acc = val_correct / val_total
                val_loss = running_loss / len(val_subsampler)
                print(f', Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')

                # 早停策略
                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc
                    fold_best_weights = self.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        print(f'Early stopping at epoch {epoch+1}')
                        break

                # 保存这一折的最佳模型权重
                if val_acc > fold_best_acc:
                    fold_best_acc = val_acc
                    fold_best_weights = self.state_dict().copy()

                # 更新学习率
                # scheduler.step(val_loss * 5e-4)

            # 更新整体最佳模型
            if fold_best_acc > best_acc:
                best_acc = fold_best_acc
                best_weights = fold_best_weights

            print(
                f'Fold {fold_idx + 1} best validation accuracy: {fold_best_acc:.4f}'
            )

        # 加载最佳模型权重
        if best_weights:
            self.load_state_dict(best_weights)
            print(
                f'Training completed. Best validation accuracy: {best_acc:.4f}'
            )

        return self

    def predict_proba(self, X):
        """返回每个样本的类别概率估计"""

        self.eval()
        device = next(self.parameters()).device
        X_tensor = self._to_tensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=32)

        all_probs = []

        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(device)
                outputs = self(x_batch)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def predict(self, X):
        """返回每个样本的预测类别标签"""

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class FixedOutputCNN:
    """
    固定输出分类器

    当标签只有一个类别时使用，对所有输入都返回该固定类别
    """

    def __init__(self, fixed_output):
        self.fixed_output = fixed_output

    def to(self, device):
        """兼容PyTorch模型的to方法"""
        return self

    def fit_cv(self, X, y, **kwargs):
        """空实现以保持接口一致性"""
        return self

    def predict_proba(self, X):
        """返回概率预测结果"""
        n_samples = X.shape[0]
        # 生成只有一个类的概率分布
        probas = np.zeros((n_samples, 5))
        probas[:, self.fixed_output] = 1.0
        return probas

    def predict(self, X):
        """对所有输入预测同一个类别"""
        return np.full((X.shape[0],), self.fixed_output)


class ModelWrapper:
    """模型包装类，封装模型和对应的标签映射
    
    属性:
        model: CNN1DClassifier 或 FixedOutputCNN 的实例
        label_map: 可选的标签映射字典
    """
    def __init__(self, model, label_map=None):
        self.model = model
        self.label_map = label_map
    
    def to(self, device):
        """将模型移动到指定设备"""
        self.model.to(device)
        return self
    
    def fit_cv(self, X, y, **kwargs):
        """训练模型"""
        return self.model.fit_cv(X, y, **kwargs)
    
    def predict_proba(self, X):
        """返回概率预测结果"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """返回预测类别"""
        return self.model.predict(X)


class OrdinalCrossEntropyLoss(nn.Module):
    """
    有序交叉熵损失函数
    """

    def __init__(self, alpha=0.5, num_classes=None):
        """
        Args:
            alpha: 损失函数的权重参数，空值距离损失和交叉熵损失的比例
            num_classes: 类别数量，用于归一化距离损失
        """
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        # 计算标准交叉熵损失
        ce_loss = self.ce_loss(outputs, targets)
        # 计算距离损失
        softmax_probs = F.softmax(outputs, dim=1)
        class_indices = torch.arange(outputs.size(1), device=outputs.device).float()
        expected_classes = torch.sum(softmax_probs * class_indices.unsqueeze(0), dim=1)
        distance_loss = torch.abs(expected_classes - targets.float())
        # 归一化距离损失项
        if self.num_classes and self.num_classes > 1:
            distance_loss = distance_loss / (self.num_classes - 1)
        
        combined_loss = ce_loss + self.alpha * distance_loss

        return combined_loss.mean()
