import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import euclidean


class Autoencoder(nn.Module):
    """
    自编码器模型
    
    输入：形如 (n_samples, n_channels) 的一对疗前疗后数据
    
    输出：治疗前后的低维特征及其欧式距离
    """
    def __init__(self, pre_data, post_data):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(pre_data.shape[1], 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, pre_data.shape[1]),
            nn.Sigmoid()
        )
        # 数据归一化
        scaler = MinMaxScaler()
        self.pre_data_scaled = scaler.fit_transform(pre_data)
        self.post_data_scaled = scaler.transform(post_data)

        # 将数据转换为 PyTorch 张量
        self.pre_data_tensor = torch.tensor(self.pre_data_scaled, dtype=torch.float32)
        self.post_data_tensor = torch.tensor(self.post_data_scaled, dtype=torch.float32)

        # 创建数据集和数据加载器
        pre_dataset = TensorDataset(self.pre_data_tensor)
        self.pre_dataloader = DataLoader(pre_dataset, batch_size=32, shuffle=True)
        
		# 初始化特征
        self.pre_features = None
        self.post_features = None

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self, n_epochs=50, learning_rate=0.001):
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            running_loss = 0.0
            for data in self.pre_dataloader:
                inputs = data[0]
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(self.pre_dataloader)}')

    def compare(self):
        # 提取低维特征
        with torch.no_grad():
            pre_features = self.encoder(self.pre_data_tensor).numpy()
            post_features = self.encoder(self.post_data_tensor).numpy()

        average_distance = np.mean([euclidean(pre_features[i], post_features[i]) for i in range(len(pre_features))])
        print("治疗前后低维特征的平均欧氏距离:", average_distance)
        self.pre_features = pre_features
        self.post_features = post_features

    def visualize(self):
        if self.pre_features is None or self.post_features is None:
            raise ValueError("特征向量为空值，请先调用 compare() 方法进行特征提取和比较。")
            return
        plt.scatter(self.pre_features[:, 0], self.pre_features[:, 1], label='Pre - treatment')
        plt.scatter(self.post_features[:, 0], self.post_features[:, 1], label='Post - treatment')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # 生成示例数据
    pre_data = np.random.rand(10000, 8)
    post_data = np.random.rand(10000, 8)

    model = Autoencoder(pre_data, post_data)
    model.train()
    model.compare()
    model.visualize()
