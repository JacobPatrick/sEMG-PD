from tsai.all import *
import torch
from sklearn.linear_model import RidgeClassifierCV

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# # 加载UCR数据集
# X, y, splits = get_UCR_data('Beef', return_split=False, on_disk=True, verbose=True)
# tfms  = [None, [Categorize()]]
# batch_tfms = [TSStandardize(by_sample=True)]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)

# # 标准示例
# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=768, drop_last=False, shuffle_train=False,
#                                device=device,
#                                batch_tfms=[TSStandardize(by_sample=True)])
# model = create_model(ROCKET, dls=dls)
# model = model.to(device)

# print("构造特征...")
# X_train, y_train = create_rocket_features(dls.train, model, verbose=False)
# X_valid, y_valid = create_rocket_features(dls.valid, model, verbose=False)
# print(X_train.shape, X_valid.shape)

# print("基于特征开始训练...")
# ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17))
# ridge.fit(X_train, y_train)
# print(f'alpha: {ridge.alpha_:.2E}  train: {ridge.score(X_train, y_train):.5f}  valid: {ridge.score(X_valid, y_valid):.5f}')
