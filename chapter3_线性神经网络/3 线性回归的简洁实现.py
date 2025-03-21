# 主要讨论利用pytorch里的框架，自动化基于梯度的学习算法中重复的工作

# 生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels =d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """
    构造一个PyTorch数据迭代器，用于批量加载数据。

    参数:
    data_arrays (tuple): 包含多个张量的元组，例如 (features, labels)。
    batch_size (int): 每个小批量数据的样本数量。
    is_train (bool, 可选): 是否为训练阶段，默认为 True。如果为 True，数据将被打乱。

    返回:
    torch.utils.data.DataLoader: 一个数据加载器对象，用于迭代批量数据。
    """
    # 创建一个 TensorDataset 对象，将输入的多个张量组合成一个数据集，data_arrays原先是个二维张量
    # 在利用data模块的tensordataset后，变成了一个数据集的嵌套元组迭代器，每个元素是每条轴单个元素构成的张量集合
    dataset = data.TensorDataset(*data_arrays)
    # 返回一个 DataLoader 对象，用于批量加载数据
    # shuffle=is_train 表示如果是训练阶段，数据将被打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 返回一个DataLoader对象

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter))) # 利用iter去构造迭代器，使用next从迭代器中获得第一项

# 定义模型，net参数是一个网络，可以有多层处理，下面只用了一层线性变换，也即全连接

# Sequential类将多个层串联到一起，当给定输入数据时，将数据传入第一层，然后将第一层的输出作为第二层的输入
# 全连接层，线性变化，核心是矩阵乘法

# nn是neutral network 的简称
from torch import nn
net = nn.Sequential(nn.Linear(2,1, True)) # 输入、输出特征，是否有偏置

# 初始化模型参数
# 使用索引可以在net中选择处理层
net[0].weight.data.normal_(0, 0.01) # data是指访问张量的数据部分，而下划线结尾表示为一个原地操作方法
net[0].bias.data.fill_(0)

# 定义损失函数
# 使用MSELoss类，叫做平方L2范数，默认情况下返回所有样本损失的平均值
# L1Loss是绝对值，对于异常值不敏感
loss = nn.MSELoss()

# 定义优化算法
# 使用torch里的optim库
trainer = torch.optim.SGD(net.parameters(), lr = 0.03) # parameters里包含了w和b两个参数，张量形式，可以梯度运算

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss: {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)