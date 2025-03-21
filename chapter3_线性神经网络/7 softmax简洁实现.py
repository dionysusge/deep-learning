import torch
from torch import nn
from d2l import torch as d2l

# 读取mnist数据集
"""
这里大概复习一下读取操作吧
读取的数据集Fashion_MNIST是来自torchvision里的数据集，这个数据集里还有其它很多很多东西
定义的函数，利用torchvision.datasets.xxx()，里面有一个train参数，通过这个来设置读取训练集还是数据集，构建一个数据集类吧
然后通过torch.utils库里的data库，用DataLoader对象加载数据集，设置批量、打乱（shuffle参数），还可以设置用于读取数据的线程数
可以转为python内置的iter对象，单次读取，也可以通过for循环读取
第一个是原数据，比如像素点信息，第二个是标签，对应什么类别
"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# 尺寸考虑的是像素点以及最后预测得到各类别的数据
# 利用nn创建一个网络，有两层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 损失函数
# 为了避免最大项产生误差，在softmax里取指数之前，会减去最大的一个y值
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.predict_ch3(net, test_iter, 8)