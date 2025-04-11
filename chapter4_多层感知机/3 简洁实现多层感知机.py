import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(), # 可以指定展开的维度，start、end两个参数，第一个维度始终不变
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) # 调用nn模块里的初始化函数

net.apply(init_weights) # 递归地将指定的函数应用到其所有子模块上

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)