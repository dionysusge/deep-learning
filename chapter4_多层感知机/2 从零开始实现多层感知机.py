import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 首先是一个单隐层，包含256个隐藏单元，然后一个全连接层映射到10个输出单元
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs)) # 展成二维，数据集是60000*28*28，所以变成60000*784
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none') # reduction参数用来调整计算损失的模式，none是为每个样本都计算，内存开销会比较大，可以用mean或者sum

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr) # 梯度下降去更新参数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)