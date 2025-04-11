import torch
from torch import nn
from d2l import torch as d2l

# 卷积（互相关运算）定义：
def corr2d(X, K):
    # 将卷积核的高、宽取出
    h, w = K.shape
    # 初始化卷积后的张量尺寸信息
    Y = torch.zeros((X.shape[0] -h +1, X.shape[1] - w + 1))
    # 完成卷积结果运算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 逐个相乘后求和，得到某位置的互相关结果
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()

    return Y

# 验证上述互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

# 卷积层同样具有参数，也即卷积核权重和标量偏置，训练时，需要初始化该权重
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    # 前向传播函数利用上述的corr2d计算，并添加一定的偏置
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 图像中目标的边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
# 定义卷积核，注意下述区别
# 卷积核要为二维，下面的实现了边缘检测
K = torch.tensor([[1.0, -1.0]]) # 二维的，只有一行而已，如果两个相邻的元素相同则输出为0，但是只能检查“水平边缘”
K2 = torch.tensor([1.0, -1.0]) # 一维，有两个元素

Y = corr2d(X, K)
print(Y)

# 将X转置后再检测，便无法检测到垂直方向的边缘了
print(X.T)
print(corr2d(X.T, K))

# 学习卷积核，也即参数通过数据自行调整
# 因为在面对更多的卷积层、更多的通道数据时，已经无法再手动设计滤波器，像上述简单的黑白像素边缘检测那样

# 那么也需要输入、输出的数据，通过随机初始化后，检查平方误差，然后计算梯度来更新卷积核

# 定义一个卷积核，输出输入通道数都为1，尺寸为（1, 2）
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 用四维输入输出格式（批量大小、通道、高度、宽度）
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

# 迭代100次，误差基本降到0
for i in range(100):
    Y_hat = conv2d(X)
    loss = (Y_hat - Y) ** 2
    print(Y_hat)
    print(Y)
    conv2d.zero_grad()
    loss.sum().backward()

    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 50 == 0:
        print(f"epoch {i+1}, loss {loss.sum():.3f}")

# 感受野
# 如果卷积核的大小2*2，那么第一层的感受野为4，第二层输出时，感受野为9，第n层应该为2^n+1