# 汇聚层
# 降低卷积层对位置的敏感度、对空间降采样的敏感性

# 与卷积核类似，汇聚层的运算符也是一个窗口，在输入上滑动，计算一个输出
# 一般来说，是该窗口覆盖区域的平均值或者最大值，称为average pooling 或者 maximum pooling

# 实现的代码
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            if mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
# 验证最大池化层
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))

# 验证平均池化层
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d (X, (2, 2), 'avg'))


# 下面利用框架里的pooling layer
# 填充与步幅
X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
print(X)

# 3 * 3大小的池化窗口，默认情况下，池化的步幅和池化窗口的大小相同
pool2d = nn.MaxPool2d(3)
print(pool2d(X)) # 故此处输出只有一个元素

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

# 输入是多通道时，池化的输出仍为多通道，池化窗口对输入的每个通道单独处理
# 不像卷积层那样，不同通道有不同的卷积核，不同输出通道也可以有不同的卷积核
X = torch.cat((X, X + 1), 1) # 第二轴是通道，第一轴是第几个数据
print(X)
print(pool2d(X))

# 汇聚层的主要优点是减轻对位置的过度敏感，比如你说图像上像素移动了几个单位这种正常情况，用池化就很好地解决
