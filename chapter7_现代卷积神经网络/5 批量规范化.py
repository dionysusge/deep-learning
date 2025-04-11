# 批量规范化简介
"""
批量规范化应用于单个可选层（也可以应用到所有层）
其原理如下：
① 在每次训练迭代中，我们首先规范化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。
② 应用比例系数和比例偏移

应用标准化后，生成的小批量的平均值为0和单位方差为1
通常包含 拉伸参数（scale）和偏移参数（shift）
它们的形状与输入相同。是需要与其他模型参数一起学习的参数

标准化可以很好地与我们的优化器配合使用，因为它可以将参数的量级进行统一
变量分布中的偏移可能会阻碍网络的收敛（出现一个比较不合理的值后，会被逐层扩大）
更深层的网络很复杂，容易过拟合。 这意味着正则化变得更加重要。

批量规范化最适应 50~100 范围中的中等批量大小
"""

# 下面介绍批量标准化在实际中是怎么应用的

"""
全连接层和卷积层的规范化有所不同
批量规范化在完整的小批量上运行，因此我们不能像以前在引入其他层时那样忽略批量大小

1、对于全连接层
将批量规范化层置于全连接层中的仿射变换和激活函数之间

2、对于卷积层
在卷积层之后和非线性激活函数之前应用批量规范化。 
当卷积有多个输出通道时，我们需要对这些通道的“每个”输出执行批量规范化
每个通道都有自己的拉伸（scale）和偏移（shift）参数，这两个参数都是标量
"""

# 下面动手实现批量规范化
import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y