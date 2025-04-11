# 填充
# 在行、列填充了多大就是卷积运算后的结果行、列增加多大

import torch
from torch import nn


# 定义一个卷积函数
def comp_conv2d(conv2d, X):
    X = X.reshape((1,1)+X.shape) # 转换为单个，单通道
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # shape参数的后两个，也就是整合了通道数以及图片数

# 直接通过torch里的nn模块使用Conv2d类构造卷积核
# 单通道输入输出，卷积核大小为3*3，padding参数可以是一个元组，指定上下、左右填充的大小，如果只是一个int，那填充的一样
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
Y = comp_conv2d(conv2d, X)
print(conv2d)
print(X)
print(Y)
print(Y.shape)

# 步幅，原尺寸减去卷积核大小，加上填充大小再加步幅后，除以步幅，向下取整就得到了输出的尺寸
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
Y = comp_conv2d(conv2d, X)
print(Y.shape)

# 看一个略复杂一点的例子（综合的）
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
# 那么对于水平（宽度）方向来说，（8 + 2 - 5 + 4）/ 2 = 2，垂直（高度）方向来说：(8 - 3 + 0 + 3) / 3 = 2
# 下面的输出结果，应当为2 * 2的尺寸
Y = comp_conv2d(conv2d, X)
print(conv2d.weight.data)
print(Y)