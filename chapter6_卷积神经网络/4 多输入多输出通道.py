# 对于多通道的输入来说，卷积核相应的也要为多通道
# 都是通过增加张量的维度来实现的，相同维度的相同索引张量进行运算
# 输出通道也可以进行指定，如果输出通道为1，那么将各输入通道卷积运算后的结果相加即可

import torch
from d2l import torch as d2l

# 2维互相关运算，correlation 2d
def corr2d_multi_in(X, K):
    # 这个zip函数，是将最顶层的索引封装起来
    for x, k in zip(X, K):
        print(x, k)
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

# 定义输入和卷积核
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

# 多输出通道
# 每一层有多个输出通道是至关重要的，这意味着把每个通道看作对不同特征的响应，更利于实现特征提取
# 并且，深度学习过程，每个通道都会起到作用，这累加起来使得最后的实际情况会比较麻烦

# 可以为每个输出通道都创建一个n = channel_in * size_in的卷积核张量
# 那么c_out个输出通道就有c_out * n个卷积核张量
# 确实，这样才真正实现了进行不同特征的提取，而不是说一张图的不同部分。这种情况下，每个卷积核都对图像不同输入通道的不同特征进行了处理

# 下面进行实现
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，也即遍历输出通道对应的卷积核，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 通过将核张量K、K+1（K中每个元素加1）、K+2连接起来，构造了一个3输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
print(K)
# 下面得到了三个通道的输出结果，每个输入通道与不同的卷积核进行作用
print(corr2d_multi_in_out(X, K))


# 1 * 1卷积层
# 如果是单通道，那么1 * 1卷积层显然毫无意义
# 但是在多通道输入的情况下，1 * 1的卷积层相当于对相同位置的像素做了线性叠加，可以看作每个像素位置的全连接层
# 然后既然是层，输出的通道数也可以是很多，也就是多个卷积核去提取特征

# 下面使用全连接层进行实现1 * 1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # 对输入、输出的形状进行调整，铺平像素，由于是1 * 1的卷积核，直接忽略掉，保留输出通道和输入通道大小即可
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    print(K.shape)
    # 得到每个输出通道上的一维像素
    # 比如 2 * 3 @ 3 * 100 => 2 * 100
    Y = torch.matmul(K, X)
    # 像素转换为二维形式
    return Y.reshape(c_o, h, w)

print("-" * 50)
X = torch.normal(0, 1, (3, 3, 3)) # 调用torch里按正态分布生成的函数，u = 0， cigma = 1， 尺寸为3 * 3 * 3，3通道、宽高均为3
K = torch.normal(0, 1, (2, 3, 1, 1)) # 输出为2通道，每个卷积核要处理三个输入通道，卷积核大小为1 * 1

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
# assert float(torch.abs(Y1 - Y2).sum()) < 1e-6