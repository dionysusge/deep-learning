# 使用块的网络
"""
经典卷积神经网络的基本组成部分是下面的这个序列：
① 带填充以保持分辨率的卷积层；
② 非线性激活函数，如ReLU；
③ 汇聚层，如最大汇聚层。

而一个VGG块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层
"""

import torch
from torch import nn
from d2l import torch as d2l

# 构造VGG块，
def vgg_block(num_convs, in_channels, out_channels):
    # 返回的是一个nn模块的Sequential对象，包含了多个层
    layers = []
    # 根据需要的卷积层量，遍历增加，包含一个卷积层和一个激活函数
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    # 在块的最后，添加一个最大池化层，提取特征
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

# VGG网络包括了多个VGG块以及最后的连接层，其实架构上和AlextNet是类似的，前面的层多次卷积，提取不同特征放到不同输出通道上
# 最后同样也为全连接层，映射到不同的分类上

"""
原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 
第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。
由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11
"""
# 第一个参数是卷积层的数量，第二个参数是输出的通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

# 构建完整的VGG网络
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 整体的卷积部分，遍历卷积块
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # 在上面完成了卷积块的构建后，下面只需要解包（解包得到的各卷积层对象不是简单的Conv2d，而是包含了Conv2d、激活层（ReLU）以及最后的最大池化）
    # 随后展平，利用矩阵乘法全连接即可
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

# 下面利用VGG同样的去训练FashionMNIST数据集

# 减少一些输出的通道数，也足够用于训练该数据集
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# 使用了略高的学习率？居然0.05就高了哈哈哈
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

# 在论文的最后，提到使用略小一些的卷积操作是更有效的？应该指识别效果吧，但是对应了更大的计算量，不过更重要的应该还是步幅