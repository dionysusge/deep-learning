# 先前AlexNet、VGG、LeNet的设计模式都是通过一系列卷积层去提取空间结构特征
# 随后通过全连接层对图像的表征进行处理
# AlexNet和VGG只是考虑了如何加深、且便捷地实现卷积操作
# 而在最后都使用了全连接层，这意味着输出的时候，放弃了表征的空间结构

# NiN提出了新的想法，在每个像素的通道上使用多层感知机

import torch
from torch import nn
from d2l import torch as d2l


# nin块中，除了常规的卷积层，还有两个1 * 1的卷积核
# 同前面章节提到的一样，1 * 1的卷积核用于不同通道、同一位置像素的线性组合，提取了信息，而不同输出通道的卷积核，提取的特征不一样
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

# 每个NiN块后有一个最大汇聚层，汇聚窗口形状为3 * 3，步幅为2
# NiN完全取消了全连接层，而是使用一个NiN块
# 其输出通道数等于标签类别的数量。最后放一个全局平均汇聚层（global average pooling layer），生成一个对数几率 （logits）
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 全局的平均池化（每个通道上所有元素）
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten()
)

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)


# 移除全连接层，最大的好处是减轻过拟合

# 下面是训练的过程
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())