# 深度卷积神经网络
# 对于先前的研究而言，其它机器学习的方法所用的数据集都是经过特征流水线处理的
# 而卷积神经网络直接将原始像素值组成，确实是有劣势的
# 比如：数据集小，训练神经网络的一些关键技巧仍然缺失，包括启发式参数初始化、随机梯度下降的变体、非挤压激活函数和有效的正则化技术

# 所以传统的机器学习流水线是这样的：
"""
获取一个有趣的数据集。在早期，收集这些数据集需要昂贵的传感器（在当时最先进的图像也就100万像素）。

根据光学、几何学、其他知识以及偶然的发现，手工对特征数据集进行预处理。

通过标准的特征提取算法，如SIFT（尺度不变特征变换） (Lowe, 2004)和SURF（加速鲁棒特征） (Bay et al., 2006)或其他手动调整的流水线来输入数据。

将提取的特征送入最喜欢的分类器中（例如线性模型或其它核方法），以训练分类器
"""

## 我觉得关键在于数据集的选择上，深度学习的神经网络，认为图像的特征本身也是需要去学习的部分，而不是人工直接分类好

# 特征应该由多个共同学习的神经网络层组成，每个层都有可学习的参数。在机器视觉中，最底层可能检测边缘、颜色和纹理

# 而更高层建立在这些底层表示的基础上，以表示更大的特征，更高的层可以检测整个物体
# 最终的隐藏神经元学习图像的综合表示


# AlexNet：
"""
AlexNet和LeNet的设计理念非常相似，但也存在显著差异

AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层

AlexNet使用ReLU而不是sigmoid作为其激活函数
"""
# sigmoid函数在输出接近0或1时，梯度几乎为0，那么当参数没有合理的初始化时，它永远无法更新

import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t', X, sep="\n")

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())