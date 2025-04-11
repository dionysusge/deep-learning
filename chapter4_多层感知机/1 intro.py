# 前面的仿射变换中的线性是个很强的假设，实际中，很多变量也许是单调的，但不一定线性相关
# 比如图像分类，用单个像素去决定分类结果其实比较荒谬，我们更希望考察其周围像素再加权的结果

# 引入隐藏层，在进行仿射变换后，对每个隐藏单元应用非线性的激活函数，就不再退化为线性模型
# 可以有多个非线性层，从而产生更有表达能力的模型
# 线性容易与现实世界矛盾，很多东西并不是单调的

# 为什么隐藏层需要是非线性的，如果隐藏层为线性，那其实最后整理完还是变换为线性模型，而且还多了参数
# 因此隐藏层要引入非线性的激活函数

# 虽然一个单隐层网络能学习任何函数， 但并不意味着我们应该尝试使用单隐藏层网络来解决所有问题。
# 事实上，通过使用更深（而不是更广）的网络(也即多个层)，我们可以更容易地逼近许多函数。

# 下面展示三个激活函数，引入非线性


import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 设置画布，包含3行2列共6个子图
fig, axes = plt.subplots(3, 2, figsize=(10, 7.5))

# ReLU
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', axes=axes[0, 0])

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', axes=axes[0, 1])

# sigmoid及其导数
x.grad.data.zero_()
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', axes=axes[1, 0])

x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', axes=axes[1, 1])

# tanh及其导数
x.grad.data.zero_()
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', axes=axes[2, 0])

x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', axes=axes[2, 1])

# 调整子图布局
plt.tight_layout()
# 显示画布
plt.show()

# ... existing code ...