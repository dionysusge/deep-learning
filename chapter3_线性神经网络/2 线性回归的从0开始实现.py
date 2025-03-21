# 先回顾一下上节重点
"""
机器学习模型中的关键要素是训练数据、损失函数、优化算法，还有模型本身。

矢量化使数学表达上更简洁，同时运行的更快。

最小化目标函数和执行极大似然估计等价。

线性回归模型也是一个简单的神经网络。
"""

# 从0开始实现整个方法，包括数据流水线、模型、损失函数和小批量随机梯度下降优化器
import random
import torch
from d2l import torch as d2l

# 生成数据集，参数w=[2, -3.4]T, b=4.2，噪声项服从正态分布
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b # 这里是矩阵和向量相乘，利用广播机制，逐个求和后降维，得到一个向量
    y += torch.normal(0, 0.01, y.shape) # 加上和y的形状相同的噪音，噪音也服从正态分布
    return X, y.reshape((-1, 1)) # y列数变成一列，行数自适应

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 生成散点图，可以观察到因变量和第i个特征的关系
d2l.set_figsize()
# s参数表示散点的大小，为标量时每个散点大小相同，可以为数组
d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1, label='Feature 0')
# 新增代码，绘制第二个特征与因变量的散点图
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1, label='Feature 1')

# 显示图例
d2l.plt.legend()
# d2l.plt.show()


# 读取数据集，打乱数据集中的样本并以小批量的方式获取
def data_iter(batch_size, features, labels):
    """
    该函数用于将特征和标签数据划分为小批量，以便进行小批量随机梯度下降。

    参数:
    batch_size (int): 每个小批量的样本数量。
    features (torch.Tensor): 特征数据，形状为 (num_examples, num_features)。
    labels (torch.Tensor): 标签数据，形状为 (num_examples, 1)。

    返回:
    生成器，每次迭代返回一个小批量的特征和标签数据。
    """
    # 获取样本总数
    num_examples = len(features)
    # 生成从 0 到 num_examples - 1 的索引列表
    indices = list(range(num_examples))
    # 随机打乱索引列表，确保样本随机读取
    random.shuffle(indices)
    # 从 0 开始，以 batch_size 为步长遍历索引列表
    for i in range(0, num_examples, batch_size):
        # 截取当前批次的索引，确保不超出样本总数
        # 下面这行语句构造了一个索引张量
        batch_indices = torch.tensor( indices[i: min(i + batch_size, num_examples)] )

        # 张量可以被索引张量截取，但是不能通过列表截取
        # 使用截取的索引获取当前批次的特征和标签数据，并通过生成器返回
        yield features[batch_indices], labels[batch_indices]

# # 调用 data_iter 函数，返回一个生成器对象
# batch_size = 10
# data_generator = data_iter(batch_size, features, labels)
#
# # 使用 for 循环迭代生成器对象
# for batch_features, batch_labels in data_generator:
#     print(f"Batch features shape: {batch_features.shape}, Batch labels shape: {batch_labels.shape}")

# GPU并行运算适合用来处理合理大小的“小批量”，每个样本可以并行地进行计算，每个样本损失函数的梯度也可以被并行计算，那处理一百个样本并不比处理一个样本的时间多太多
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, "\n", y)
    break
# 在深度学习框架中实现的内置迭代器会比上述for循环快得多


## 初始化模型参数
# 从正态分布获得w
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# 将偏置初始化为0
b = torch.zeros(1, requires_grad=True)

# 在初始化后，任务就变成了更新这些参数，直到足够拟合我们的数据
# 每次需要向减小损失的方向更新参数，使用autograd自动微分

#  定义模型，也即预测x和y的关系，指定初始化的参数再逐步更新
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数，利用平方损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):  #@save
    """
    小批量随机梯度下降算法，用于更新模型的参数。
    梯度是函数值上升最快的方向，不妨想象一个多元函数f(x,y,z)，其在某个给定位置的梯度向量就表明了一个方向
    让参数沿着梯度的反方向去变化，就可以减小函数值，而变化的幅度或者说在该方向移动的距离取决于学习率
    而损失函数多项式里，这个参数前的项数，等于该批次的数目，由于累加了，所以考虑除以批次数
    为了减小损失函数，要让被看作自变量的两个参数朝着下降最快的方向，

    参数:
    params (list): 包含需要更新的模型参数的列表，例如模型的权重和偏置。
    lr (float): 学习率，控制参数更新的步长。
    batch_size (int): 小批量数据的样本数量。

    返回:
    None，直接在原地更新参数。
    """
    # 上下文管理器，禁用梯度计算，因为在更新参数时不需要计算梯度
    with torch.no_grad():
        # 遍历需要更新的参数列表
        for param in params: # 也即w和b
            # 根据小批量随机梯度下降的公式更新参数
            # 学习率乘以参数的梯度，再除以批量大小，最后从参数中减去该值
            param -= lr * param.grad / batch_size
            # 将参数的梯度清零，以便下一次迭代计算新的梯度
            param.grad.zero_()


# 开始训练
# 概括来说，执行以下流程
# 1、初始化参数（超参数）
# 2、重复以下训练，直至完成
#   计算梯度g
#   更新参数w = w-ng

lr = 0.03 # 学习率
num_epochs = 3 # 迭代周期
net = linreg # 回归模型
loss = squared_loss # 损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 输出训练后，训练参数与真实参数的误差
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')