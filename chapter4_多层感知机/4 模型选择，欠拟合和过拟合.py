# 如果有足够多的神经元、层数和训练迭代周期， 模型最终可以在训练集上达到完美的精度，此时测试集的准确性却下降了。这种现象称为过拟合。
# 过拟合是指模型在训练集上表现良好，但在测试集上表现不佳的现象。它通常发生在模型过于复杂时，导致模型学习到了训练数据中的噪声和细节，而不是数据的真实模式。

# 过拟合的原因：
# 1. 模型复杂度过高：模型的参数数量远大于训练数据的数量，导致模型能够拟合训练数据中的噪声。
# 2. 训练数据不足：训练数据量不足以覆盖数据的真实分布，导致模型无法学习到有效的模式。
# 3. 训练时间过长：模型在训练数据上训练的时间过长，导致模型学习到了训练数据中的噪声。

# 泛化误差
# 泛化误差是指模型在未见过的数据上的预测误差。它是评估模型性能的重要指标。泛化误差可以分为三部分：
# 1. 偏差：模型对训练数据的拟合程度。偏差越大，模型在训练数据上的预测误差越大。
# 2. 方差：模型对训练数据的敏感程度。方差越大，模型在训练数据上的预测误差越大。
# 3. 噪声：数据中的随机误差。噪声越大，模型在训练数据上的预测误差越大。

# 解决过拟合的方法：
# 1. 减少模型复杂度：减少模型的参数数量，降低模型的复杂度。
# 2. 增加训练数据：增加训练数据的数量，覆盖数据的真实分布。
# 3. 正则化：在损失函数中加入正则化项，限制模型的复杂度。
# 4. 提前停止：在训练过程中监控模型在验证集上的性能，当验证集性能不再提升时停止训练。

# 下面引入一个拟合多项式的例子来说明过拟合的现象。我们将使用一个简单的多项式函数来生成数据
# 生成多项式
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]

# 对模型进行训练、测试
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# 使用三阶多项式：
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])

# 使用线性函数
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])

# 使用高阶多项式
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)

# 上面三种，高阶多项式对应过强的表达能力，即使高阶的系数接近0（多次训练后），但实际仍存在误差
# 线性模型则表达能力不足，其实再怎么训练，不仅训练集的准确率差，验证集也无法提高

# 下面的内容将会围绕解决过拟合展开

