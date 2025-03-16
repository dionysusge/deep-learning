# linear regression

# 噪声项，考虑观测误差的存在

# 损失函数，损失均值
# 在训练模型时，希望寻找一组参数w、b，使得损失均值降到最低

# 线性回归问题存在解析解
import numpy as np
from d2l.torch import Timer

# 生成一些示例数据
np.random.seed(0)
n_samples = 100
n_features = 2 # 数据的列数
X = np.random.randn(n_samples, n_features)
print(X)
# 添加偏置项，增加一列值
X = np.hstack((X, np.ones((n_samples, 1))))
print(X)
# 添加真实权重
true_w = np.array([1, 2, 3])
# 利用广播机制，复制true_w与数据相乘
# dot函数，两个参数都是一维的时候结果是标量，一个为二维、另一个为一维时，得到的是一个一维数组，两个都为二维数组时，执行矩阵乘法
y = np.dot(X, true_w) + np.random.randn(n_samples) * 0.1  # 添加了一定的噪声，得到最终的目标值y
print(y)
# 计算解析解
# "@" 是计算点积，两个二维数组，得到的是一个矩阵，这里是方阵
# np.linalg是linear algebra模块，有线性代数相关的一系列运算，比如行列式、范数(也可以用torch里的norm去求)
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y

print("真实权重:", true_w)
print("解析解得到的权重:", w_hat)

# 随机梯度下降
# 在无法得到解析解的情况下，我们可以使用随机梯度下降来训练模型，几乎可以优化所有的深度学习模型
# 随机梯度下降的基本思想是，每次随机选择一小批样本，计算其梯度，并更新模型参数
# 梯度下降最简单的用法是计算损失函数关于模型参数的导数（梯度），然后沿着梯度的反方向更新模型参数

# B表还是批量大小 batch size
# η表示学习率 learning rate
# 这些是手动预先指定的，不在训练过程中更新，称为超参数
# 调参是选择超参数的过程，根据训练迭代结果来调整

# 梯度下降的方法能够使得损失向最小值慢慢收敛，但却不能在有限的步数里非常精确地达到最小值

# 在深度学习过程中，每个损失平面上都存在一个最小化损失的参数，但是如何让训练集整体的损失最小，这组参数的寻找是个难题
# 而更难的地方在于，要在测试集或者从未见过的数据上实现较小的损失，这一过程称为泛化


# ----------------------------------------------------------------
# ！！矢量化加速
# 在训练模型时，希望同时处理小批量的样本，需要对计算进行矢量化，而不是编写开销高昂的for循环
# 下面是一个示例
import math
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones([n])
b = torch.ones([n])

# 现在对工作负载进行测试
# 使用for循环：
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')
print(c)

# 使用重载的 + 运算符，趋近于0秒了
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')
print(d)

# 正态分布和平方损失
# 先定义一个正态分布的计算函数
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
# 可视化正态分布：
x = np.arange(-7, 7, 0.01)

# 给定mu、sigma
params = [[0, 1], [0, 2], [3, 1]]
# 下面的列表推导式得到一个嵌套列表，每个元素是一个array数组，numpy库里的ndarray对象
print([normal(x, mu, sigma) for mu, sigma in params])
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()

# 一般会假设观测中包含了噪声，而噪声服从正态分布
# 推导可知，最小化均方误差等价于对线性模型的最大似然估计


# 线性回归到深度网络
# 可以将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络
# 对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连
# 我们将这种变换称为全连接层（fully-connected layer）或称为稠密层（dense layer）