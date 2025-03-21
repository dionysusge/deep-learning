
import torch
from d2l import torch as d2l
from d2l.torch import Accumulator
from d2l.torch import Animator

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 初始化模型参数

# 原始数据集每个样本都是28*28的图像，在这里将展平它们，看作长度为784的向量，这里暂时只把每个像素位置看作一个特征
num_inputs = 784
# 输出的维度要与数据集里类别一样多
num_outputs = 10
# 那么权重是一个784 * 10的矩阵（每个像素都要乘以每个类别所具有的权重），偏置向量为1 * 10的行向量
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作

# 先回顾一下sum运算，可以通过dim指定运算的轴，keepdim参数用于是否保存原先的轴数
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True), sep="\n")

# 实现softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) # 按列加，得到某行的和
    return X_exp / partition  # 这里应用了广播机制，每个参数都除以所在行的partition

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X, X_prob, X_prob.sum(1), sep="\n")


# 定义softmax模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat里的两个参数，前一个选择了行，后一个选择了列，那么就选择了两个元素，1行1列和2行3列的
print(y_hat[[0, 1], y])

# 由上述操作，下面定义损失函数，交叉熵
# 那其实交叉熵就是在模型处理之后，某行已经给出了分类概率，选定某个位置的概率去取对数而已，如果说越接近于1，取完对数就越接近0
# 如果越接近于0，那么交叉熵会变得很大
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(y_hat, y))

# 分类精度
# 必须输出硬判断时，选择概率最高的类
# 使用argmax获得每行中最大元素的索引来获得预测类别，和真实的y进行比较
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    # 检查 y_hat 是否为二维张量且列数大于 1，即是否为多分类的预测结果
    # len(y_hat.shape)输出维度数，后者输出列数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 如果是多分类预测结果，使用 argmax 函数获取每行最大值的索引，即预测的类别
        y_hat = y_hat.argmax(axis=1)
    # 将预测结果的类型转换为与真实标签相同的类型，并与真实标签进行比较，得到一个布尔型张量
    cmp = y_hat.type(y.dtype) == y
    # 将布尔型张量转换为与真实标签相同的类型，并求和，得到预测正确的样本数量
    return float(cmp.type(y.dtype).sum())

# 使用自定义的数据来检验一下函数
print(accuracy(y_hat, y) / len(y))


# 对于任意迭代器data_iter可访问的数据集，可以评估在任意模型net的精度
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    # 创建一个 Accumulator 对象，用于累加两个值，分别是正确预测的样本数和总的样本数
    metric = Accumulator(2)  # 正确预测数、预测总数，这里定义了一个Accumulator类
    # 使用 torch.no_grad() 上下文管理器，在该上下文内不进行梯度计算，以减少内存消耗和计算量
    with torch.no_grad():
        # 遍历数据迭代器中的每个批次，X 是输入数据，y 是对应的真实标签
        for X, y in data_iter:
            # 调用 accuracy 函数计算当前批次的正确预测数，并将其添加到 metric 中
            # 同时将当前批次的样本总数（通过 y.numel() 获取）也添加到 metric 中
            metric.add(accuracy(net(X), y), y.numel())
    # 计算并返回模型在整个数据集上的精度，即正确预测数除以预测总数
    print("-" * 59, metric[0], metric[1], sep="\n")
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """
    训练模型一个迭代周期
    参数:
    net (torch.nn.Module): 要训练的模型
    train_iter (iterator): 训练数据集的迭代器
    loss (function): 损失函数
    updater (torch.optim.Optimizer or function): 优化器，可以是 PyTorch 内置的优化器，也可以是自定义的优化器

    返回:
    tuple: 包含训练损失和训练精度的元组
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置的优化器和损失函数
            # 清空优化器中的梯度信息
            updater.zero_grad()
            # 计算损失的平均值并进行反向传播，计算梯度
            l.mean().backward()
            # 根据计算得到的梯度更新模型参数
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            # 计算损失的总和并进行反向传播，计算梯度
            l.sum().backward()
            # 调用自定义优化器更新参数，传入当前批次的样本数量
            updater(X.shape[0])
        # 累加当前批次的损失总和、正确预测的样本数和样本总数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,)) #train_metrics + (test_acc,)：将 train_metrics 元组和 (test_acc,) 元组拼接在一起

    # 解包训练损失和训练准确率
    train_loss, train_acc = train_metrics
    # 断言训练损失小于 0.5，如果不满足条件则抛出异常并显示当前训练损失
    assert train_loss < 0.5, train_loss
    # 断言训练准确率在 0.7 到 1 之间，如果不满足条件则抛出异常并显示当前训练准确率
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    # 断言测试准确率在 0.7 到 1 之间，如果不满足条件则抛出异常并显示当前测试准确率
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=8):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1)) # argmax返回了最大值的索引，用getxx函数获取到类别
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()


lr = 0.1
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
predict_ch3(net, test_iter)