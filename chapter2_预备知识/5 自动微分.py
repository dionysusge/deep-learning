import torch

# 为变量x创建初始值
x = torch.arange(4.0)
print(x)
# 设置为True就表明要对张量x进行梯度计算，以下划线结尾通常表示原地操作
x.requires_grad_(True)
"""
当你想要计算某个函数关于 x 的梯度时，就需要把 x 的 requires_grad 属性设置为 True。
之后，在执行前向传播计算函数值，再调用 backward() 方法进行反向传播，
PyTorch 就会自动计算出函数关于 x 的梯度，并将其存储在 x.grad 属性中。
"""
print(x.grad) # 目前为空

# 现在计算y，相当于y = 2 * x_1 ^ 2 + 2 * x_2 ^ 2....得到一个标量值，但也是一个函数，对每个变量逐次求导就得到了反向传播的结果
y = 2 * torch.dot(x,x)

# 现在通过反向传播来自动计算y关于x每个分量的梯度，也即y = 2 * x^2 对于所取x值的导数
y.backward()
print(x.grad)

# 验证是否正确
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
# 这里的y，相当于 y = x1 + x2 +....
# 对x1或者任意单变量求导，得到1
y = x.sum()
y.backward()
print(x.grad)

# 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_() # 清空梯度
y = x * x # 相同位置元素相乘

# 相乘后求和，相当于y = x1 ^ 2 + x2 ^ 2....，求导数
# 等价于y.backward(torch.ones(len(x)))
"""
当调用 y.backward(gradient) 时，gradient 这个参数表示的是后续损失函数对 y 中每个元素的梯度。
这里传入 torch.ones(len(x))，意味着后续损失函数对 y 中每个元素的梯度都是 1。
根据链式法则，x 的某个元素的梯度等于 y 中对应元素关于该 x 元素的梯度乘以 gradient 中对应的值。
因为 y 中第 i 个元素是 xi^2，它关于 xi 的梯度是 2 * xi，而 gradient 中对应的值是 1。
"""

y.sum().backward()
print(x.grad)


## 分离计算

x.grad.zero_()
y = x * x

"""
detach() 方法会创建一个新的张量，这个新张量和原张量共享数据内存，
但不会记录其在计算图中的历史信息，也就是它不会参与梯度的计算。
具体来说，新张量的 requires_grad 属性会被设置为 False，即便原张量的 requires_grad 属性为 True。
"""
u = y.detach() # u不参与梯度运算，只是复制了一份相同的副本，相当于此时u是个常数
z = u * x # z = x_i^2 * x

z.sum().backward() # 那么求和后对z求导，每个都恰好等于x_i^2，故肯定和u相等
# 反向传播只会标记require_grad = True的变量进行求导
print(x.grad == u) # u = x^2
print(x.grad)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

## 使用python控制流进行梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward() # 得到的应该是2^n * a，求导的话得到2^n
print(d)
print(a.grad)
# 所以下述等式成立
print(a.grad == d/a)




### 计算二阶导的做法，要设置create_graph = True来保留计算图
# 创建一个需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 定义一个函数 y = x^2
y = x ** 2

# 计算一阶导数
first_derivative = torch.autograd.grad(y, x, create_graph=True)[0]

# 计算二阶导数
second_derivative = torch.autograd.grad(first_derivative, x)[0]

print(f"一阶导数: {first_derivative}")
print(f"二阶导数: {second_derivative}")

import math
# 绘制f(x)=sinx和导函数的图像
x = torch.arange(-math.pi,math.pi,0.1, requires_grad=True)
print(x)
y = torch.sin(x)
print(y)
y.sum().backward()

z = x.grad
print(z)

# 这里还不能直接转换为numpy数组，因为会破坏计算图，要用其副本
import matplotlib.pyplot as plt
plt.plot(x.detach(), y.detach(), label="sin(x)")
plt.plot(x.detach(), z.detach(), label="sin'(x)")
# 添加图例
plt.legend()
# 添加坐标轴标签
plt.xlabel("x")
plt.ylabel("y")
# 显示网格线
plt.grid(True)
# 显示图像
plt.show()