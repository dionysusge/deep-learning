import torch

x = torch.arange(1, 36, 3,  dtype=torch.float32)
print(x)
# 形状，比如几行几列
print(x.shape)

# 元素的总个数
print(x.numel())

# 元素不变，形状改变
X = x.reshape(3, 4)
print(X)
print(X.shape)
print(X.numel())

# 传入尺寸，元素全为0
y = torch.zeros(2, 3, 4)
print(y)

# 传入尺寸，元素全为1
z = torch.ones(2, 3, 4)
print(z)

# 生成均值为0，方差为1的正态分布
print(torch.randn(3, 4))

# 利用torch.tensor，直接传入一个列表或者元组创建张量
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# 实现二元运算
x1 = torch.tensor([1.0, 2, 4, 8])
y1 = torch.tensor([2, 2, 2, 2])
print(x1 + y1)
print(x1 - y1)
print(x1 * y1)
print(x1 / y1)
print(x1 ** y1)

# 指数运算，还有其它的根号、三角等
z = torch.exp(X)
print(z)

# 合并两个张量，从0开始递减，0是最高维度，在二维里，0是行，1是列
# 下面的x1是按列合并，得到3行8列
# 下面的x2是按行合并，得到6行4列
# 合并时，除了按照的维度外，其它维度的尺寸必须相同
x1 = torch.cat((X, z), dim = -1)
x2 = torch.cat((X, z), dim = 0)
print(x1, x2, sep="\n")

# 判断两个同尺寸的张量各个元素相等情况
print(X == z)
# 判断两个张量是否完全相同
print(torch.equal(X, z))
# 对张量中所有元素求和
print(z.sum())


# 不同尺寸的张量运算，广播机制
# 在不同的维度上，如果尺寸不同，向大的尺寸看齐，但是一个张量无法在不同维度上同时变大，只能有一个维度变大
# 比如X是3行4列，y是1行3列，那下面这样会报错，因为无法在两个维度上同时变大
# a = torch.arange(12).reshape((3, 4))
# b = torch.arange(3).reshape((1, 3))

# 如果只有一个维度变大，那这个维度的元素会复制，直到和另一个张量的尺寸相同，然后进行相加
a = torch.arange(12).reshape((3, 4))
b = torch.arange(4).reshape((1, 4))
print(a, b, sep="\n")
print(a + b)

# 3*1和1*2的张量也可以通过广播机制相加
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b, sep="\n")
print(a + b)

# 和python数组一样，张量的元素也可以通过索引访问，第一个索引为0，最后一个索引为-1
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(x[-1]) # 最后一行的元素
print(x[1, 2]) # 第2行第3列的元素，或者x[1][2]
print(x[1:3]) # 第2行到第3行的元素

# 为张量赋值
x[0, 0] = 9
print(x)

# 为张量赋值，使用切片，一、二行，1-3列
x[0:2, 0:3] = 12
print(x)

# 对张量进行运算后，其在内存中的id会改变，指向新分配的内存空间
before = id(y)
y = y + x
print(id(y) == before)
z = torch.zeros_like(y)

# 在机器学习中，我们通常希望原地操作，因为参数量大
# 并且其它引用仍然会指向旧的内存空间，可能会无意中导致错误
# 下面是利用切片实现原地操作，先创建了和y同尺寸的0张量，然后把运算结果给这个张量
z = torch.zeros_like(y)
print(id(z))
z[:] = x + y
print(id(z))

# 或者利用 += 也可以实现原地操作，如果上述的x、y不再用到，直接x[:] = x + y，然后释放x、y的内存
before = id(x)
x += X
print(id(x) == before)


# torch对象和numpy数组可以简单地实现转换，共享底层内存，对torch对象使用numpy函数即可
# 但是，对numpy数组使用torch函数，会重新分配内存，且返回的torch对象和原来的数组互不影响，就如下面的B
A = x.numpy()
B = torch.tensor(A)
print(type(A), type(B))
print(A, x, sep="\n")
A += 1
print(A, x, sep="\n")
x += 1
print(A, x, sep="\n")

# 将大小为1的张量转换为python标量
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))

# 转为list
print(x.tolist())

# 将list展平，利用numpy
print(x.numpy().flatten().tolist())

# 对张量进行维度变换，-1表示自动计算
print(x.reshape(-1))
print(x.reshape(1, -1))
print(x.reshape(-1, 1))

# 对张量进行比较，如果维度不同会利用广播机制，返回一个bool张量
print(x > 9)
print(x > z)