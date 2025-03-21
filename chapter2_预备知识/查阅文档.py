## 查找模块中所有函数和类
# 调用dir函数可以知道模块中可调用的函数和类
import torch
dir(torch.distributions)

#通常可以忽略以"__"（双下划线)开始和结束的函数，它们是python的特殊对象
# "_"(单下划线)开始的函数，它们通常是内部函数。
# 根据剩余的函数名或属性名，我们可能会猜测这个模块提供了各种生成随机数的方法
# 包括从均匀分布(uniform)正态分布(normal)和多项分布(multinomial)中采样

# 查找特定函数和类的用法
help(torch.ones_like)