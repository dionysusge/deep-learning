import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# 构建一个概率向量
fair_probs = torch.ones([6]) / 6
print(fair_probs)
# 构造multinomial类，total_count是抽样数，fair_probs是概率向量
# sample函数用于抽样，可以传入一个元组，代表输出的形状，做多少组抽样
x = multinomial.Multinomial(10000, fair_probs).sample((3, 4))

# 计算相对频率
print(x / 10000)

# 随着次数增加收敛到真实概率
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0) # 计算某个轴上的累加和，按行累加，每次新增的行都会加上之前的行
print(cum_counts)
print(cum_counts.sum(dim = 1, keepdims = True)) # 按列数据合并到第一列，keepdims=True保留维度
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True) # 每次实验的相对频率，逐次累加

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))

d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()