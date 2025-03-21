
# 创建一个人工数据集，并存储在csv（逗号分隔值）文件
import os
import pandas as pd

# os.path打印的是os库所在的路径
print(os.path)
# 使用os.path.join()和.join的区别：前者会考虑不同操作系统的路径分隔符，自动进行调整
print(os.path.join('..', 'data'))

# makedirs在给定位置下创建一个data文件夹，exist_ok这个参数表示如果文件夹已经存在，就不创建了
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data下的文件路径
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# 创建并打开一个文件，w模式写入的是字符串
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,Anshun,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# 从创建的csv文件中加载原始数据集
data = pd.read_csv(data_file)
print(data)

# 处理缺失值，一般有插值和删除两种方法
# 可以通过iloc选择行和列，:表示选择所有行和列
# 下面input为前两列，output为最后一列
# 其实这样提供了一点优化的方案，之前用列名读好麻烦哈哈哈，不过相对来说那种更明确一些
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 对于input里缺少的数据，用同一列的均值去替代
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 对于类别或者离散值，会转化列名，在原列名的基础上加可能的离散名，比如nan、Anshun、Pave
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 现在类别转化为了0、1表示的数字，可以转化为张量

import torch

# dataframe类型的数据要先转为numpy类型，再用torch里的tensor函数去转化为张量
x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x, y, sep="\n")

# 使用删除法去处理缺失值，噢那这个dropna函数，只有行内有na值就会删除
inputs = data.iloc[:, 0:2]
inputs_deleted = inputs.dropna()
print("删除缺失值后的输入数据:")
print(inputs_deleted)

# 上述情况直接用无参的dropna删除得较多，以下是指定某行为空才删除，以及整行都缺失才删除
# 这里假设只在 'NumRooms' 列有缺失值时才删除该行
inputs_deleted_subset = inputs.dropna(subset=['NumRooms'])
print("指定 NumRooms 列缺失才删除后的输入数据:")
print(inputs_deleted_subset)

# 整行都缺失才删除
inputs_deleted_all = inputs.dropna(how='all')
print("整行都缺失才删除后的输入数据:")
print(inputs_deleted_all)

