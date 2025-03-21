import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 读取数据集
# 获得一个ToTensor对象，主要作用是将 PIL 图像或者 NumPy 数组转换为 PyTorch 的张量（torch.Tensor），
# 同时会将图像像素值从 [0, 255] 范围归一化到 [0.0, 1.0] 范围。
trans = transforms.ToTensor()
""" 
其它的一些预处理操作

transforms.Normalize(mean, std)：对张量进行归一化处理，将图像数据的每个通道按照指定的均值（mean）和标准差（std）进行标准化。
transforms.RandomCrop(size)：对图像进行随机裁剪，裁剪出指定大小的图像区域。
transforms.RandomHorizontalFlip()：以一定的概率对图像进行水平翻转，增加数据的多样性。
"""

# torchvision框架里的dataset有很多现有的数据集
# 比较经典的MNIST数据集太早了，1998年，作为基准过于简单，这里使用2017年的新数据集，下载并读取到内存中

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)
"""
Fashion-MNIST由10个类别的图像组成，每个类别由训练数据集(train dataset)中的6000张图像 和测试数据集中的1000张图像组成。
因此，训练集和测试集分别包含60000和10000张图像。测试数据集不会用于训练，只用于评估型性能。
"""



print(mnist_train) # 一个FashionMNIST对象，打印其相关信息比如数据量、位置
print(type(mnist_test[0])) # 元组
print(type(mnist_test[0][0])) # 一个torch.Tensor向量，
print(mnist_test[0][0].shape) # 每个输入图像的高度、宽度均为28像素，通道数为1

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签
    'T恤', '裤子', '套头衫', '连衣裙', '外套',
    '凉鞋', '衬衫', '运动鞋', '包', '短靴
    '"""
    # 这里按照数据集里的标签顺序构建一个对应字符串
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # 每个i是一个张量，对应类别的索引，将其强制转换为int然后给到上述列表，将数据集中的索引转换为文字标签
    return [text_labels[int(i)] for i in labels]

# 可视化样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """
    绘制图像列表。

    参数:
    imgs (list): 包含图像数据的列表，可以是 PyTorch 张量或 PIL 图像。
    num_rows (int): 子图的行数。
    num_cols (int): 子图的列数。
    titles (list, 可选): 每个子图的标题列表，默认为 None。
    scale (float, 可选): 控制图像显示的缩放比例，默认为 1.5。

    返回:
    axes (numpy.ndarray): 包含所有子图的坐标轴对象的数组。
    """
    # 根据子图的行数、列数和缩放比例计算图像显示的尺寸
    figsize = (num_cols * scale, num_rows * scale)
    # 创建一个包含指定行数和列数子图的图形对象和坐标轴对象
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    # 将二维的坐标轴对象数组展平为一维数组，方便后续遍历
    axes = axes.flatten()
    # 遍历坐标轴对象和图像数据
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        """zip(axes, imgs)：将 axes 和 imgs 这两个可迭代对象中的元素一一配对，形成一个新的可迭代对象。
        axes 通常是一个包含多个子图坐标轴对象的数组，imgs 是一个包含多个图像数据的列表。
        enumerate()：这是 Python 的内置函数，用于将一个可迭代对象转换为一个枚举对象，同时返回元素的索引和元素本身。
        在这个循环中，enumerate(zip(axes, imgs)) 会为每一对 (ax, img) 分配一个索引 i
        """
        # 判断图像数据是否为 PyTorch 张量
        if torch.is_tensor(img):
            # 如果是张量，将其转换为 NumPy 数组并显示图像
            ax.imshow(img.numpy())
        else:
            # 如果是 PIL 图像，直接显示图像
            ax.imshow(img)
        # 隐藏 x 轴
        ax.axes.get_xaxis().set_visible(False)
        # 隐藏 y 轴
        ax.axes.get_yaxis().set_visible(False)
        # 如果提供了标题列表
        if titles:
            # 为当前子图设置标题
            ax.set_title(titles[i])
    # 返回包含所有子图的坐标轴对象的数组
    return axes

# 直接用torchutils工具包里data模块的DataLoader对象获取小批量18份数据
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18))) # 返回两个参数，第一个是图像数据，第二个是图像标签
print(X.shape) # 这里的形状是18份1通道，28*28的图片，要转换一下
show_images(X.reshape(18, 28, 28), 3, 6, titles=get_fashion_mnist_labels(y))
# d2l.plt.show()

# 现在观测一下读取完数据所用的时间

# 每批读取256份数据
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    """
    进程数过少
    CPU 利用率不足：当 num_workers 设置得比较少的时候，例如设置为 1 或者 0（意味着只使用主进程来读取数据），CPU 可能无法充分被利用。在数据读取和预处理的过程中，会存在大量的计算操作，像图像的解码、裁剪、归一化等。如果只有少数几个进程负责这些操作，那么 CPU 可能会处于空闲状态，不能充分发挥其计算能力，从而导致数据读取的速度变慢。
    数据加载和模型训练串行执行：如果数据加载的速度跟不上模型训练的速度，模型就需要等待数据加载完成才能继续训练，这样会造成训练过程中的停顿，降低整体的训练效率。
    
    进程数过多
    进程间通信开销增大：每个子进程在读取和处理数据时，需要和主进程进行通信，将处理好的数据传递给主进程。当 num_workers 设置得过多时，进程间的通信开销会显著增加。例如，大量的子进程需要频繁地向主进程发送数据，这会占用大量的系统资源，如内存和带宽，从而导致数据读取的速度变慢。
    系统资源竞争：过多的子进程会竞争系统的资源，如 CPU、内存和磁盘 I/O。当多个进程同时访问磁盘时，会导致磁盘 I/O 冲突，从而降低数据读取的速度。此外，过多的进程还可能会导致系统的内存不足，引发内存交换（swapping），进一步降低系统的性能。
    """
    return 4 # 这里的进程数要适量，多或少都会降低速度


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

# 整合所有组件，完成数据集读取函数的定义
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    # 如果提供了 resize 参数
    if resize:
        # 在转换列表的开头插入调整图像大小的操作
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    # resize参数将图像从32*32调整为了64*64
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

print(dir(torchvision.datasets))