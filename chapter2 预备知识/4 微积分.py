import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def f(x):
    return 3 * x ** 2 - 4 * x
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
# 逐步让h趋近于0，在x=1处的导数应该是2
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

# SVG 是一种基于 XML 的矢量图形格式，它可以无损缩放，并且在不同分辨率的设备上都能保持清晰的显示效果
def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    # rcParams是matplotlib中的一个全局变量，它包含了matplotlib的所有配置参数，比如线条颜色、字体大小、图表大小等
    d2l.plt.rcParams['figure.figsize'] = figsize
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    # 设置 x 轴的标签
    axes.set_xlabel(xlabel)
    # 设置 y 轴的标签
    axes.set_ylabel(ylabel)
    # 设置 x 轴的缩放类型
    axes.set_xscale(xscale)
    # 设置 y 轴的缩放类型
    axes.set_yscale(yscale)
    # 设置 x 轴的取值范围
    axes.set_xlim(xlim)
    # 设置 y 轴的取值范围
    axes.set_ylim(ylim)
    # 如果提供了图例列表，则添加图例
    if legend:
        axes.legend(legend)
    # 添加网格线
    axes.grid()
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """
    绘制数据点

    :param X: 输入数据的 x 坐标，可以是单个列表或数组，也可以是多个列表或数组的列表
    :param Y: 输入数据的 y 坐标，可选参数，默认为 None
    :param xlabel: x 轴的标签，可选参数，默认为 None
    :param ylabel: y 轴的标签，可选参数，默认为 None
    :param legend: 图例列表，可选参数，默认为 None
    :param xlim: x 轴的取值范围，可选参数，默认为 None
    :param ylim: y 轴的取值范围，可选参数，默认为 None
    :param xscale: x 轴的缩放类型，可选参数，默认为 'linear'
    :param yscale: y 轴的缩放类型，可选参数，默认为 'linear'
    :param fmts: 线条格式的元组，可选参数，默认为 ('-', 'm--', 'g-.', 'r:')'-'：表示实线。
        'm--'：表示洋红色（magenta）的虚线。
        'g-.'：表示绿色（green）的点划线。
        'r:'：表示红色（red）的点线。
    :param figsize: 图表的大小，可选参数，默认为 (3.5, 2.5)
    :param axes: matplotlib 的 Axes 对象，可选参数，默认为 None
    """
    # 如果没有提供图例列表，将其初始化为空列表
    if legend is None:
        legend = []

    # 设置图表的大小
    set_figsize(figsize)
    # 如果没有提供 Axes 对象，获取当前的 Axes 对象
    axes = axes if axes else d2l.plt.gca()

    # 辅助函数，判断 X 是否只有一个轴
    def has_one_axis(X):
        """
        判断 X 是否只有一个轴

        :param X: 输入数据
        :return: 如果 X 只有一个轴，返回 True；否则返回 False

        hasattr(X, "ndim") and X.ndim == 1：
        hasattr(X, "ndim")：检查 X 对象是否具有 ndim 属性。在 numpy 数组中，ndim 属性表示数组的维度。
        X.ndim == 1：如果 X 有 ndim 属性，进一步检查其维度是否为 1。如果满足这两个条件，说明 X 是一个一维的 numpy 数组。
        isinstance(X, list) and not hasattr(X[0], "__len__")：
        isinstance(X, list)：检查 X 是否为列表类型。
        not hasattr(X[0], "__len__")：如果 X 是列表，检查其第一个元素是否没有 __len__ 属性。如果一个对象没有 __len__ 属性，说明它不是可迭代对象，那么 X 就是一个一维列表。
        """
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    # 如果 X 只有一个轴，将其转换为包含单个元素的列表
    if has_one_axis(X):
        X = [X]
    # 如果没有提供 Y，将 X 视为 y 坐标，并将 X 初始化为空列表
    if Y is None:
        X, Y = [[]] * len(X), X
    # 如果 Y 只有一个轴，将其转换为包含单个元素的列表
    elif has_one_axis(Y):
        Y = [Y]
    # 如果 X 和 Y 的长度不相等，将 X 重复多次以匹配 Y 的长度
    if len(X) != len(Y):
        X = X * len(Y)

    # 清除 Axes 对象上的所有内容
    axes.cla()
    # 遍历 X、Y 和 fmts，绘制数据点
    for x, y, fmt in zip(X, Y, fmts):
        # 如果 x 不为空，使用 x 和 y 绘制
        if len(x):
            axes.plot(x, y, fmt)
        # 否则，只使用 y 绘制
        else:
            axes.plot(y, fmt)

    # 设置坐标轴的标签、范围、缩放类型和图例
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show()

# 偏导，某个多元函数对某个变量的导数组合起来的向量，称为偏导数向量
# 链式法则