# KNN


K近邻算法学习


---------------------------------------------------------
#numpy.tile函数说明

格式：
>* tile（A,reps）

>* A：array_like

>* 输入的array

>* reps：array_like

>* A沿各个维度重复的次数



```python
>>> import numpy as np

>>> A=[1,2]

>>> np.tile(A,2)

[1,2,1,2]

>>> np.tile(A,(2,3))

[[1,2,1,2，1,2], [1,2,1,2,1,2]]

>>> np.tile(A,(2,2,3))

[[[1,2,1,2,1,2], [1,2,1,2,1,2]],

[[1,2,1,2,1,2], [1,2,1,2,1,2]]]
```



###reps的数字从后往前分别对应A的第N个维度的重复次数。

如：
>* np.tile（A,2）表示A的第一个维度重复2遍。

>* np.tile（A,(2,3)）表示A的第一个维度重复3遍，然后第二个维度重复2遍。

>* np.tile（A,(2,2,3)）表示A的第一个维度重复3遍，第二个维度重复2遍，第三个维度重复2遍。




----------------------------------------------------------------------------------------

#numpy.sum函数说明

###没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加

```python
    >>> import numpy as np

    >>> a=np.sum([[0,1,2],[2,1,3]])

    >>> a

    9

    >>> a.shape

    ()

    >>> a=np.sum([[0,1,2],[2,1,3]],axis=0)

    >>> a

    array([2, 2, 5])

    >>> a.shape

    (3,)

    >>> a=np.sum([[0,1,2],[2,1,3]],axis=1)

    >>> a

    array([3, 6])

    >>> a.shape

    (2,)

```


------------------------------------------------------------------------------------

#numpy.argsort函数说明

###argsort函数返回的是数组值从小到大的索引值




例1：
```python
    >>> import numpy as np

    >>> x = np.array([3, 1, 2])

    >>> np.argsort(x)  #按升序排列

    array([1, 2, 0])

    >>> x[np.argsort(x)] #通过索引值排序后的数组

    array([1, 2, 3])

    >>> np.argsort(-x) #按降序排列

    array([0, 2, 1])

    >>> x[np.argsort(-x)]

    array([3, 2, 1])

    >>> x = np.array([[0, 3, 5], [2, 2, 3]])

    >>> np.argsort(x, axis=0) #按列排序

    array([[0, 1, 1],

       [1, 0, 0]])

    >>> x[np.argsort(x, axis=0)]

    array([[[0, 3, 5],

        [2, 2, 3],

        [2, 2, 3]],

       [[2, 2, 3],

        [0, 3, 5],

        [0, 3, 5]]])

    >>> np.argsort(x, axis=1) #按行排序

    array([[0, 1, 2],

       [0, 1, 2]])

    >>> x = np.array([1,4,2,5])  #数组倒序

    >>> x[::-1]

    array([5, 2, 4, 1])
```



--------------------------------------------------------------------------------

#sort函数与sorted函数说明

>* sort对列表list进行排序，而sorted可以对list或者iterator进行排序

>* sort函数对列表list进行排序时会影响列表list本身，而sorted不会
```python
    >>> a = [1,2,1,4,3,5]

	>>> a.sort()

	>>> a

	[1, 1, 2, 3, 4, 5]

	>>> sorted(a)

	[1, 1, 2, 3, 4, 5]

	>>> a

	[1, 2, 1, 4, 3, 5]
```
#sorted(iterable，cmp，key，reverse）

###其中参数：
>* iterable可以是list或者iterator；
>* cmp是带两个参数的比较函数；
>* key 是带一个参数的函数；
>* reverse为False或者True；

例1：

###cmp(x,y) 函数用于比较2个对象，如果 x < y 返回 -1, 如果 x == y 返回 0, 如果 x > y 返回 1
```python
    >>> list1 = [('david', 90), ('mary',90), ('sara',80),('lily',95)]

	>>> sorted(list1,cmp = lambda x,y: cmp(x[0],y[0]))

	[('david', 90), ('lily', 95), ('mary', 90), ('sara', 80)]

	>>> sorted(list1,cmp = lambda x,y: cmp(x[1],y[1]))

	[('sara', 80), ('david', 90), ('mary', 90), ('lily', 95)]

	>>> sorted(list1,key = lambda list1: list1[0])

	[('david', 90), ('lily', 95), ('mary', 90), ('sara', 80)]

	>>> sorted(list1,key = lambda list1: list1[1])

	[('sara', 80), ('david', 90), ('mary', 90), ('lily', 95)]

	>>> sorted(list1,reverse = True)

	[('sara', 80), ('mary', 90), ('lily', 95), ('david', 90)]

	>>> sorted(list1,reverse = False)

	[('david', 90), ('lily', 95), ('mary', 90), ('sara', 80)]
```
###Python 3.X 的版本中已经没有 cmp 函数，如果你需要实现比较功能，需要引入 operator 模块，适合任何对象
>* 介绍operator.itemgetter函数：operator.itemgetter函数获取的不是值，而是定义了一个函数
```python
	>>> import operator

	>>> a = [1,2,3]

	>>> b = operator.itemgetter(0)

	>>> b(a)

	1
```
>* 用operator.itemgetter函数排序
```python
	>>> from operator import itemgetter

    >>> sorted(list1, key=itemgetter(1))

    [('sara', 80), ('david', 90), ('mary', 90), ('lily', 95)]

    >>> sorted(list1, key=itemgetter(0))

    [('david', 90), ('lily', 95), ('mary', 90), ('sara', 80)]
```
>* 多级排序
```python
    >>> sorted(list1, key=itemgetter(0,1))

    [('david', 90), ('lily', 95), ('mary', 90), ('sara', 80)]
```
# Matplotlib.pylot的简单用法
```python
    >>> import matplotlib.pyplot as plt

    >>> plt.figure()  # 创建一幅图

    >>> plt.plot(x,y)  # 画出曲线

    >>> plt.show()  # 显示
```
>* plt.figure()  # 创建一幅图，参数可有可无
figsize=(width, height)  # width、height指定图的大小，单位inch
dpi  # dot per inch，像素密度，类似于iphone显示屏的ppi，视网膜屏的ppi要求326ppi,在960x640的3.5寸屏上。
用plt调用figure(),没有保存其返回值，函数有返回值，返回一个Figure对象

>* plt.plot(x,y)，将数据画成曲线图，x对应横坐标，y对应纵坐标，x，y都是一个一维的list。
plot除了x,y这两个参数，还有其他参数，比如：指定线的样式为虚线，点线。指定颜色和线的宽度，可以使用关键字参数指定。指定label，作为图例的文字。

>* 关于标注和标题
plt.xlabel(text),plt.ylabel(text),plt.title(text)，使用关键字参数fontsize=16制定字体大小。公式的输入支持latex格式的公式输入，即两个$中间写latex的公式，保证字符串是raw格式。在线的latex公式编辑测试，http://www.codecogs.com/latex/eqneditor.php

>* 关于坐标轴范围，通过plt.axis([xmin xmax ymin ymax]指定

>* 关于网格，plt.grid(True)显示网格

>* 关于图例，在画曲线的时候制定了label，则plt.legend()就可以了。在matplotlib里面。有专门的一个legend的类

#subplot简单介绍

>* subplot能在一张图里放多个子图，与Matlab里的subplot类似

>* pyplot是一个有状态的对象，包含了当前的图，画图区域等

>* pyplot通过调用subplot或者add_subplot来增加子图

>* p1 = plt.subplot(211) 或者 p1 = plt.subplot(2,1,1)， 表示创建一个2行，1列的图，p1为第一个子图

>* 使用p1来调用相关的函数在p1上画曲线，设置标注标题图例等

>* 与pyplot相同的是，可以直接使用pyplot画图，添加label等，也可以是通过调用P1实现。
与pyplot不同的是，有一些函数的名字不太一样，添加坐标轴的标注的函数为set_xlabel和set_ylabel

>* 添加标题set_title，只是给子图添加标题，由于pyplot是一个有状态的对象，所以pyplot.title也是给当前子图添加标题。
如果要给整个图添加标题，可以使用pyplot.suptitle(text)

#scatter()绘制散点图
>使用plot()绘图时，如果指定样式参数为仅绘制数据点，那么所绘制的就是一幅散列图。但是这种方法所绘制的点无法单独指定颜色和大小。
scatter()所绘制的散列图却可以指定每个点的颜色和大小。
matplotlib.pyplot.scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts= None)

>* scatter()的前两个参数是数组，分别指定每个点的X轴和Y轴的坐标。

>* s参数指定点的大 小，值和点的面积成正比。它可以是一个数，指定所有点的大小；也可以是数组，分别对每个点指定大小。

>* c参数指定每个点的颜色，可以是数值或数组。这里使用一维数组为每个点指定了一个数值。通过颜色映射表，每个数值都会与一个颜色相对应。默认的颜色映射表中蓝色与最小值对应，红色与最大值对应。当c参数是形状为(N,3)或(N,4)的二维数组时，则直接表示每个点的RGB颜色。

>* marker参数设置点的形状，可以是个表示形状的字符串，也可以是表示多边形的两个元素的元组，第一个元素表示多边形的边数，第二个元素表示多边形的样式，取值范围为0、1、2、3。0表示多边形，1表示星形，2表示放射形，3表示忽略边数而显示为圆形。

>* alpha参数设置点的透明度

>* lw参数设置线宽，lw是line width的缩写

>* facecolors参数为“none”时，表示散列点没有填充色

>* cmap参数默认为None，Colormap实例

>* norm参数默认为None，数据亮度设置0-1，float数据

>* vmin，vmax参数默认为None，亮度设置，如果norm实例已使用，该参数无效

# numpy.tile和numpy.repeat用法区别
tile的案例
```python
>>> from numpy import *
>>> a=array([10,20])
>>> tile(a,(3,2)) #构造3X2个copy
array([[10, 20, 10, 20],
       [10, 20, 10, 20],
       [10, 20, 10, 20]])
>>> tile(42.0,(3,2))
array([[ 42.,  42.],
       [ 42.,  42.],
       [ 42.,  42.]])
>>>
```
repeat的案例
```python
>>> from numpy import *
>>> repeat(7.,4)
array([ 7.,  7.,  7.,  7.])
>>> a=array([10,20])
>>> a.repeat([3,2])
array([10, 10, 10, 20, 20])
>>> repeat(a,[3,2])
array([10, 10, 10, 20, 20])
>>> a=array([[10,20],[30,40]])
>>> a.repeat([3,2],axis=0)
array([[10, 20],
       [10, 20],
       [10, 20],
       [30, 40],
       [30, 40]])
>>> a.repeat([3,2],axis=1)
array([[10, 10, 10, 20, 20],
       [30, 30, 30, 40, 40]])
>>>
```
#numpy中的矩阵除法
linalg.slove(mat1, mat2)












































