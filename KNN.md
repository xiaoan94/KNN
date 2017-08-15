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


















































