# KNN
K近邻算法学习

------------------------------------------------------------------------------
#numpy.tile函数说明
格式：tile（A,reps）
* A：array_like
* 输入的array
* reps：array_like
* A沿各个维度重复的次数

import numpy as np
举例：A=[1,2]
1. np.tile(A,2)
结果：[1,2,1,2]
2. np.tile(A,(2,3))
结果：[[1,2,1,2，1,2], [1,2,1,2,1,2]]
3. np.tile(A,(2,2,3))
结果：[[[1,2,1,2,1,2], [1,2,1,2,1,2]],
[[1,2,1,2,1,2], [1,2,1,2,1,2]]]

reps的数字从后往前分别对应A的第N个维度的重复次数。
如：np.tile（A,2）表示A的第一个维度重复2遍。
np.tile（A,(2,3)）表示A的第一个维度重复3遍，然后第二个维度重复2遍。
np.tile（A,(2,2,3)）表示A的第一个维度重复3遍，第二个维度重复2遍，第三个维度重复2遍。

----------------------------------------------------------------------------------------
#numpy.sum函数说明
#没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加

[python] view plain copy

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

------------------------------------------------------------------------------------
#numpy.argsort函数说明
#argsort函数返回的是数组值从小到大的索引值

例1：
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

--------------------------------------------------------------------------------
#sort函数与sorted函数说明
#sort对列表list进行排序，而sorted可以对list或者iterator进行排序
#sort函数对列表list进行排序时会影响列表list本身，而sorted不会
    >>> a = [1,2,1,4,3,5]
	>>> a.sort()
	>>> a
	[1, 1, 2, 3, 4, 5]
	>>> sorted(a)
	[1, 1, 2, 3, 4, 5]
	>>> a
	[1, 2, 1, 4, 3, 5]
#sorted(iterable，cmp，key，reverse）
其中参数：iterable可以是list或者iterator；
         cmp是带两个参数的比较函数；
		 key 是带一个参数的函数；
		 reverse为False或者True；
例1：
    #cmp(x,y) 函数用于比较2个对象，如果 x < y 返回 -1, 如果 x == y 返回 0, 如果 x > y 返回 1
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
	#Python 3.X 的版本中已经没有 cmp 函数，如果你需要实现比较功能，需要引入 operator 模块，适合任何对象
















