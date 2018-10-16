#两个常用到的函数np.shape和np.reshape
#np.shape  用于得到数组的维数
#np.reshaoe 将数组的维数重置

import numpy as np

array=np.arange(0,27,1)  # 27是不包括的  #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
print(array)
array=array.reshape(3,3,3)
"""
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]
"""
print(array)

#print(array.shape[1])  #返回值为3，即维数
def reshape_array(array):
    return  array.reshape(array.shape[0]*array.shape[1],array.shape[2])
print('---------------------------------')
print(reshape_array(array))
"""
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [12 13 14]
 [15 16 17]
 [18 19 20]
 [21 22 23]
 [24 25 26]
"""