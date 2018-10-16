#规范化行后能够快速的实现梯度下降的收敛
#方法是通过函数将x变化成x/\\x\\
#||x||即每行的平方差
import numpy as np
arrya=np.array([[1,3,4],[4,5,6]])
print(arrya)  #[[1 3 4][4 5 6]]
def normRow(array):
    x= np.linalg.norm(array,axis=1,keepdims=True)  #asix表示按行还是lie来计算平方差。  keepdims的意思是保持之前的维度
    print(x) #[[5.09901951][8.77496439]]
    print('------------')
    return array/x  #广播，当维数不同时，在计算加减乘除使会将扩展小的维数在做计算
print(normRow(arrya))  #[[0.19611614 0.58834841 0.78446454][0.45584231 0.56980288 0.68376346]]
