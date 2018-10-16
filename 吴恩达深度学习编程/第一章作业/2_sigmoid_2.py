#输入为矩阵或者向量
# then a Python operation such ass=x+ 3ors=1xwilloutput s as a vector of the same size as x.
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
x_1 = np.array([[1,2,3],[1,2,3]])
def sigmoid_2(x):
    return 1/(1+np.exp(-x))

y=sigmoid_2(x)
print(y)  #[0.73105858 0.88079708 0.95257413 0.98201379 0.99330715]

print(sigmoid_2(x_1))
#[[0.73105858 0.88079708 0.95257413]
#[0.73105858 0.88079708 0.95257413]]
