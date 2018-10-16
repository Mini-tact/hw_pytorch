"""
在深度学习里，你需要解决非常大的数据集。于是，一个非计算最优函数会成为一个巨大的瓶颈在你的算法和能计算的模型中会话费非常多的时间
去计算。为了确认你的代码是可能够有效计算，你可以使用向量化
"""
import time
import numpy as np
x1=np.random.rand(100)
x2=np.random.rand(100)

tic = time.process_time()
dot=0
for i in range(len(x1)):
    for j in range(len(x2)):
        dot = x1[i]*x2[j]
toc = time.process_time()
print("the time is %f"%((toc-tic)*1000))

print('--------------------------------------')


tic = time.process_time()
array = np.zeros((len(x1),len(x2)))
np.dot(x1,x2)
toc = time.process_time()
print("the  dot array runtime is %f"%((toc-tic)*1000))


print('--------------------------------------')


tic = time.process_time()
array = np.zeros((len(x1),len(x2)))
np.multiply(x1,x2)
toc = time.process_time()
print("the matul array runtime is %f"%((toc-tic)*1000))