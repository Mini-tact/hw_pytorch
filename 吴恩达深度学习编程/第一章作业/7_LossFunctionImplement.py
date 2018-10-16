"""
实现基于numpy的损失函数向量化版本，你会发现abs是机器有用的
损失唱歌用来计算模型的表现，损失越大，预期的输出和真实值的差距就会越大。
在深度学习中，你使用优化算法像梯度下降算法训练那你的模型和去减少你的花费
L1损失的定义时所有差的绝对值的和
"""
import numpy as np

def f1(y_hat,y):
    return sum(abs(y-y_hat))

#如果y_hat的维数为m,y的维数为m 则得到的输出也是为m

y_hat = np.array([1,2,3,4])
y = np.array([5,6,7,8])
print(f1(y_hat,y))  # 16

#实现L2损失的向量化运算。这里I有几个方法可以显示，但是你或许会找到np.dot是好用的
#L2损失函数定义为所有向量的平方差的和

def l2(y_hat,y):
    return np.dot(abs(y_hat-y),abs(y-y_hat))

print('L2的结果是%d'%l2(y_hat,y))
