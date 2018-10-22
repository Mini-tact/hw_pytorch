"""
1.开发一个直观的反向传播并观察它数据上是如何工作的
2.构建更多层的隐藏层。更加复杂的网络
3.构建所有辅助函数以实现具有一个隐藏层的完整模型

你需要学会:
1.该杀2个类别的神经网络通过一个简单的隐藏层
2.使用非线性的激活函数，例如hanh
3.计算整个loss的花费
4.使用前向和反响传播
"""
import  numpy as np
import  matplotlib.pyplot as plt
import  sklearn
import sklearn.linear_model
import sklearn.datasets

from 吴恩达深度学习编程.第一章作业.planar_utils_9 import load_planar_dataset, plot_decision_boundary

'''
数据
X包含了点的横纵坐标
Y包含了点的颜色
其中X的形状为(2,400)
    Y的形状为(1,400)
'''

X, Y= load_planar_dataset()


'''
简单的logistic回归
'''
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
#
#
# predict = clf.predict(X.T)
# print(float((np.dot(Y, predict) +np.dot(1-Y, 1-predict))))
# print('Accuracy of logistic regression: %d ' %float((np.dot(Y, predict) +np.dot(1-Y, 1-predict))/float(Y.size)*100)+'%')
#
# #数据的可视化
# plt.title("Logistic Regression")
# plt.figure()
# plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.show()

"""
神经网络模型
油简单的logistic可知模型的准确率是不高的，因此我们放弃这中方案，选用一层神经网络

1.定义一个神经网络结构（带输入单元和输出单元的）
2.初始化模型的参数
3. LOOP:
        执行前向传播
        计算损失
        执行反向传播得到梯度
        更新参数
"""

'''
1.定义神经网络结构
定义三个变量
n_x 输入层的大小
n_h 隐藏层的大小
N_y 输出层的大小
'''
def network(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

'''
初始化模型的参数
1.确定你参数大小是不是正确
2.使用随机变量初始化值
3.初始化bias向量为零向量
'''
def init_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1= np.random.randn(n_h, n_x)*0.01
    b1= np.zeros((n_h),1)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y), 1)

    init_parameters= {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return init_parameters

def sigmid(X):
    return 1/(1+np.exp(-X))

"""
THE LOOP
执行前向传播
结构：
   1.以分类的数学表达式构建
   2.使用函数sigmoid（），自己在程序中构建
   3.可以使用函数‘np.tanh()’ ，他是np中的一部分
   4.执行的步骤为：
        1.从字典表init_parameters取回初始化的参数
        2.执行前向传播，计算Z1,A1,Z2,A2
   5.在反响传播中用到的之需要存储在缓存中，需要在反向传播的函数中使用到
"""
def forward_propagation(X, init_param):
    W1 = init_param['W1']
    b1 = init_param['b1']
    W2 = init_param['W2']
    b2 = init_param['b2']

    Z1= np.dot(W1, X)+b1
    A1= sigmid(Z1)

    Z2= np.dot(W2, X)+b2
    A2= sigmid(Z1)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return cache

#计算损失函数
def computer_cost(result, Y, parameters):

    logprobs = np.multiply(np.log(result),Y)+np.multiply(np.log(1-result),1-Y)
    cost= -np.sum(logprobs)/Y.shape[1]
    cost= np.squeeze(cost)
    return cost

"""
计算反向传播
反向传播是深度学习中最困难的部分，有很强的数学性

"""
def back_propagation(param, cache, X, Y):
    W1 = param['W1']
    b1 = param['b1']
    W2 = param['W2']
    b2 = param['b2']

    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']

    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T, dZ2)*A1(1-A1)
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
        'dW1':dW1,
        'db1':db1,
        'dW2':dW2,
        'db1':db2,
    }
    return grads

"""
更新参数
使用梯度下降，通过dW1,dW2,db1,bd2来更新w1,w2,b1,b2
"""
def updata_param(param, grads, learning_rate = 1.2):
    W1 = param['W1']
    b1 = param['b1']
    W2 = param['W2']
    b2 = param['b2']

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }
    return parameters


"""
训练模型
"""
def model(X, Y, n_h, num_iterations=10000):
    np.random.seed(3)
    n_x = network(X, Y)[0]
    n_y = network(X, Y)[2]

    param = init_parameters(n_x, n_h, n_y)
    W1 = param["W1"]
    b1 = param["b1"]
    W2 = param["W2"]
    b2 = param["b2"]

    for i in range(0,num_iterations):
        cache = forward_propagation(X, param)

        cost = computer_cost(cache['A2'], Y, param)

        grads = back_propagation(param, cache, X, Y)

        param = updata_param(param,grads)

        if i%100 ==0:
            print("cost after iteration %i:%f"%(i, cost))

    return param



param = model(X, Y, 4)


























