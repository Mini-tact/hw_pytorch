"""
调整隐藏层的大小
运行以下的程序，你会观察到不同模型的行为对于含有不同隐藏神经元的个数
"""
import  numpy as np
import  matplotlib.pyplot as plt
from 吴恩达深度学习编程.第一章作业.planar_utils_9 import load_planar_dataset, plot_decision_boundary

X, Y= load_planar_dataset()

def network(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def init_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    init_parameters= {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return init_parameters

def sigmid(X):
    return 1/(1+np.exp(-X))

def forward_propagation(X, init_param):
    W1 = init_param['W1']
    b1 = init_param['b1']
    W2 = init_param['W2']
    b2 = init_param['b2']

    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1)+b2
    A2 = sigmid(Z2)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return cache

def computer_cost(result, Y, parameters):

    logprobs = np.multiply(np.log(result),Y)+np.multiply(np.log(1-result),1-Y)
    cost= -np.sum(logprobs)/Y.shape[1]
    cost= np.squeeze(cost)
    return cost

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
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
        'dW1':dW1,
        'db1':db1,
        'dW2':dW2,
        'db2':db2,
    }
    return grads

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

def predict(param,X):

    cache = forward_propagation(X, param)
    prediction = np.round(cache['A2'])

    return prediction



hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

for i, n_h in enumerate(hidden_layer_sizes):
    plt.figure()
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = model(X, Y, n_h, 10000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()



















