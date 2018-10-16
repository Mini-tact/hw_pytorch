#sigmoid求导  g(x)=a=1(1+x^-x)
#sigmoid的导数为 a(1-a)
import numpy as np
def sigmoid_grad(x):
    gx = 1/(1+np.exp(-x))
    dg = gx*(1-gx)  #dgwei sigmoid的导数  将导数存在一个数组中，可以可以用于方向传到计算
    return dg,gx

print(sigmoid_grad(1)) #(0.19661193324148185, 0.7310585786300049)