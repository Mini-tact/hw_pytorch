"""
1.使用np.exp()和plt来描绘数据
"""
import numpy as np
import matplotlib.pyplot as plt
##sigmoid=1/(1+e^-x)
def sigmoid(input):
    input=-input
    return 1/(1+np.exp(input))

x = np.linspace(-99,99,200)
print(x)
y=[]
for i in range(len(x)):
    y.append(sigmoid(x[i]))

print(y)
plt.plot(x,y,'b-')
plt.show()




