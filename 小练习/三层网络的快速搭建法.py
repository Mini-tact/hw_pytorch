import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+torch.rand(x.size())
x,y=Variable(x),Variable(y)

plt.show()
layer =torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)

optimizer = optim.SGD(layer.parameters(),lr=0.5)
loss_func = torch.nn.MSELoss()

plt.ion()  #设置成实时打印的过程
plt.show()
print(layer)

for t in range(100):
    prediction=layer
    loss = loss_func(prediction,y)

    optimizer.zero_grad()  # 将梯度置零，不置会与上一步的梯度进行相加
    loss.backward()  # 自动求导
    optimizer.step()

    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy())
        plt.text(0.5,0,'Loss=%.4f'%loss.item(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
