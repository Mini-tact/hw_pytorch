import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)#在X上随机生成100个点
y=x.pow(2)+0.2*torch.rand(x.size()) #以X为点根据y=x^2进行上下浮动

x,y = Variable(x),Variable(y) #把x，y编程Variable类型

#定义一个三层全连接网络
class Net(torch.nn.Module):
    def __init__(self,layer_input,layer_hidden,layer_output): #初始化层信息
        super(Net,self).__init__()
        #定义一个线性化的层
        self.hidden_layer = torch.nn.Linear(layer_input,layer_hidden)  #输入为输入层的神经元个数和隐藏层神经元的个数
        self.output_layer = torch.nn.Linear(layer_hidden,layer_output)#输入隐藏层格合数和输出层的个数


    def forward(self,x):  #定义前向的传导过程，使用激励函数优化输出
        x = F.relu(self.hidden_layer(x))#把隐藏层输出的进行激励函数优化
        x=self.output_layer(x)
        return x

net = Net(1,10,1)  #实例化一个三层全连接网络
print(net)

plt.ion()  #设置成实时打印的过程
plt.show()

#优化
optimizer = optim.SGD(net.parameters(),lr=0.5) #使用SGD优化函数
loss_func = torch.nn.MSELoss() #使用MSE来优化输出与预期的值

for t in range(100):  #进行100次的优化
    prediction = net(x)    #得到预期的值

    loss = loss_func(prediction,y)#得到学习后的值

    optimizer.zero_grad() #将梯度置零，不置会与上一步的梯度进行相加
    loss.backward()  #自动求导
    optimizer.step()

    #学习过程的多动态化显示
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy())
        plt.text(0.5,0,'Loss=%.4f'%loss.item(),fontdict={'size':20,'color':'red'})
        plt.pause(0.1)




