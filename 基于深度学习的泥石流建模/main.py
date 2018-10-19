from 基于深度学习的泥石流建模 import const
from 基于深度学习的泥石流建模.dataset import deal_data
from 基于深度学习的泥石流建模.model import Net
import torch.optim as optim
import torch

#获取数据
deal_data(r'C:\Users\50828\PycharmProjects\hw_pytorch\基于深度学习的泥石流建模\西藏地灾隐患点（日喀则）.xls',1,const.user_col)

#新建一个神经网络
Neu_network = Net(13,100,100,4)
print(Neu_network)
#优化
optimizer = optim.SGD(Neu_network.parameters(),lr=0.01) #使用SGD优化函数
loss_func = torch.nn.CrossEntropyLoss()

#训练
for i in range(100):
    prediction = Neu_network(const.train)
    loss = loss_func(prediction,const.train_Value)
    #if(i%5==0):
    print('Loss=%.4f'%loss.item())

    optimizer.zero_grad()  # 将梯度置零，不置会与上一步的梯度进行相加
    loss.backward()  # 自动求导
    optimizer.step()

#测试



#存储模型
