

import torch
import numpy
import torch


train_Value =torch.LongTensor([2,0,1,3,1,2,3,2,2,2,1,2,0,0,1])
train_Value =train_Value.reshape(len(train_Value),1)
train_Value =torch.LongTensor(train_Value)

print(train_Value)
# for i in range(len(train_Value)):
#     if(train_Value[i] == 0):
#         y = torch.Tensor([1, 0, 0, 0])
#     elif(train_Value[i] == 1):
#         y = torch.LongTensor([0, 1, 0, 0])
#     elif (train_Value[i] == 2):
#         y = torch.Tensor([0, 0, 1, 0])
#         print(y)
#     else:
#         y = torch.Tensor([0, 0, 0, 1])

print(one_hot)