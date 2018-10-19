import torch
import torch.nn as nn

class Net(nn.Module):
    #13，100，100，4
    def __init__(self,input,hidden_1,hidden_2,out_put):
        super(Net,self).__init__()
        layer_1 = nn.Sequential()
        layer_1.add_module('Liner_1',nn.Linear(input,hidden_1))
        layer_1.add_module('tanh_1',nn.ReLU(True))
        layer_1.add_module('drop_1',nn.Dropout())
        self.layer_1 = layer_1

        layer_2 = nn.Sequential()
        layer_2.add_module('Liner_2', nn.Linear(hidden_1, hidden_2))
        layer_2.add_module('tanh_2',nn.ReLU(True))
        layer_2.add_module('drop_2',nn.Dropout())
        self.layer_2 = layer_2

        layer_3 = nn.Sequential()
        layer_3.add_module('Liner_3', nn.Linear(hidden_2, out_put))
        layer_3.add_module('softmax',nn.Softmax(dim=1))
        self.layer_3 = layer_3

    def forward(self,net):
        layer_1 = self.layer_1(net)
        #print("layer_1="+str(layer_1.shape))
        layer_2 = self.layer_2(layer_1)
        #print("layer_2=" + str(layer_2.shape))
        out_value = self.layer_3(layer_2)
        return out_value