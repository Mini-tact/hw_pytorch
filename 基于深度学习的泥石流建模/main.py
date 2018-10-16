from 基于深度学习的泥石流建模.dataset import deal_data

data_col=[123,124,126,127,128,129,130,131,132,133,134,135,136]
data = deal_data('X:\pytorch\基于深度学习的山体滑坡\西藏地灾隐患点（日喀则）.xls',1,data_col)
print(data.data())
