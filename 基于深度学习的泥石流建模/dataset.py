import numpy as np
import pandas as pd
import torch.autograd.variable as variable
from sklearn.model_selection import train_test_split
import 基于深度学习的泥石流建模.const as const
import  torch


def dataframe_to_variable_double(param):
    #param=param.T
    return variable(torch.from_numpy(param.values).float())

def dataframe_to_variable_float(param):
    #param=param.T
    train_Value=variable(torch.from_numpy(param.values)).long()
    # train_Value = train_Value.reshape(len(train_Value), 1)
    # train_Value = torch.LongTensor(train_Value)
    # train_Value = torch.zeros(len(train_Value), 4).scatter_(1, train_Value, 1).float()
    return train_Value

class deal_data():
    def __init__(self,file_path,sheet,col):
        self.datatset = pd.DataFrame(pd.read_excel(file_path,sheet_name=sheet,usecols=col))
        self.deal_d()

    def deal_d(self):
        #将样本数字化
        data=self.datatset
        data = pd.DataFrame(data.replace(const.dictionary))

        #样本分割
        #train_all, Verification_all = train_test_split(data,test_size=0.2,random_state=0)  # 随机选择80%作为模型训练，剩余作为验证
        X_Train, X_Test = train_test_split(data, test_size=0.2, random_state=0)  # 随机选择80%作为测试集，剩余作为训练集
        #训练集
        train = X_Train.iloc[0:len(X_Train), 0:13]
        const.train = dataframe_to_variable_double(train) # 训练的部分

        train_Value = X_Train.iloc[0:len(X_Train), 13]  # 验证的部分
        const.train_Value = dataframe_to_variable_float(train_Value) # 训练的部分

        #测试集
        Test = X_Test.iloc[0:len(X_Test), 0:13]  # 训练的部分
        const.Test = dataframe_to_variable_double(Test)

        Test_Value = X_Test.iloc[0:len(X_Test), 13]  # 验证的部分
        const.Test_Value = dataframe_to_variable_float(Test_Value)










        #验证集
        # Verification = Verification_all.iloc[0:len(Verification_all), 0:13]  # 训练的部分
        # const.Verification = dataframe_to_variable(Verification)
        #
        # Verification_Value = Verification_all.iloc[0:len(Verification_all),13]  # 验证的部分
        # const.Verification_Value = dataframe_to_variable(Verification_Value)
        # SMOTE算法来增加少数类样本，使得样本平均，来提升模型的精确度




