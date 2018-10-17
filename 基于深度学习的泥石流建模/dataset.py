import pandas as pd
import 基于深度学习的泥石流建模.const as const
from sklearn.model_selection import train_test_split

class deal_data():
    def __init__(self,file_path,sheet,col):
        self.datatset = pd.DataFrame(pd.read_excel(file_path,sheet_name=sheet,usecols=col))
        self.deal_d()

    def deal_d(self):
        #将样本数字化
        data=self.datatset
        data = pd.DataFrame(data.replace(const.dictionary))

        #样本分割
        train, Verification = train_test_split(data,test_size=0.8,random_state=0)  # 随机选择80%作为模型训练，剩余作为验证
        X_Train, X_Test = train_test_split(train, test_size=0.8, random_state=0)  # 随机选择80%作为测试集，剩余作为训练集
        #训练集
        const.train = X_Train.iloc[0:len(data), 0:13]  # 训练的部分
        const.train_Value = X_Train.iloc[0:len(data), 13]  # 验证的部分
        #测试集
        const.Test = X_Test.iloc[0:len(data), 0:13]  # 训练的部分
        const.Test_Value = X_Test.iloc[0:len(data), 13]  # 验证的部分
        #验证集
        const.Verification = Verification.iloc[0:len(data), 0:13]  # 训练的部分
        const.Verification_Value = Verification.iloc[0:len(data), 13]  # 验证的部分
        # SMOTE算法来增加少数类样本，使得样本平均，来提升模型的精确度



