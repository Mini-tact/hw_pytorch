from openpyxl import load_workbook
from openpyxl.styles import Alignment
import pandas as pd
# 创建一个engine='openpyxl'的 ExcelWriter 对象 writer

class wirte_data_into_excle():
    def __init__(self,filename):
        self.writer = pd.ExcelWriter(filename, engine='openpyxl')
        self.filename = filename
    def write(self):
        return 1



