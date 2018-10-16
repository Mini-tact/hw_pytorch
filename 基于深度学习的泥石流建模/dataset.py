import pandas as pd
class deal_data():
    def __init__(self,file_path,sheet,usecols):
        self.datatset = pd.DataFrame(pd.read_excel(file_path,sheet_name=sheet,usecols=usecols))


    def _split_datat(self):
        return 1


    def data(self):
        return self.datatset

