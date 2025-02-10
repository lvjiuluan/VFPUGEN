import numpy as np
import pandas as pd
import pickle
import os
import logging
from enums.SplitRatio import SplitRatio
from utils.DataProcessUtils import vertical_split
from consts.Constants import DATASETS_PATH
from enums.HideRatio import HideRatio

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd


class DataSet:
    def __init__(self, baseFileName):
        self.baseFileName = baseFileName
        self.orig_file = f"{baseFileName}_orig.csv"
        self.clean_file = f"{baseFileName}_clean.csv"
        self.coded_label_file = f"{baseFileName}_coded_label.csv"
        self.coded_onehot_file = f"{baseFileName}_coded_onehot.csv"

        # 读取CSV文件
        self.orig_data = pd.read_csv(self.orig_file)
        self.clean_data = pd.read_csv(self.clean_file)
        self.coded_label_data = pd.read_csv(self.coded_label_file)
        self.coded_onehot_data = pd.read_csv(self.coded_onehot_file)


class BankDataset(DataSet):
    def __init__(self):
        super().__init__("bank")


class CensusDataset(DataSet):
    def __init__(self):
        super().__init__("census")


class CreditDataset(DataSet):
    def __init__(self):
        super().__init__("credit")


def get_all_dataset():
    return [BankDataset(), CensusDataset(), CreditDataset()]
