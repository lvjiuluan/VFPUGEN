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

from consts.Constants import DATASETS_PATH


class DataSet:
    def __init__(self, baseFileName):
        self.baseFileName = baseFileName
        self.files = {
            "orig": f"{baseFileName}_orig.csv",
            "clean": f"{baseFileName}_clean.csv",
            "coded_label": f"{baseFileName}_coded_label.csv",
            "coded_onehot": f"{baseFileName}_coded_onehot.csv"
        }

        # 读取CSV文件
        self.orig_data = self._load_data("orig")
        self.clean_data = self._load_data("clean")
        self.coded_label_data = self._load_data("coded_label")
        self.coded_onehot_data = self._load_data("coded_onehot")

    def _load_data(self, file_type="orig"):
        """
        加载指定类型的 CSV 文件，并将其存储在 `self.data` 中。

        参数:
        - file_type (str): 要加载的文件类型，默认为 'orig'，可以是 'clean', 'coded_label', 或 'coded_onehot'

        返回:
        - pandas.DataFrame: 加载的数据
        """
        if file_type not in self.files:
            logging.error(f"无效的文件类型: {file_type}. 请选择 'orig', 'clean', 'coded_label', 或 'coded_onehot'.")
            raise ValueError(f"无效的文件类型: {file_type}.")

        file_path = os.path.join(DATASETS_PATH, self.files[file_type])

        if not os.path.exists(file_path):
            logging.error(f"文件 {file_path} 不存在!")
            raise FileNotFoundError(f"文件 {file_path} 不存在!")

        try:
            logging.info(f"加载文件: {file_path}")
            self.data = pd.read_csv(file_path)
            logging.info(f"成功加载 {file_type} 数据。")
            return self.data
        except Exception as e:
            logging.error(f"加载文件 {file_path} 时发生错误: {e}")
            raise


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
