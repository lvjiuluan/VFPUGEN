import sys
sys.path.append(r'D:\PyCharmProjects\VFPUGEN')
sys.path.append(r'/root/VFPUGEN')
sys.path.append(r'D:\PycharmProjects\VFPUGEN')
%load_ext autoreload
%autoreload 2

%matplotlib inline

from datasets.DataSet import BankDataset, CensusDataset, CreditDataset
from utils.DataProcessUtils import *
from consts.Constants import DATASETS_PATH
import os
from utils.FateUtils import convert_ipynb_to_py

bank_dataset, census_dataset, credit_dataset = BankDataset(), CensusDataset(), CreditDataset()

df_A, df_B, y = transform_and_save_df(
    bank_dataset.coded_onehot_data,
    split_cols=32,
    file_path=DATASETS_PATH,
    name_prefix='VFPU_GEN_BANK_',
    name_suffix='.csv'
)

df_A, df_B, y = transform_and_save_df(
    census_dataset.coded_onehot_data,
    split_cols=32,
    file_path=DATASETS_PATH,
    name_prefix='VFPU_GEN_CENSUS_',
    name_suffix='.csv'
)

df_A, df_B, y = transform_and_save_df(
    credit_dataset.coded_onehot_data,
    split_cols=32,
    file_path=DATASETS_PATH,
    name_prefix='VFPU_GEN_CREDIT_',
    name_suffix='.csv'
)

convert_ipynb_to_py('切分数据.ipynb','split_data')

