"""
data_prepare_script.py

该脚本旨在自动化下载和预处理数据集的过程。
脚本执行以下任务：
1. 使用wget下载数据集。
2. 对数据集进行预处理：
   2.1 处理缺失值和None值。
   2.2 将类别列编码为one-hot编码。
   2.3 归一化数值列。

使用方法：
在安装了必要库（pandas, numpy, sklearn, wget）的Python环境中运行此脚本。


_orig (最原始的版本，刚下载)

_clean (缺失值、处理，标签列编码，标签列统一编码为y) 特征列 + y

_coded_one (one-hot 编码，归一化等)

作者：lvjiuluan
日期：2025年2月9日
"""
import os
import sys

import pandas as pd

sys.path.append('D:\PyCharmProjects\VFPUGEN')
sys.path.append(r'C:\Users\Administrator\PycharmProjects\VFPUGEN')
sys.path.append(r'/root/VFPUGEN')
sys.path.append(r'D:\PycharmProjects\VFPUGEN')

from consts.Constants import DATASETS_PATH
from utils.DatasetsPrepareUtils import download_file, preprocess_dataframe, preprocess_features

# 下载原始csv文件
download_file('https://raw.githubusercontent.com/lvjiuluan/DataSets/refs/heads/main/bank_orig.csv', DATASETS_PATH,
              'bank_orig.csv')
download_file('https://raw.githubusercontent.com/lvjiuluan/DataSets/refs/heads/main/census_orig.csv', DATASETS_PATH,
              'census_orig.csv')
download_file('https://raw.githubusercontent.com/lvjiuluan/DataSets/refs/heads/main/credit_orig.csv', DATASETS_PATH,
              'credit_orig.csv')

# 缺失值、处理，标签列编码，标签列统一编码为y 特征列 + y
# bank
df = pd.read_csv(os.path.join(DATASETS_PATH, 'bank_orig.csv'))
df = preprocess_dataframe(df)
df.to_csv(os.path.join(DATASETS_PATH, 'bank_clean.csv'), index=None)

# census
df = pd.read_csv(os.path.join(DATASETS_PATH, 'census_orig.csv'))
df = preprocess_dataframe(df)
df.to_csv(os.path.join(DATASETS_PATH, 'census_clean.csv'), index=None)

# credit
df = pd.read_csv(os.path.join(DATASETS_PATH, 'credit_orig.csv'), skiprows=1)
df = df.drop(columns=['ID'])
df = preprocess_dataframe(df)
df.to_csv(os.path.join(DATASETS_PATH, 'credit_clean.csv'), index=None)

# one-hot 编码，归一化

# bank
df = pd.read_csv(os.path.join(DATASETS_PATH, 'bank_clean.csv'))
category_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                     'cons.conf.idx', 'euribor3m', 'nr.employed']
target_variable = 'y'
df_one, df_label = preprocess_features(df, category_columns, numerical_columns, target_variable)
df_one.to_csv(os.path.join(DATASETS_PATH, 'bank_coded_onehot.csv'), index=None)
df_label.to_csv(os.path.join(DATASETS_PATH, 'bank_coded_label.csv'), index=None)

# census
df = pd.read_csv(os.path.join(DATASETS_PATH, 'census_clean.csv'))
category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country']
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
target_variable = 'y'
df_one, df_label = preprocess_features(df, category_columns, numerical_columns, target_variable)
df_one.to_csv(os.path.join(DATASETS_PATH, 'census_coded_onehot.csv'), index=None)
df_label.to_csv(os.path.join(DATASETS_PATH, 'census_coded_label.csv'), index=None)

# credit
df = pd.read_csv(os.path.join(DATASETS_PATH, 'credit_clean.csv'))
category_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numerical_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
target_variable = 'y'
df_one, df_label = preprocess_features(df, category_columns, numerical_columns, target_variable)
df_one.to_csv(os.path.join(DATASETS_PATH, 'credit_coded_onehot.csv'), index=None)
df_label.to_csv(os.path.join(DATASETS_PATH, 'credit_coded_label.csv'), index=None)
