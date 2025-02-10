import sys
sys.path.append(r'D:\PyCharmProjects\VFPUGEN')
sys.path.append(r'/root/VFPUGEN')
sys.path.append(r'D:\PycharmProjects\VFPUGEN')
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

from consts.Constants import DATASETS_PATH
import pandas as pd
from utils.DatasetsPrepareUtils import split_list_by_rate
import os
from utils.FateUtils import convert_ipynb_to_py

rate_A = 0.5
rate_B = 0.5

# bank
category_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                    'poutcome']
numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                     'cons.conf.idx', 'euribor3m', 'nr.employed']
target_variable = 'y'
df = pd.read_csv(os.path.join(DATASETS_PATH,'bank_clean.csv'))
A_cat_colums, B_cat_colums = split_list_by_rate(category_columns,[rate_A,rate_B])
A_num_colums, B_num_colums = split_list_by_rate(numerical_columns,[rate_A,rate_B])
A_colums = A_cat_colums + A_num_colums
B_colums = B_cat_colums + B_num_colums + [target_variable]
df_A = df[A_colums]
df_B = df[B_colums]
df_A.to_csv(os.path.join(DATASETS_PATH,'bank_clean_A.csv'),index=None)
df_B.to_csv(os.path.join(DATASETS_PATH,'bank_clean_B.csv'),index=None)


# census
category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country']
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
numerical_columns = 'y'

df = pd.read_csv(os.path.join(DATASETS_PATH,'census_clean.csv'))
A_cat_colums, B_cat_colums = split_list_by_rate(category_columns,[rate_A,rate_B])
A_num_colums, B_num_colums = split_list_by_rate(numerical_columns,[rate_A,rate_B])
A_colums = A_cat_colums + A_num_colums
B_colums = B_cat_colums + B_num_colums + [target_variable]
df_A = df[A_colums]
df_B = df[B_colums]
df_A.to_csv(os.path.join(DATASETS_PATH,'census_clean_A.csv'),index=None)
df_B.to_csv(os.path.join(DATASETS_PATH,'census_clean_B.csv'),index=None)


# credit
category_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numerical_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
target_variable = 'y'

df = pd.read_csv(os.path.join(DATASETS_PATH,'credit_clean.csv'))
A_cat_colums, B_cat_colums = split_list_by_rate(category_columns,[rate_A,rate_B])
A_num_colums, B_num_colums = split_list_by_rate(numerical_columns,[rate_A,rate_B])
A_colums = A_cat_colums + A_num_colums
B_colums = B_cat_colums + B_num_colums + [target_variable]
df_A = df[A_colums]
df_B = df[B_colums]
df_A.to_csv(os.path.join(DATASETS_PATH,'credit_clean_A.csv'),index=None)
df_B.to_csv(os.path.join(DATASETS_PATH,'credit_clean_B.csv'),index=None)



