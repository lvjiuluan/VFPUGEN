"""
本脚本用于将多个数据集按照给定的拆分比例，将特征随机拆分为 A 方和 B 方两部分。
例如：对于 bank 数据集，将特征分为 A 方（不包含目标变量）和 B 方（包含目标变量）。
拆分时将类别型特征和数值型特征分别随机打乱后按照比例分配。

使用方法：
    python split_datasets.py --rate_A 0.5 --rate_B 0.5 --datasets_path ./datasets

注意：
    1. rate_A + rate_B 必须等于 1
    2. 脚本中会对每个数据集进行日志记录，便于排查数据读取和输出问题
"""


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
import os
import argparse
import logging
import pandas as pd

#!/usr/bin/env python
# -*- coding: utf-8 -*-


def process_dataset(dataset_name, input_file, output_file_A, output_file_B,
                    category_columns, numerical_columns, target_variable,
                    rate_A, rate_B):
    """
    根据指定的列配置和拆分比例处理数据集：
        1. 读取 CSV 文件
        2. 随机拆分类别型和数值型特征列表
        3. 生成 A 方（不包含目标变量）和 B 方（包含目标变量）的列集合
        4. 保存拆分后的数据到指定的 CSV 文件中

    参数：
        dataset_name: 数据集名称（用于日志输出）
        input_file: 输入 CSV 文件路径
        output_file_A: A 方数据保存路径
        output_file_B: B 方数据保存路径
        category_columns: 数据集中所有类别型特征（列表）
        numerical_columns: 数据集中所有数值型特征（列表）
        target_variable: 目标变量名称（字符串）
        rate_A: A 方特征拆分比例
        rate_B: B 方特征拆分比例
    """
    logging.info(f"开始处理数据集：{dataset_name}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logging.error(f"读取 {input_file} 失败，错误信息：{e}")
        return

    logging.info(f"{dataset_name} 数据读取完成，数据形状：{df.shape}")

    # 对类别型和数值型特征分别进行随机拆分
    A_cat_columns, B_cat_columns = split_list_by_rate(category_columns.copy(), [rate_A, rate_B])
    A_num_columns, B_num_columns = split_list_by_rate(numerical_columns.copy(), [rate_A, rate_B])

    A_columns = A_cat_columns + A_num_columns
    # B 方特征需要额外包含目标变量
    B_columns = B_cat_columns + B_num_columns + [target_variable]

    # 检查拆分后的列是否都存在于数据集中
    missing_A = set(A_columns) - set(df.columns)
    missing_B = set(B_columns) - set(df.columns)
    if missing_A:
        logging.warning(f"{dataset_name} 中 A 方缺少以下列：{missing_A}")
    if missing_B:
        logging.warning(f"{dataset_name} 中 B 方缺少以下列：{missing_B}")

    # 只选择数据集中存在的列
    df_A = df[[col for col in A_columns if col in df.columns]]
    df_B = df[[col for col in B_columns if col in df.columns]]

    try:
        df_A.to_csv(output_file_A, index=False)
        df_B.to_csv(output_file_B, index=False)
        logging.info(f"{dataset_name} 处理完成：A 方数据保存至 {output_file_A}；B 方数据保存至 {output_file_B}")
    except Exception as e:
        logging.error(f"{dataset_name} 数据保存失败，错误信息：{e}")


def main(rate_A, rate_B, datasets_path):
    """
    主函数，根据指定的拆分比例和数据集路径依次处理 bank、census、credit 数据集。
    """
    # 处理 bank 数据集
    process_dataset(
        dataset_name="bank",
        input_file=os.path.join(datasets_path, 'bank_clean.csv'),
        output_file_A=os.path.join(datasets_path, 'bank_clean_A.csv'),
        output_file_B=os.path.join(datasets_path, 'bank_clean_B.csv'),
        category_columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
                          'contact', 'month', 'day_of_week', 'poutcome'],
        numerical_columns=['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                           'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'],
        target_variable='y',
        rate_A=rate_A,
        rate_B=rate_B
    )

    # 处理 census 数据集
    process_dataset(
        dataset_name="census",
        input_file=os.path.join(datasets_path, 'census_clean.csv'),
        output_file_A=os.path.join(datasets_path, 'census_clean_A.csv'),
        output_file_B=os.path.join(datasets_path, 'census_clean_B.csv'),
        category_columns=['workclass', 'education', 'marital-status', 'occupation',
                          'relationship', 'race', 'sex', 'native-country'],
        numerical_columns=['age', 'fnlwgt', 'education-num', 'capital-gain',
                           'capital-loss', 'hours-per-week'],
        target_variable='y',
        rate_A=rate_A,
        rate_B=rate_B
    )

    # 处理 credit 数据集
    process_dataset(
        dataset_name="credit",
        input_file=os.path.join(datasets_path, 'credit_clean.csv'),
        output_file_A=os.path.join(datasets_path, 'credit_clean_A.csv'),
        output_file_B=os.path.join(datasets_path, 'credit_clean_B.csv'),
        category_columns=['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3',
                          'PAY_4', 'PAY_5', 'PAY_6'],
        numerical_columns=['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                           'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
                           'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'],
        target_variable='y',
        rate_A=rate_A,
        rate_B=rate_B
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="按照指定比例拆分数据集特征，生成 A 方（不含目标变量）和 B 方（含目标变量）数据。"
    )
    parser.add_argument("--rate_A", type=float, default=0.5, help="A 方拆分比例，默认 0.5")
    parser.add_argument("--rate_B", type=float, default=0.5, help="B 方拆分比例，默认 0.5")
    args = parser.parse_args()

    # 配置日志：输出时间、日志级别和消息内容
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 检查拆分比例是否合法
    if abs(args.rate_A + args.rate_B - 1.0) > 1e-6:
        logging.error("rate_A 与 rate_B 之和必须等于 1！")
        exit(1)

    main(rate_A=args.rate_A, rate_B=args.rate_B, datasets_path=DATASETS_PATH)




