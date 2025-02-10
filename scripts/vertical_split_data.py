import sys
sys.path.append(r'D:\PyCharmProjects\VFPUGEN')
sys.path.append(r'/root/VFPUGEN')
sys.path.append(r'D:\PycharmProjects\VFPUGEN')
# %load_ext autoreload
# %autoreload 2
#
# %matplotlib inline

import argparse
from datasets.DataSet import BankDataset, CensusDataset, CreditDataset
from utils.DataProcessUtils import transform_and_save_df
from consts.Constants import DATASETS_PATH


def main(split_cols=32):
    """
    主函数，执行数据转换和保存。

    参数:
    - split_cols (int): 列分割数，默认值为 32。
    """
    # 初始化数据集
    bank_dataset, census_dataset, credit_dataset = BankDataset(), CensusDataset(), CreditDataset()

    # 执行转换并保存数据
    df_A, df_B, y = transform_and_save_df(
        bank_dataset.coded_onehot_data,
        split_cols=split_cols,
        file_path=DATASETS_PATH,
        name_prefix='VFPU_GEN_BANK_',
        name_suffix='.csv'
    )

    df_A, df_B, y = transform_and_save_df(
        census_dataset.coded_onehot_data,
        split_cols=split_cols,
        file_path=DATASETS_PATH,
        name_prefix='VFPU_GEN_CENSUS_',
        name_suffix='.csv'
    )

    df_A, df_B, y = transform_and_save_df(
        credit_dataset.coded_onehot_data,
        split_cols=split_cols,
        file_path=DATASETS_PATH,
        name_prefix='VFPU_GEN_CREDIT_',
        name_suffix='.csv'
    )


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Process datasets and save the results.')
    parser.add_argument('--split_cols', type=int, default=32, help='Number of columns to split. Default is 32.')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数并传递参数
    main(split_cols=args.split_cols)

