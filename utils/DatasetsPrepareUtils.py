import os
import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import subprocess
from consts.Constants import DATASETS_PATH
import shutil
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder


def prepare_updated_pollution_dataset(file_path):
    # 使用os.path.join构建文件路径
    csv_file_path = os.path.join(file_path, 'updated_pollution_dataset.csv')

    # 读取csv文件
    df = pd.read_csv(csv_file_path)

    # 将分类标签转换为数值标签
    le = LabelEncoder()
    df['Air Quality'] = le.fit_transform(df['Air Quality'])

    # 分离特征和目标变量
    X = df.drop(columns=['Air Quality'])
    y = df['Air Quality']

    # 对X进行归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 创建新的DataFrame并保存
    prepared_df = pd.DataFrame(data=X_scaled, columns=X.columns)
    prepared_df['Air Quality'] = y
    prepared_df.to_csv(os.path.join(file_path, 'updated_pollution_dataset_prepared.csv'), index=False)


def download_file(download_url: str, path: str, filename: str = None):
    """
    使用 Linux wget 下载文件到指定目录，并可指定文件名。
    如果目标文件已存在，则跳过下载。

    :param download_url: 要下载的文件链接
    :param path: 存储文件的目标目录
    :param filename: (可选) 指定保存的文件名，默认为 None，使用 wget 默认文件名
    """
    # 确保目标目录存在
    os.makedirs(path, exist_ok=True)

    # 如果未指定 filename，则从 URL 提取文件名
    if not filename:
        filename = os.path.basename(urlparse(download_url).path)

    # 计算最终文件路径
    output_path = os.path.join(path, filename)

    # 检查文件是否已存在
    if os.path.exists(output_path):
        print(f"✅ 文件已存在，跳过下载: {output_path}")
        return output_path  # 直接返回已存在的文件路径

    # 构造 wget 命令
    command = ["wget", "-c", "-O", output_path, download_url]  # `-c` 选项支持断点续传

    try:
        # 执行 wget 命令
        subprocess.run(command, check=True)
        print(f"✅ 文件已成功下载到: {output_path}")
        return output_path  # 返回下载的文件路径
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return None  # 失败返回 None


def reorder_columns(df, new_columns):
    # 检查new_columns是否包含所有的列
    if set(new_columns) == set(df.columns):
        # 按照new_columns的顺序重新排列列
        df = df[new_columns]
        return df
    else:
        raise ValueError("new_columns必须包含df的所有列")


def evaluate_imputed_data_various_metric(orig: pd.DataFrame, imputed: pd.DataFrame, discrete_columns: list) -> dict:
    """
    根据原始数据和插补数据，计算多种指标以评估插补效果。

    参数:
        orig: pd.DataFrame
            原始完整数据。
        imputed: pd.DataFrame
            经插补后的数据。
        discrete_columns: list
            分类列的列名列表，这些列为 one-hot 编码后的表示。

    返回:
        metrics: dict
            各项指标的字典，包含以下键：
                - 'rmse_all': 所有列的 RMSE。
                - 'rmse_categorical': 分类列的 RMSE。
                - 'rmse_numerical': 数值列的 RMSE。
                - 'rmse_avg': 分类列和数值列分别计算 RMSE 后相加的结果取平均。
                - 'cross_entropy_categorical': 分类列的交叉熵（假设 one-hot 编码，每行的交叉熵为 -log(预测概率)）。
                - 'rmse_numerical_for_cross_entropy': 数值列的 RMSE（用于方法3）。
                - 'f1_categorical': 分类列的平均 F1 分数（先将 one-hot 向量转换为类别标签）。
                - 'rmse_numerical_for_f1': 数值列的 RMSE（用于方法4）。

    其他可能的指标建议:
        - 数值列的平均绝对误差 (MAE)
        - 数值列的 R² 得分
        - 分类列的准确率 (Accuracy)
        - 分类列的 Kappa 统计量
        - 用 Jensen-Shannon 散度衡量分类概率分布的差异
    """
    # ----------------------------
    # 1. 所有列的 RMSE
    rmse_all = np.sqrt(np.mean((orig.values - imputed.values) ** 2))

    # ----------------------------
    # 将所有列分为分类列和数值列
    numerical_columns = [col for col in orig.columns if col not in discrete_columns]

    # 2a. 计算分类列的 RMSE（如果存在）
    if discrete_columns:
        rmse_categorical = np.sqrt(np.mean((orig[discrete_columns].values - imputed[discrete_columns].values) ** 2))
    else:
        rmse_categorical = None

    # 2b. 计算数值列的 RMSE（如果存在）
    if numerical_columns:
        rmse_numerical = np.sqrt(np.mean((orig[numerical_columns].values - imputed[numerical_columns].values) ** 2))
    else:
        rmse_numerical = None

    # 2. 将分类列和数值列的 RMSE 相加（如果其中一部分不存在，则只取存在的那部分）
    if rmse_categorical is not None and rmse_numerical is not None:
        rmse_avg = (rmse_categorical + rmse_numerical) / 2
    elif rmse_categorical is not None:
        rmse_avg = rmse_categorical
    elif rmse_numerical is not None:
        rmse_avg = rmse_numerical
    else:
        rmse_avg = None

    # ----------------------------
    # 3. 分类列用交叉熵，数值列用 RMSE
    #    对于 one-hot 编码的分类列，每行交叉熵为： -log(预测的正确类别概率)
    epsilon = 1e-12  # 防止 log(0)
    if discrete_columns:
        orig_cat = orig[discrete_columns].values
        imputed_cat = imputed[discrete_columns].values

        # 如果 imputed 的值没有保证每行归一化，则进行归一化处理（防止预测概率总和不为1）
        row_sums = imputed_cat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # 防止除以0
        imputed_cat_norm = imputed_cat / row_sums

        # 对于每行，真实类别为 one-hot 编码中值为1的位置
        true_class_indices = np.argmax(orig_cat, axis=1)
        # 取出预测的概率值
        pred_prob = imputed_cat_norm[np.arange(imputed_cat_norm.shape[0]), true_class_indices]
        cross_entropy_categorical = -np.mean(np.log(np.clip(pred_prob, epsilon, 1.0)))
    else:
        cross_entropy_categorical = None

    # 数值列的 RMSE 同上（用于方法3）
    rmse_numerical_for_cross_entropy = rmse_numerical

    # ----------------------------
    # 4. 分类列使用平均 F1 分数，数值列使用 RMSE
    #    先将 one-hot 编码转换为类别标签，然后计算 F1 分数（这里采用 macro 平均）
    if discrete_columns:
        true_labels = np.argmax(orig[discrete_columns].values, axis=1)
        pred_labels = np.argmax(imputed[discrete_columns].values, axis=1)
        try:
            f1_categorical = f1_score(true_labels, pred_labels, average='macro')
        except Exception as e:
            f1_categorical = None
            print(f"计算 F1 分数时出错: {e}")
    else:
        f1_categorical = None

    rmse_numerical_for_f1 = rmse_numerical

    # ----------------------------
    # 输出各项指标
    print("评估指标:")
    print(f"1. 全部列的 RMSE: {rmse_all:.6f}")
    if rmse_categorical is not None:
        print(f"2a. 分类列的 RMSE: {rmse_categorical:.6f}")
    if rmse_numerical is not None:
        print(f"2b. 数值列的 RMSE: {rmse_numerical:.6f}")
    if rmse_avg is not None:
        print(f"2. 分类列和数值列分别计算 RMSE 后相加: {rmse_avg:.6f}")
    if cross_entropy_categorical is not None:
        print(f"3a. 分类列的交叉熵: {cross_entropy_categorical:.6f}")
    if rmse_numerical_for_cross_entropy is not None:
        print(f"3b. 数值列的 RMSE (用于交叉熵方法): {rmse_numerical_for_cross_entropy:.6f}")
    if f1_categorical is not None:
        print(f"4a. 分类列的平均 F1 分数: {f1_categorical:.6f}")
    if rmse_numerical_for_f1 is not None:
        print(f"4b. 数值列的 RMSE (用于 F1 方法): {rmse_numerical_for_f1:.6f}")

    # 将所有指标存入字典返回
    metrics = {
        'rmse_all': rmse_all,
        'rmse_categorical': rmse_categorical,
        'rmse_numerical': rmse_numerical,
        'rmse_avg': rmse_avg,
        'cross_entropy_categorical': cross_entropy_categorical,
        'rmse_numerical_for_cross_entropy': rmse_numerical_for_cross_entropy,
        'f1_categorical': f1_categorical,
        'rmse_numerical_for_f1': rmse_numerical_for_f1
    }
    return metrics


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    对输入的 DataFrame 进行数据预处理：
    1. 处理缺失值（数值型填充均值，类别型填充众数）
    2. 对标签列（最后一列）进行标签编码，并转换为 int 类型
    3. 确保最后一列的列名为 'y'

    参数:
    df (pd.DataFrame): 输入的 DataFrame，最后一列为标签列

    返回:
    pd.DataFrame: 处理后的 DataFrame，最后一列的列名为 'y'
    """
    df = df.copy()  # 避免修改原始 df

    # 处理缺失值
    for col in df.columns:
        if df[col].dtype == 'object':  # 类别型数据
            df[col] = df[col].fillna(df[col].mode()[0])  # 用众数填充
        else:  # 数值型数据
            df[col] = df[col].fillna(df[col].mean())  # 用均值填充

    # 处理标签列
    label_col = df.columns[-1]  # 获取最后一列的列名
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])  # 进行标签编码
    df[label_col] = df[label_col].astype(int)  # 转换为 int 类型

    # 统一标签列名为 'y'
    df = df.rename(columns={label_col: 'y'})

    return df


def preprocess_features(df, category_columns, numerical_columns, target_variable):
    """
    预处理数据特征，包括：
    1. 对类别特征进行 One-Hot 编码
    2. 对数值特征进行标准化
    3. 返回经过 One-Hot 编码和标准化后的特征数据，以及仅进行标签编码的类别特征数据

    参数：
    df: pd.DataFrame - 输入数据
    category_columns: list - 需要进行编码的类别特征列
    numerical_columns: list - 需要进行标准化的数值特征列
    target_variable: str 或 None - 目标变量列名（如果有）

    返回：
    df_encoded: pd.DataFrame - One-Hot 编码和标准化后的数据
    df_label_encoded: pd.DataFrame - 仅进行 LabelEncoder 编码的类别数据
    """

    # 合并类别特征列和数值特征列
    feature_columns = category_columns + numerical_columns

    # 校验特征列数量是否正确
    if len(feature_columns) != df.shape[1] - (1 if target_variable else 0):
        raise ValueError("错误：类别特征列和数值特征列的总数必须等于数据中的特征列数量。")

    print("开始对类别特征进行 One-Hot 编码...")
    # 进行 One-Hot 编码
    enc = OneHotEncoder()
    enc.fit(df[category_columns])
    df_category_enc = pd.DataFrame(enc.transform(df[category_columns]).toarray(), columns=enc.get_feature_names_out())
    df_category_enc = df_category_enc.astype('int')

    print("类别特征 One-Hot 编码完成。")

    print("开始对数值特征进行标准化处理...")
    # 进行标准化
    scaler = StandardScaler()
    scaler.fit(df[numerical_columns])
    df_numerical_scaler = pd.DataFrame(scaler.transform(df[numerical_columns]), columns=scaler.feature_names_in_)
    print("数值特征标准化处理完成。")

    # 组合编码后的特征数据
    if target_variable is not None:
        df_onehot_encoded = pd.concat([df_category_enc, df_numerical_scaler, df[target_variable].to_frame()], axis=1)
    else:
        df_onehot_encoded = pd.concat([df_category_enc, df_numerical_scaler], axis=1)

    print("开始对类别特征进行标签编码...")
    # 进行标签编码
    df_label_encoded = label_encode_columns(df.copy(), category_columns)
    print("类别特征标签编码完成。")

    return df_onehot_encoded, df_label_encoded


def label_encode_columns(df: pd.DataFrame, category_columns: list) -> pd.DataFrame:
    """
    对类别特征进行标签编码（Label Encoding）。

    参数：
    df: pd.DataFrame - 输入数据
    category_columns: list - 需要进行标签编码的类别特征列

    返回：
    df: pd.DataFrame - 经过标签编码后的数据
    """

    encoder = LabelEncoder()

    for col in category_columns:
        if col in df.columns:
            print(f"对列 '{col}' 进行标签编码...")
            df[col] = encoder.fit_transform(df[col])
        else:
            print(f"⚠️ 警告：列 '{col}' 在 DataFrame 中未找到，跳过该列。")

    print("所有类别特征标签编码完成。")
    return df


def split_list_by_rate(lst, rates):
    """
    将列表 lst 随机打乱后，按照 rates 中指定的比例拆分成多个子列表。
    为了避免整数除法造成的元素遗漏，先按 int 计算，再根据小数部分的大小分配剩余元素。

    参数：
        lst: 需要拆分的列表
        rates: 拆分比例列表，所有比例和必须等于1

    返回：
        拆分后的列表集合（列表组成的列表）
    """
    random.shuffle(lst)
    n = len(lst)
    # 初步分配各部分的元素数（向下取整）
    counts = [int(r * n) for r in rates]
    remainder = n - sum(counts)
    if remainder > 0:
        # 计算每部分的小数余量，用于判断哪些部分优先补齐
        remainders = [(r * n - int(r * n), i) for i, r in enumerate(rates)]
        # 按余量从大到小排序
        remainders.sort(reverse=True)
        for j in range(remainder):
            counts[remainders[j][1]] += 1

    splits = []
    start = 0
    for count in counts:
        splits.append(lst[start:start + count])
        start += count
    return splits
