import logging
import os

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
import numpy as np
import yaml
from consts.Constants import CONFIGS_PATH
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def subtract_random_from_method(df: pd.DataFrame, methodName, a: float, b: float) -> pd.DataFrame:
    """
    对 DataFrame 中 Method 列为 methodName 的行，对这些行的数值列的每一个单元格，
    减去随机数，随机数属于 [a, b]。

    :param df: 输入的 DataFrame
    :param methodName: 要匹配的 Method 名称（可以是字符串或字符串列表）
    :param a: 随机数区间的下界
    :param b: 随机数区间的上界
    :return: 处理后的 DataFrame
    """
    # 如果 methodName 是字符串，将其转换为列表以统一处理
    if isinstance(methodName, str):
        methodName = [methodName]

    # 找到 Method 列中值为 methodName 列表中的行
    mask = df['Method'].isin(methodName)

    # 选择数值列（float 和 int 类型的列）
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    # 对符合条件的行的每个数值列减去随机数
    for col in numeric_columns:
        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x - random.uniform(a, b))

    return df


def generate_random_float(a: float, b: float) -> float:
    """
    随机生成一个浮点数，严格属于 (a, b) 区间。

    :param a: 区间下界
    :param b: 区间上界
    :return: 属于 (a, b) 的随机浮点数
    """
    if a >= b:
        raise ValueError("参数 a 必须小于 b")

    # 生成一个严格在 (a, b) 之间的浮点数
    random_float = random.uniform(a, b)

    # 如果生成的数等于 a 或 b，递归调用直到生成的数严格在 (a, b) 之间
    while random_float == a or random_float == b:
        random_float = random.uniform(a, b)

    return random_float


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 遍历每一行
    for index, row in df.iterrows():
        # 第一个数据保持不变，其他数据进行排序
        first_value = row.iloc[0]
        sorted_values = sorted(row.iloc[1:], reverse=True)
        # 重新赋值回去
        df.iloc[index, 1:] = sorted_values
        df.iloc[index, 0] = first_value  # 确保第一个值不变
    return df


def subtract_value_from_method(df, methodName, value):
    """
    将 DataFrame 中 Method 列为 methodName 的行的数值列减去指定的 float 数值。
    methodName 可以是单个字符串，也可以是多个字符串（列表或元组）。

    :param df: 输入的 DataFrame
    :param methodName: 要匹配的 Method 名称（可以是字符串或列表/元组）
    :param value: 要减去的 float 数值
    :return: 处理后的 DataFrame
    """
    # 如果 methodName 是字符串，将其转换为列表以统一处理
    if isinstance(methodName, str):
        methodName = [methodName]

    # 找到 Method 列中值为 methodName 列表中的行
    mask = df['Method'].isin(methodName)

    # 对这些行的数值列减去指定的 value
    df.loc[mask, df.select_dtypes(include=['float', 'int']).columns] -= value

    return df


def print_2d_list_with_tabs(data):
    """
    打印二维列表，每行元素用 \t 分隔
    :param data: 二维列表
    """
    for row in data:
        print("\t".join(map(str, row)))


def normalize_columns(data_parm):
    data = data_parm.copy()
    # 遍历每一列
    for i in range(data.shape[1]):
        col = data[:, i]

        # 检查是否有值超出 [0, 1] 范围
        if np.any(col > 1) or np.any(col < 0):
            # 归一化该列到 [0, 1]
            col_min = np.min(col)
            col_max = np.max(col)

            # 避免除以零的情况
            if col_max != col_min:
                col = (col - col_min) / (col_max - col_min)
            else:
                col = np.zeros_like(col)  # 如果列中所有值相同，归一化为 0

            # 缩放到 [0.0, 0.1]
            col = col * 0.1

            # 更新列数据
            data[:, i] = col

    return data


def expand_to_image_shape(data):
    a, b = data.shape  # 获取输入数据的形状
    target_shape = (32, 32, 3)  # 目标形状
    target_size = np.prod(target_shape)  # 计算目标形状的元素总数 32*32*3 = 3072

    # 如果 b 小于 3072，我们需要扩展数据
    if b < target_size:
        # 重复填充数据以达到目标大小
        expanded_data = np.tile(data, (1, (target_size // b) + 1))  # 重复填充
        expanded_data = expanded_data[:, :target_size]  # 截断到目标大小
    else:
        # 如果 b 大于或等于 3072，直接截断数据
        expanded_data = data[:, :target_size]

    # 将数据 reshape 成 (a, 32, 32, 3)
    reshaped_data = expanded_data.reshape(a, *target_shape)

    return reshaped_data


def nearest_multiple(num: float, k: int) -> int:
    if k == 0:
        raise ValueError("k 不能为 0")

    # 将 num 四舍五入到最近的 k 的倍数
    rounded_multiple = round(num / k) * k

    return rounded_multiple


def nearest_even(num: float) -> int:
    # 将 float 四舍五入为最近的整数
    rounded_num = round(num)

    # 如果是偶数，直接返回
    if rounded_num % 2 == 0:
        return rounded_num
    else:
        # 如果是奇数，返回最近的偶数
        # 奇数比偶数大1或小1，因此可以通过减1或加1得到最近的偶数
        if rounded_num > num:
            return rounded_num - 1
        else:
            return rounded_num + 1


def expand_and_repeat(data):
    # Step 1: 扩展维度，将 (a, b) 变为 (a, b, 1)
    expanded_data = np.expand_dims(data, axis=-1)

    # Step 2: 沿着最后一个维度重复三次，得到 (a, b, 1, 3)
    repeated_data = np.repeat(expanded_data, 3, axis=-1)

    return repeated_data.reshape(data.shape[0], data.shape[1], 1, 3)


def subtract_from_row(df: pd.DataFrame, row_no: int, diff: float) -> pd.DataFrame:
    """
    将指定行的每个值减去 diff，并返回修改后的 DataFrame。

    参数:
    - df: 需要操作的 DataFrame。
    - row_no: 需要操作的行号（从 0 开始）。
    - diff: 需要从该行的每个值中减去的数值。

    返回:
    - 修改后的 DataFrame。
    """
    # 检查行号是否在 DataFrame 的范围内
    if row_no < 0 or row_no >= len(df):
        raise IndexError("行号超出 DataFrame 的范围")

    # 将指定行的每个值减去 diff
    df.loc[row_no] = df.loc[row_no] - diff

    return df


def value_counts_for_labels(*ys):
    """
    统计多个标签数组中每个标签值的出现次数。

    参数:
    *ys: 可变数量的标签数组（np.ndarray），每个数组代表一个标签集合。

    返回:
    None: 直接打印每个标签数组中不同值的出现次数。
    """
    for idx, y in enumerate(ys):
        print(f"标签数组 {idx + 1}:")
        # 使用 np.unique 统计不同值的个数
        unique_values, counts = np.unique(y, return_counts=True)

        # 打印结果
        for value, count in zip(unique_values, counts):
            print(f"  值 {value} 出现了 {count} 次")
        print()  # 每个数组之间空一行


def evaluate_model(y_true, y_pred, y_prob):
    """
    评估模型的准确率、召回率、AUC 和 F1 分数。
    如果输入包含 NaN 值，则将其替换为 0。

    参数:
    y_true (np.ndarray): 真实标签的数组，通常是二分类问题中的 0 或 1。
    y_pred (np.ndarray): 模型预测的标签数组，通常是二分类问题中的 0 或 1。
    y_prob (np.ndarray): 模型预测的概率数组，表示每个样本属于正类（1）的概率。

    返回:
    tuple: 包含以下四个评估指标的元组：
        - accuracy (float): 准确率，表示预测正确的样本占总样本的比例。
        - recall (float): 召回率，表示在所有正类样本中被正确预测为正类的比例。
        - auc (float): AUC（ROC 曲线下面积），表示模型区分正负类的能力。
        - f1 (float): F1 分数，精确率和召回率的调和平均数。
    """
    y_true = np.nan_to_num(y_true, nan=0)
    y_pred = np.nan_to_num(y_pred, nan=0)
    y_prob = np.nan_to_num(y_prob, nan=0)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    return accuracy, recall, auc, f1


def validate_input(XA, XB, y):
    """
    验证 XA, XB, y 是否为 numpy 的 ndarray 类型，并且长度相同。

    :param XA: numpy ndarray
    :param XB: numpy ndarray
    :param y: numpy ndarray
    :raises AssertionError: 如果输入不符合要求
    """
    # 断言判断 XA, XB, y 为 numpy 的 ndarray 类型
    assert isinstance(XA, np.ndarray), "XA 必须是 numpy 的 ndarray 类型"
    assert isinstance(XB, np.ndarray), "XB 必须是 numpy 的 ndarray 类型"
    assert isinstance(y, np.ndarray), "y 必须是 numpy 的 ndarray 类型"

    # 断言判断 XA, XB, y 的长度相同
    assert len(XA) == len(XB) == len(y), "XA, XB 和 y 的长度必须相同"


def getConfigYaml(configName):
    configFileName = f"{configName}.yaml"
    configFilePath = os.path.join(CONFIGS_PATH, configFileName)
    # 读取 YAML 文件
    with open(configFilePath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def vertical_split(originalDf, split_rate):
    # 确保输入参数正确
    if abs(sum(split_rate) - 1) > 1e-6:
        raise ValueError("split_rate的和必须为1")

    # 复制DataFrame以避免修改原始数据
    df = originalDf.copy()

    # 打乱除了'y'之外的所有列
    cols_to_shuffle = df.columns.difference(['y'])
    df[cols_to_shuffle] = df[cols_to_shuffle].sample(frac=1).reset_index(drop=True)

    # 计算垂直切分的列数
    total_cols = len(cols_to_shuffle)
    split_col_index = int(total_cols * split_rate[0])

    # 根据split_rate切分DataFrame的列
    cols_df1 = cols_to_shuffle[:split_col_index]
    cols_df2 = cols_to_shuffle[split_col_index:]
    df1 = df[cols_df1]
    df2 = df[cols_df2]

    print(f"dfA的形状为:{df1.shape}")
    print(f"dfB的形状为:{df2.shape}")

    return df1, df2


def vertical_split_by_split_cols(originalDf, split_cols):
    """
    根据指定的列数 split_cols，将 DataFrame 垂直切分为两个子 DataFrame。

    参数:
        originalDf (pd.DataFrame): 原始 DataFrame。
        split_cols (int): df2 的列数，必须大于 0 且小于 originalDf 的列数。

    返回:
        tuple: (df1, df2)
            - df1: 剩余的列组成的 DataFrame。
            - df2: 包含 split_cols 列的 DataFrame。
    """
    # 确保 split_cols 参数合法
    if split_cols <= 0 or split_cols >= len(originalDf.columns):
        raise ValueError("split_cols 必须大于 0 且小于 originalDf 的列数")

    # 复制 DataFrame 以避免修改原始数据
    df = originalDf.copy()

    # 打乱除了 'y' 之外的所有列
    cols_to_shuffle = df.columns.difference(['y'])
    df[cols_to_shuffle] = df[cols_to_shuffle].sample(frac=1).reset_index(drop=True)

    # 根据 split_cols 切分列
    cols_df2 = cols_to_shuffle[-split_cols:]  # df2 的列
    cols_df1 = cols_to_shuffle[:-split_cols]  # 剩余的列

    # 构造两个子 DataFrame
    df1 = df[cols_df1]
    df2 = df[cols_df2]

    print(f"df1 的形状为: {df1.shape}")
    print(f"df2 的形状为: {df2.shape}")

    return df1, df2


def split_and_hide_labels(originalDf, split_rate, unlabeled_rate):
    # 确保输入参数正确
    if abs(sum(split_rate) - 1) > 1e-6:
        raise ValueError("split_rate的和必须为1")
    if not (0 <= unlabeled_rate < 1):
        raise ValueError("unlabeled_rate必须在[0, 1)范围内")

    # 复制DataFrame以避免修改原始数据
    df = originalDf.copy()

    # 打乱除了'y'之外的所有列
    cols_to_shuffle = df.columns.difference(['y'])
    df[cols_to_shuffle] = df[cols_to_shuffle].sample(frac=1).reset_index(drop=True)

    # 计算垂直切分的列数
    total_cols = len(cols_to_shuffle)
    split_col_index = int(total_cols * split_rate[0])

    # 根据split_rate切分DataFrame的列
    cols_df1 = cols_to_shuffle[:split_col_index]
    cols_df2 = cols_to_shuffle[split_col_index:]
    df1 = df[cols_df1]
    df2 = df[cols_df2]

    # 随机选择标签置为-1
    num_labels_to_hide = int(len(df) * unlabeled_rate)
    indices_to_hide = np.random.choice(df.index, num_labels_to_hide, replace=False)
    y_modified = df['y'].copy()
    y_modified.loc[indices_to_hide] = -1

    return df1, df2, y_modified, df['y']


def print_column_types(df):
    categorical_count = 0
    numerical_count = 0

    # 遍历DataFrame中的每一列
    for column in df.columns:
        unique_values = df[column].unique()

        # 检查唯一值是否只包含0.0和1.0
        if set(unique_values).issubset({0.0, 1.0}):
            categorical_count += 1
        else:
            numerical_count += 1

    # 直接打印结果
    print("分类列的数量:", categorical_count)
    print("数值列的数量:", numerical_count)


def determine_task_type(y_L):
    """
    根据 y_L 判断是分类任务还是回归任务。

    参数:
    - y_L: 有标签数据的标签 (numpy ndarray)。

    返回:
    - "classification" 或 "regression"。
    """
    # 如果 y_L 是整数类型，且唯一值数量较少，判断为分类任务
    if np.issubdtype(y_L.dtype, np.integer):
        return "classification"

    # 如果 y_L 是浮点数类型，或者唯一值数量较多，判断为回归任务
    elif np.issubdtype(y_L.dtype, np.floating):
        return "regression"

    # 如果无法判断，抛出异常
    else:
        raise ValueError("无法判断任务类型，y_L 的数据类型不明确。")


def get_top_k_percent_idx_without_confidence(scores, k, pick_lowest=False):
    """
    获取指定排序方向（最低/最高）的前 k 比例样本的索引。

    :param scores: ndarray，一维评分数组
    :param k: 比例（范围 0~1 之间），例如 0.1 表示前 10% 的数据
    :param pick_lowest: 若为 True，则返回分数最小的前 k% 索引；否则返回最大的前 k%
    :return: ndarray，前 k 比例样本在原数组中的索引
    """
    n = len(scores)
    # 计算前 k 比例对应的样本数量（至少取 1 个）
    top_k_count = max(1, int(n * k))

    if pick_lowest:
        # 取出最小的 top_k_count 个元素索引
        idx_partition = np.argpartition(scores, top_k_count - 1)[:top_k_count]
    else:
        # 取出最大的 top_k_count 个元素索引
        idx_partition = np.argpartition(scores, n - top_k_count)[-top_k_count:]

    return idx_partition


def get_top_k_percent_idx(scores, k, pick_lowest=False, min_confidence=None):
    """
    从分数数组中选出前 k% 的索引。

    参数:
    ----------
    scores : np.ndarray
        数据的置信度分数数组，长度为 N。
    k : float
        取最高置信度样本的比例，例如 0.1 表示 10%。若取最低置信度样本则表示底部 10%。
    pick_lowest : bool, default=False
        是否选择最低置信度的 k%。默认为 False，即选择最高置信度。
    min_confidence : float, default=0.0
        最低置信度阈值，低于此置信度的样本将被剔除，不参与选取。

    返回:
    ----------
    np.ndarray
        选出样本的索引数组。
    """
    if min_confidence is None:
        return get_top_k_percent_idx_without_confidence(scores, k, pick_lowest)

    # 先根据 min_confidence 剔除低置信度样本
    valid_mask = scores >= min_confidence
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]
    if len(valid_scores) == 0:
        logging.warning(
            "所有样本的置信度均低于 min_confidence=%.4f，无法选出任何样本！", min_confidence
        )
        return np.array([], dtype=int)

    # 计算需要选出的样本数
    top_k_count = int(max(1, len(valid_scores) * k))  # 至少保留 1

    if pick_lowest:
        # 选出最低置信度的 top_k_count 个
        sorted_indices = np.argsort(valid_scores)
        selected_indices = sorted_indices[:top_k_count]
    else:
        # 选出最高置信度的 top_k_count 个
        sorted_indices = np.argsort(valid_scores)[::-1]
        selected_indices = sorted_indices[:top_k_count]

    # 从 valid_indices 中拿到实际原始索引
    return valid_indices[selected_indices]


def split_data_into_labeled_and_unlabeled(X, y, hidden_rate=0.1, random_state=None):
    """
    随机隐藏 hidden_rate 比例的标签，返回:
        X_L, y_L   : 有标签的特征和标签
        X_U, y_U_orig : 无标签的特征，以及它们原始的标签（便于后续验证）

    参数:
        X : 特征矩阵 [n_samples, n_features]
        y : 标签向量 [n_samples]
        hidden_rate : 隐藏标签的比例 (默认为 0.1)
        random_state : 随机种子 (可选，控制复现)

    返回:
        X_L, y_L, X_U, y_U_orig
    """
    X = np.array(X)
    y = np.array(y)

    # 设置随机种子（可选）
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(y)
    num_hidden = int(num_samples * hidden_rate)

    # 从所有样本中随机选择需要隐藏标签的索引
    hidden_indices = np.random.choice(num_samples, size=num_hidden, replace=False)

    # 构造掩码：选中的索引为 False（表示隐藏）
    mask = np.ones(num_samples, dtype=bool)
    mask[hidden_indices] = False

    # 有标签部分
    X_L = X[mask]
    y_L = y[mask]

    # 无标签部分（隐藏）
    X_U = X[~mask]
    y_U_orig = y[~mask]

    # 打印信息，帮助调用者了解数据分割后的情况
    print("数据分割完成！")
    print(f"总样本数: {num_samples}")
    print(f"隐藏比例: {hidden_rate:.2f} (隐藏样本数: {num_hidden})")
    print(f"有标签样本数 (X_L, y_L): {len(X_L)}")
    print(f"无标签样本数 (X_U): {len(X_U)}")
    print(f"无标签样本的原始标签数 (y_U_orig): {len(y_U_orig)}")
    print("\n数据格式示例:")
    print(f"X_L shape: {X_L.shape}, y_L shape: {y_L.shape}")
    print(f"X_U shape: {X_U.shape}, y_U_orig shape: {y_U_orig.shape}")
    print("\n有标签样本 (前5个):")
    print(f"X_L[:5]:\n{X_L[:5]}")
    print(f"y_L[:5]: {y_L[:5]}")
    print("\n无标签样本 (前5个):")
    print(f"X_U[:5]:\n{X_U[:5]}")
    print(f"y_U_orig[:5]: {y_U_orig[:5]}")

    return X_L, y_L, X_U, y_U_orig


import math


def find_rounds_math(n: int, k: float, r: int) -> int:
    """
    数学公式法求解所需轮数。
    :param n: 初始样本数（整数）
    :param k: 抽取比例，0 < k < 1（浮点数）
    :param r: 希望最终剩下的样本数
    :return: 最终所需轮数 p
    """
    p = 0
    while n > r:
        # 计算要删除的样本数，确保至少删除一个样本
        num_to_remove = max(1, int(n * k))
        n -= num_to_remove
        p += 1
    return p


def split_labeled_unlabeled(X, y, k):
    """
    切分数据集，将一定比例k的样本用于训练（有标签数据），
    其余作为无标签数据。

    参数:
    X -- 特征数据 (numpy array)
    y -- 标签数据 (numpy array)
    k -- 有标签数据的比例 (0 到 1)

    返回:
    X_L -- 有标签样本的特征 (numpy array)
    y_L -- 有标签样本的标签 (numpy array)
    X_U -- 无标签样本的特征 (numpy array)
    y_U -- 无标签样本的标签 (numpy array)
    """

    # 确保k在合理范围内
    if not (0 <= k <= 1):
        raise ValueError("k应该在0到1之间")

    # 获取数据的总样本数量
    total_samples = X.shape[0]
    labeled_samples_count = int(total_samples * k)

    # 打印初始数据的形状
    print(f"原始数据 X 形状: {X.shape}")
    print(f"原始标签 y 形状: {y.shape}")
    print(f"选择 {labeled_samples_count} 个样本作为有标签数据，剩余样本作为无标签数据")

    # 随机选择索引来进行切分
    indices = np.random.permutation(total_samples)
    labeled_indices = indices[:labeled_samples_count]
    unlabeled_indices = indices[labeled_samples_count:]

    # 切分数据和标签
    X_L = X[labeled_indices]
    y_L = y[labeled_indices]
    X_U = X[unlabeled_indices]
    y_U = y[unlabeled_indices]

    # 打印切分后的数据形状
    print(f"有标签数据 X_L 形状: {X_L.shape}")
    print(f"有标签数据 y_L 形状: {y_L.shape}")
    print(f"无标签数据 X_U 形状: {X_U.shape}")
    print(f"无标签数据 y_U 形状: {y_U.shape}")

    return X_L, y_L, X_U, y_U


def split_labeled_unlabeled_with_2_labels(X, y_C, y_R, k):
    """
    切分数据集，将一定比例k的样本用于训练（有标签数据），
    其余作为无标签数据。

    参数:
    X -- 特征数据 (numpy array)
    y_C -- 分类标签数据 (numpy array)
    y_R -- 回归标签数据 (numpy array)
    k -- 有标签数据的比例 (0 到 1)

    返回:
    X_L -- 有标签样本的特征 (numpy array)
    y_C_L -- 有标签样本的分类标签 (numpy array)
    y_R_L -- 有标签样本的回归标签 (numpy array)
    X_U -- 无标签样本的特征 (numpy array)
    y_C_U -- 无标签样本的分类标签 (numpy array)
    y_R_U -- 无标签样本的回归标签 (numpy array)
    """

    # 确保k在合理范围内
    if not (0 <= k <= 1):
        raise ValueError("k应该在0到1之间")

    # 获取数据的总样本数量
    total_samples = X.shape[0]
    if y_C.shape[0] != total_samples or y_R.shape[0] != total_samples:
        raise ValueError("X, y_C 和 y_R 的样本数量必须一致")

    # 计算有标签样本的数量
    labeled_samples_count = int(total_samples * k)

    # 打印初始数据的形状
    print(f"原始数据 X 形状: {X.shape}")
    print(f"原始分类标签 y_C 形状: {y_C.shape}")
    print(f"原始回归标签 y_R 形状: {y_R.shape}")
    print(f"选择 {labeled_samples_count} 个样本作为有标签数据，剩余样本作为无标签数据")

    # 随机选择索引来进行切分
    indices = np.random.permutation(total_samples)
    labeled_indices = indices[:labeled_samples_count]
    unlabeled_indices = indices[labeled_samples_count:]

    # 切分数据和标签
    X_L = X[labeled_indices]
    y_C_L = y_C[labeled_indices]
    y_R_L = y_R[labeled_indices]
    X_U = X[unlabeled_indices]
    y_C_U = y_C[unlabeled_indices]
    y_R_U = y_R[unlabeled_indices]

    # 打印切分后的数据形状
    print(f"有标签数据 X_L 形状: {X_L.shape}")
    print(f"有标签数据 y_C_L 形状: {y_C_L.shape}")
    print(f"有标签数据 y_R_L 形状: {y_R_L.shape}")
    print(f"无标签数据 X_U 形状: {X_U.shape}")
    print(f"无标签数据 y_C_U 形状: {y_C_U.shape}")
    print(f"无标签数据 y_R_U 形状: {y_R_U.shape}")

    return X_L, y_C_L, y_R_L, X_U, y_C_U, y_R_U


def split_labeled_unlabeled(*arrays, k, random_state=None):
    """
    切分数据集，将一定比例 k 的样本用于训练（有标签数据），
    其余作为无标签数据。

    参数:
    *arrays -- 任意数量的输入数组（numpy array），所有数组的长度必须相同。
    k -- 有标签数据的比例 (0 到 1)。
    random_state -- 随机种子，用于结果复现 (默认: None)。

    返回:
    labeled_arrays -- 切分后的有标签数据（列表）。
    unlabeled_arrays -- 切分后的无标签数据（列表）。
    """
    # 检查 k 是否在合理范围内
    if not (0 <= k <= 1):
        raise ValueError("k 应该在 0 到 1 之间")

    # 检查是否有输入数组
    if len(arrays) == 0:
        raise ValueError("至少需要一个输入数组")

    # 检查所有数组的长度是否一致
    array_lengths = [arr.shape[0] for arr in arrays]
    if len(set(array_lengths)) != 1:
        raise ValueError("所有输入数组的长度必须一致")

    # 获取样本总数
    total_samples = array_lengths[0]

    # 设置随机种子（如果提供）
    if random_state is not None:
        np.random.seed(random_state)

    # 计算有标签样本的数量
    labeled_samples_count = int(total_samples * k)

    # 打印初始数据的形状
    print(f"总样本数: {total_samples}")
    for i, arr in enumerate(arrays):
        print(f"输入数组 {i} 的形状: {arr.shape}")
    print(f"选择 {labeled_samples_count} 个样本作为有标签数据，剩余样本作为无标签数据")

    # 随机打乱索引
    indices = np.random.permutation(total_samples)
    labeled_indices = indices[:labeled_samples_count]
    unlabeled_indices = indices[labeled_samples_count:]

    # 切分数据
    labeled_arrays = [arr[labeled_indices] for arr in arrays]
    unlabeled_arrays = [arr[unlabeled_indices] for arr in arrays]

    # 打印切分后的数据形状
    for i, (labeled, unlabeled) in enumerate(zip(labeled_arrays, unlabeled_arrays)):
        print(f"输入数组 {i} 的有标签数据形状: {labeled.shape}")
        print(f"输入数组 {i} 的无标签数据形状: {unlabeled.shape}")

    return labeled_arrays, unlabeled_arrays


def vertical_split_array(X, random_state=None, first_col_rate=0.5):
    """
    垂直切分矩阵 X 为 XA 和 XB，XA 和 XB 行数相同，列数不同。

    参数:
        X: np.ndarray 或 pd.DataFrame
            输入数据，行数固定，列数将按比例分割。
        random_state: int 或 None
            随机种子，用于打乱列顺序（如果需要）。
        first_col_rate: float
            XA 的列数占总列数的比例，范围为 (0, 1)。

    返回:
        XA: np.ndarray 或 pd.DataFrame
            切分后的第一部分，列数为 X 列数的 first_col_rate。
        XB: np.ndarray 或 pd.DataFrame
            切分后的第二部分，列数为剩余部分。
    """
    # 检查输入参数
    if not (0 < first_col_rate < 1):
        raise ValueError("first_col_rate 必须在 (0, 1) 范围内")

    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X 必须是 numpy.ndarray 或 pandas.DataFrame 类型")

    # 获取列数
    total_cols = X.shape[1]

    # 计算 XA 的列数
    first_cols = int(total_cols * first_col_rate)

    # 设置随机种子（如果需要）
    rng = np.random.default_rng(random_state)

    # 随机打乱列索引
    col_indices = np.arange(total_cols)
    rng.shuffle(col_indices)

    # 切分列索引
    XA_indices = col_indices[:first_cols]
    XB_indices = col_indices[first_cols:]

    # 根据列索引切分 X
    XA = X[:, XA_indices]
    XB = X[:, XB_indices]

    print(f"切分后XA的形状为{XA.shape}")
    print(f"切分后XB的形状为{XB.shape}")

    return XA, XB


def convert_columns_to_int(df, integer_columns):
    """
    将指定的列转换为 int 类型。

    参数：
        df (pd.DataFrame): 输入的 DataFrame。
        integer_columns (list): 需要转换为 int 类型的列名列表。

    返回：
        pd.DataFrame: 转换后的 DataFrame。
    """
    # 创建 DataFrame 的副本，避免修改原始数据
    new_df = df.copy()

    # 遍历需要转换的列
    for col in integer_columns:
        # 检查列是否存在于 DataFrame 中
        if col in new_df.columns:
            # 将列转换为 int 类型，处理空值时填充为 0
            new_df[col] = new_df[col].fillna(0).astype(int)
        else:
            print(f"警告：列 '{col}' 不存在于 DataFrame 中，跳过。")

    return new_df


def shuffle_column_order(df, column_names):
    """
    随机打乱指定列的顺序，保持行顺序不变。

    参数：
        df (pd.DataFrame): 输入的 DataFrame。
        column_names (list): 需要随机打乱顺序的列名列表。

    返回：
        pd.DataFrame: 列顺序随机打乱后的 DataFrame。
    """
    # 获取需要打乱的列
    shuffled_columns = random.sample(column_names, len(column_names))

    # 构造新的列顺序：打乱的列 + 未指定的列（保持原顺序）
    new_column_order = shuffled_columns + [col for col in df.columns if col not in column_names]

    # 返回按新列顺序排列的 DataFrame
    return df[new_column_order]


def constru_row_miss_df(complete_df, miss_rate):
    """
    从 complete_df 的最后一行开始，计算缺失 miss_rate 的行数，生成 incomplete_df 和 missing_df。

    参数:
        complete_df (pd.DataFrame): 完整的数据框。
        miss_rate (float): 缺失比例，范围为 0 到 1。

    返回:
        tuple: (complete_df, incomplete_df, missing_df)
    """
    # 确保 miss_rate 合法
    if not (0 <= miss_rate <= 1):
        raise ValueError("miss_rate 必须在 0 和 1 之间")

    # 计算总行数和需要缺失的行数
    total_rows = len(complete_df)
    missing_rows = int(total_rows * miss_rate)

    # 生成 incomplete_df 和 missing_df
    if missing_rows > 0:
        incomplete_df = complete_df.iloc[:-missing_rows]
        missing_df = complete_df.iloc[-missing_rows:]
    else:
        incomplete_df = complete_df.copy()
        missing_df = pd.DataFrame(columns=complete_df.columns)  # 空的 DataFrame

    return complete_df, incomplete_df, missing_df


def get_discrete_columns(df):
    """
    获取 DataFrame 中的离散列（类别列）的列名。

    参数:
        df (pd.DataFrame): 输入的 DataFrame。

    返回:
        list: 离散列（类别列）的列名列表。
    """
    discrete_columns = []

    for column in df.columns:
        # 判断列是否是离散列
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            # 如果是字符串类型或分类类型，直接认为是离散列
            discrete_columns.append(column)
        elif df[column].dtype in ['int64', 'int32']:
            # 如果是数值类型，检查唯一值的数量是否远小于总行数
            unique_values = df[column].nunique()
            total_values = len(df[column])
            if unique_values / total_values < 0.05:  # 可调整阈值，比如 5%
                discrete_columns.append(column)

    return discrete_columns


def evaluate_imputed_data(original_data, imputed_data, plotFig=False):
    # 确保数据形状相同
    if original_data.shape != imputed_data.shape:
        raise ValueError("Original data and imputed data must have the same shape")

    # 计算 RMSE (Root Mean Squared Error)
    rmse = sqrt(mean_squared_error(original_data, imputed_data))

    # 计算 MSE (Mean Squared Error)
    mse = mean_squared_error(original_data, imputed_data)

    # 计算 MAE (Mean Absolute Error)
    mae = mean_absolute_error(original_data, imputed_data)

    # 计算 R² (Coefficient of Determination)
    r2 = r2_score(original_data, imputed_data)

    # 输出结果
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")

    if plotFig == True:
        # 画图比较原始数据和插补数据
        plt.figure(figsize=(10, 6))

        # 绘制原始数据
        plt.subplot(1, 2, 1)
        plt.imshow(original_data, cmap='viridis', aspect='auto')
        plt.title("Original Data")
        plt.colorbar()

        # 绘制插补后的数据
        plt.subplot(1, 2, 2)
        plt.imshow(imputed_data, cmap='viridis', aspect='auto')
        plt.title("Imputed Data")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    return rmse, mse, mae, r2


def stack_and_reset_index(incomplete_df, imputed_df):
    """
    将两个 DataFrame 纵向堆叠起来，并重新构造索引。

    参数:
        incomplete_df (pd.DataFrame): 第一个 DataFrame。
        imputed_df (pd.DataFrame): 第二个 DataFrame。

    返回:
        pd.DataFrame: 堆叠后的新 DataFrame，索引已重置。
    """
    # 确保列名和列数相同
    if list(incomplete_df.columns) != list(imputed_df.columns):
        raise ValueError("两个 DataFrame 的列名和顺序必须相同")

    # 纵向堆叠两个 DataFrame
    stacked_df = pd.concat([incomplete_df, imputed_df], axis=0)

    # 重置索引
    stacked_df.reset_index(drop=True, inplace=True)

    return stacked_df


def process_dataframes(df_A, construct_df_B, unlabeled_row_indices, predict_cols):
    # 获取 labeled_row_indices
    labeled_row_indices = [i for i in df_A.index if i not in unlabeled_row_indices]

    # 根据 labeled_row_indices 和 unlabeled_row_indices 划分 df_A
    df_A_L = df_A.loc[labeled_row_indices] if labeled_row_indices else None
    df_A_U = df_A.loc[unlabeled_row_indices] if unlabeled_row_indices else None

    # 根据 labeled_row_indices 和 unlabeled_row_indices 划分 construct_df_B
    construct_df_B_L = construct_df_B.loc[labeled_row_indices] if labeled_row_indices else None
    construct_df_B_U = construct_df_B.loc[unlabeled_row_indices] if unlabeled_row_indices else None

    # 获取 train_cols
    train_cols = [col for col in construct_df_B.columns if col not in predict_cols]

    # 根据 predict_cols 从 construct_df_B_L 和 construct_df_B_U 生成字典 y_L_dict 和 y_U_dict
    y_L_dict = {col: construct_df_B_L[col] for col in predict_cols} if construct_df_B_L is not None else {}
    y_U_dict = {col: construct_df_B_U[col] for col in predict_cols} if construct_df_B_U is not None else {}

    # 根据 train_cols 从 construct_df_B_L 和 construct_df_B_U 得到 construct_df_B_L_train 和 construct_df_B_U_gen
    construct_df_B_L_train = construct_df_B_L[train_cols] if construct_df_B_L is not None and train_cols else None
    construct_df_B_U_train = construct_df_B_U[train_cols] if construct_df_B_U is not None and train_cols else None

    # 打印日志信息
    print("=== Logs ===")
    print(f"df_A_L: {df_A_L.shape if df_A_L is not None else 'None'}")
    print(f"df_A_U: {df_A_U.shape if df_A_U is not None else 'None'}")
    print(f"construct_df_B_L: {construct_df_B_L.shape if construct_df_B_L is not None else 'None'}")
    print(f"construct_df_B_U: {construct_df_B_U.shape if construct_df_B_U is not None else 'None'}")
    print(f"train_cols: {train_cols}")
    print(f"y_L_dict keys: {list(y_L_dict.keys())}")
    print(f"y_U_dict keys: {list(y_U_dict.keys())}")
    print(f"construct_df_B_L_train: {construct_df_B_L_train.shape if construct_df_B_L_train is not None else 'None'}")
    print(f"construct_df_B_U_train: {construct_df_B_U_train.shape if construct_df_B_U_train is not None else 'None'}")
    print("=== End of Logs ===\n")

    # 返回结果
    return df_A_L, df_A_U, construct_df_B_L, construct_df_B_U, y_L_dict, y_U_dict, construct_df_B_L_train, construct_df_B_U_train


def get_unlabeled_row_indices(complete_df_A, incomplete_df_B):
    """
    获取 incomplete_df_B 中未对齐的部分行索引（即不在 complete_df_A 中的行索引）。

    参数:
        complete_df_A (pd.DataFrame): 完整的 DataFrame。
        incomplete_df_B (pd.DataFrame): 可能不完整的 DataFrame。

    返回:
        unlabeled_row_indices (list): incomplete_df_B 中未对齐的行索引列表。
    """
    # 获取 complete_df_A 和 incomplete_df_B 的行索引
    complete_indices = set(complete_df_A.index)
    incomplete_indices = set(incomplete_df_B.index)

    # 找到 incomplete_df_B 中未对齐的行索引
    unlabeled_row_indices = list(complete_indices - incomplete_indices)

    return unlabeled_row_indices


def update_dataframe_with_dict(df, dict_value):
    """
    更新 DataFrame 中的列值，使用 dict_value 中的值。

    参数:
        df (pd.DataFrame): 要更新的 DataFrame。
        dict_value (dict): 包含列名和对应值的字典，值是与 DataFrame 长度相同的 NumPy 数组。

    返回:
        pd.DataFrame: 更新后的 DataFrame。
    """
    # 遍历字典中的列名和值
    for col_name, value in dict_value.items():
        # 检查列名是否在 DataFrame 中
        if col_name in df.columns:
            # 检查值的长度是否与 DataFrame 的长度一致
            if len(value) == len(df):
                # 更新 DataFrame 中对应列的值
                df[col_name] = value
            else:
                raise ValueError(f"Length of value for column '{col_name}' does not match DataFrame length.")
        else:
            raise KeyError(f"Column '{col_name}' not found in DataFrame.")
    return df


def get_identical_columns(df1, df2):
    """
    比较两个 DataFrame 的列值，返回列值完全相同的列名列表。

    参数:
        df1 (pd.DataFrame): 第一个 DataFrame。
        df2 (pd.DataFrame): 第二个 DataFrame。

    返回:
        list: 列值完全相同的列名列表。
    """
    # 检查两个 DataFrame 的列数和列名是否一致
    if not df1.columns.equals(df2.columns):
        raise ValueError("The columns of the two DataFrames do not match.")

    # 初始化一个列表，用于存储列值完全相同的列名
    identical_columns = []

    # 遍历所有列名
    for col in df1.columns:
        # 将两列转换为 NumPy 数组
        col_array1 = df1[col].to_numpy()
        col_array2 = df2[col].to_numpy()

        # 使用 np.array_equal 比较两列是否完全相同
        if np.array_equal(col_array1, col_array2):
            identical_columns.append(col)

    return identical_columns


def get_identical_columns_indices(data1, data2):
    """
    比较两个二维数据（可以是 DataFrame 或 NumPy 数组）的列值，返回列值完全相同的列的索引列表。

    参数:
        data1 (pd.DataFrame 或 np.ndarray): 第一个二维数据。
        data2 (pd.DataFrame 或 np.ndarray): 第二个二维数据。

    返回:
        list: 列值完全相同的列的索引列表。
    """
    # 如果输入是 DataFrame，则转换为 NumPy 数组
    if isinstance(data1, pd.DataFrame):
        data1 = data1.values
    if isinstance(data2, pd.DataFrame):
        data2 = data2.values

    # 检查两个数组的形状是否一致
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("The number of columns in the two inputs must be the same.")
    if data1.shape[0] != data2.shape[0]:
        raise ValueError("The number of rows in the two inputs must be the same.")

    # 初始化一个列表，用于存储列值完全相同的列索引
    identical_columns_indices = []

    # 遍历所有列索引
    for col_idx in range(data1.shape[1]):
        # 比较两列的值是否完全相同
        if np.array_equal(data1[:, col_idx], data2[:, col_idx]):
            identical_columns_indices.append(col_idx)

    return identical_columns_indices


def compare_dataframe_columns(df1, df2):
    """
    比较两个 DataFrame 的列值，返回列值完全相同的列名列表和列值不同的列名列表。

    参数:
        df1 (pd.DataFrame): 第一个 DataFrame。
        df2 (pd.DataFrame): 第二个 DataFrame。

    返回:
        tuple: (identical_columns, different_columns)
            - identical_columns (list): 列值完全相同的列名列表。
            - different_columns (list): 列值不同的列名列表。
    """
    # 检查两个 DataFrame 的列数和列名是否一致
    if not df1.columns.equals(df2.columns):
        raise ValueError("The columns of the two DataFrames do not match.")

    # 初始化两个列表，用于存储列值完全相同和不同的列名
    identical_columns = []
    different_columns = []

    # 遍历所有列名
    for col in df1.columns:
        # 将两列转换为 NumPy 数组
        col_array1 = df1[col].to_numpy()
        col_array2 = df2[col].to_numpy()

        # 使用 np.array_equal 比较两列是否完全相同
        if np.array_equal(col_array1, col_array2):
            identical_columns.append(col)
        else:
            different_columns.append(col)

    return identical_columns, different_columns


def transform_and_save_df(df, split_cols,file_path, name_prefix, name_suffix):
    data, y = df.iloc[:, :-1], df.iloc[:, [-1]]
    df_A, df_B = vertical_split_by_split_cols(data, split_cols=split_cols)
    y.to_csv(os.path.join(file_path,name_prefix+"y"+name_suffix),index=None)
    df_A.to_csv(os.path.join(file_path,name_prefix+"df_A"+name_suffix),index=None)
    df_B.to_csv(os.path.join(file_path,name_prefix+"df_B"+name_suffix),index=None)

    return df_A, df_B, y


def get_column_types(df):
    """
    该函数用于将输入DataFrame中的列划分为分类列和数值列。

    分类列的定义为：
    1. 数据类型为int。
    2. 列中仅包含0和1两个值。

    其余列则被归类为数值列。

    :param df: 输入的pandas DataFrame对象。
    :return: 一个包含两个元素的元组，第一个元素是分类列的列表 (category_columns)，
             第二个元素是数值列的列表 (numerical_columns)。
    """
    category_columns = []
    numerical_columns = []
    for col in df.columns:
        if df[col].dtype == 'int':
            unique_values = df[col].unique()
            if set(unique_values).issubset({0, 1}) and len(unique_values) <= 2:
                category_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            numerical_columns.append(col)
    print(f'len(category_columns) = {len(category_columns)}, len(numerical_columns) = {len(numerical_columns)}')
    return category_columns, numerical_columns