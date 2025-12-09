import numpy as np
import torch
from scipy.io import loadmat
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset, random_split

def extract_segments_eeg(eeg_data, window_length=1, fs=200):
    """
    从EEG数据中提取目标和非目标时间段

    该函数从预处理的EEG数据中提取两个关键时间窗口：
    1. 非目标窗口：视频播放开始后1-2秒（基线期）
    2. 目标窗口：视频播放开始后5-6秒（目标出现期）

    论文参考：第4.1节数据描述 - "EEG信号在视频播放期间记录，目标通常在5秒左右出现"

    参数:
        eeg_data: EEG数据数组，形状为 [channels, timepoints, trials]
        window_length: 提取窗口的长度（秒），默认1秒
        fs: 采样频率（Hz），默认200Hz

    返回:
        X: 特征张量，形状为 [samples, channels, timepoints]
        y: 标签张量，形状为 [samples]，0表示非目标，1表示目标
    """
    # 计算时间窗口的起止索引
    no_target_start = int(1 * fs)  # 非目标窗口起始：1秒
    no_target_end = int((1 + window_length) * fs)  # 非目标窗口结束：2秒
    target_start = int(5 * fs)  # 目标窗口起始：5秒
    target_end = int((5 + window_length) * fs)  # 目标窗口结束：6秒

    # 提取时间段
    no_target_data = eeg_data[:, no_target_start:no_target_end, :]  # [channels, timepoints, trials]
    target_data = eeg_data[:, target_start:target_end, :]

    # 重塑数据：将trials维度移到第一维
    no_target_data_reshaped = no_target_data.reshape(no_target_data.shape[2], -1)  # [trials, channels*timepoints]
    target_data_reshaped = target_data.reshape(target_data.shape[2], -1)

    # 创建标签
    no_target_labels = np.zeros(no_target_data_reshaped.shape[0])  # 非目标标签：0
    target_labels = np.ones(target_data_reshaped.shape[0])  # 目标标签：1

    # 合并数据和标签
    X = np.concatenate([no_target_data_reshaped, target_data_reshaped], axis=0)
    y = np.concatenate([no_target_labels, target_labels], axis=0)

    # 重塑为正确的形状：[samples, channels, timepoints]
    X = X.reshape(X.shape[0], eeg_data.shape[0], no_target_data.shape[1])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 打乱数据顺序
    X, y = shuffle(X, y, random_state=42)
    return X, y

def extract_segments_path(file_path, window_length=1, fs=200, init_time=1.0, target_time=5.0):
    """
    从MATLAB文件路径加载并提取EEG数据段

    该函数直接从.mat文件加载EEG数据，然后提取指定时间窗口的数据。
    与extract_segments_eeg不同，该函数支持自定义时间窗口位置。

    论文参考：第4.1节 - "数据以MATLAB格式存储，包含EEG结构体"

    参数:
        file_path: MATLAB文件路径（.mat格式）
        window_length: 提取窗口的长度（秒），默认1秒
        fs: 采样频率（Hz），默认200Hz
        init_time: 非目标窗口起始时间（秒），默认1.0秒
        target_time: 目标窗口起始时间（秒），默认5.0秒

    返回:
        X: 特征张量，形状为 [samples, channels, timepoints]
        y: 标签张量，形状为 [samples]，0表示非目标，1表示目标
    """
    # 加载MATLAB文件
    data = loadmat(file_path)
    eeg_data = data['EEG']['data'][0][0]  # 提取EEG数据：[channels, timepoints, trials]

    # 计算时间窗口的起止索引
    no_target_start = int(init_time * fs)
    no_target_end = int((init_time + window_length) * fs)
    target_start = int(target_time * fs)
    target_end = int((target_time + window_length) * fs)

    # 提取时间段
    no_target_data = eeg_data[:, no_target_start:no_target_end, :]
    target_data = eeg_data[:, target_start:target_end, :]

    # 转置数据：[channels, timepoints, trials] -> [trials, channels, timepoints]
    no_target_data = np.transpose(no_target_data, (2, 0, 1))
    target_data = np.transpose(target_data, (2, 0, 1))

    # 重塑为2D数组以便拼接
    no_target_data_reshaped = no_target_data.reshape(no_target_data.shape[0], -1)
    target_data_reshaped = target_data.reshape(target_data.shape[0], -1)

    # 创建标签
    no_target_labels = np.zeros(no_target_data_reshaped.shape[0])
    target_labels = np.ones(target_data_reshaped.shape[0])

    # 合并数据和标签
    X = np.concatenate([no_target_data_reshaped, target_data_reshaped], axis=0)
    y = np.concatenate([no_target_labels, target_labels], axis=0)

    # 重塑为正确的形状：[samples, channels, timepoints]
    X = X.reshape(X.shape[0], eeg_data.shape[0], no_target_data.shape[2])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

def create_dataloaders(X, y, batch_size=16, train_ratio=0.8, val_ratio=0.1):
    """
    创建训练、验证和测试数据加载器（单被试内划分）

    将单个被试的数据随机划分为训练集、验证集和测试集。

    参数:
        X: 特征张量 [samples, channels, timepoints]
        y: 标签张量 [samples]
        batch_size: 批次大小，默认16
        train_ratio: 训练集比例，默认0.8（80%）
        val_ratio: 验证集比例，默认0.1（10%）

    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器（剩余10%）
    """
    dataset = TensorDataset(X, y)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_dataloaders_folds(X_train, y_train, X_val, y_val, batch_size=16):
    """
    创建数据加载器（用于预划分的折叠数据）

    当数据已经预先划分为训练集和验证集时使用此函数。

    参数:
        X_train: 训练特征张量
        y_train: 训练标签张量
        X_val: 验证特征张量
        y_val: 验证标签张量
        batch_size: 批次大小，默认16

    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器（与val_loader相同）
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_cross_subject_dataloaders(file_paths, window_length=1, batch_size=16, test_index=7):
    """
    创建跨被试数据加载器（留一被试交叉验证）

    该函数实现留一被试交叉验证（Leave-One-Subject-Out Cross-Validation, LOSO-CV）：
    - 一个被试作为测试集
    - 一个被试作为验证集
    - 其余被试作为训练集

    论文参考：第4.2节 - "使用留一被试交叉验证评估模型的泛化能力，
    这是脑机接口研究中的标准评估方法"

    参数:
        file_paths: MATLAB文件路径列表，每个文件对应一个被试
        window_length: 提取窗口的长度（秒），默认1秒
        batch_size: 批次大小，默认16
        test_index: 测试被试的索引，默认7

    返回:
        train_loader: 训练数据加载器（多个被试的数据合并）
        val_loader: 验证数据加载器（单个被试）
        test_loader: 测试数据加载器（单个被试）
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # 选择验证集索引：如果测试集不是最后一个被试，则验证集为最后一个被试；否则为倒数第二个
    val_index = len(file_paths) - 1 if test_index != len(file_paths) - 1 else len(file_paths) - 2

    # 遍历所有被试文件
    for i, file_path in enumerate(file_paths):
        X, y = extract_segments_path(file_path, window_length=window_length, init_time=1, target_time=5.0)

        if i == test_index:
            # 测试集：当前被试
            X_test, y_test = X, y
        elif i == val_index:
            # 验证集：选定的验证被试
            X_val, y_val = X, y
        else:
            # 训练集：其余所有被试
            X_train.append(X)
            y_train.append(y)

    # 合并所有训练被试的数据
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    # 创建数据集
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
