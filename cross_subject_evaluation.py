"""
PSAFNet跨被试评估脚本

该脚本实现留一被试交叉验证（Leave-One-Subject-Out Cross-Validation, LOSO-CV），
这是脑机接口研究中评估模型泛化能力的标准方法。

论文参考：第4.2节 - "使用8折留一被试交叉验证评估PSAFNet在不同被试间的泛化性能，
这种评估方式能够真实反映模型在新用户上的表现"

实验流程：
1. 对8个被试进行8折交叉验证
2. 每折中：1个被试作测试集，1个被试作验证集，其余6个被试作训练集
3. 记录每个被试的准确率、命中率和虚警率
4. 计算所有被试的平均性能和标准差

作者：Wonder-How
邮箱：wonderhow@bit.edu.cn
论文：Brain-inspired deep learning model for EEG-based low-quality video target detection
      with phased encoding and aligned fusion (ESWA, 2025)
"""

import numpy as np
import torch
import os
from torchinfo import summary
from data_processing import create_cross_subject_dataloaders
from train import train, test
from utils import set_seed
from PSAFNet import PSAFNet
from my_config import config

# ==================== 环境设置 ====================
# 启用CUDA启动阻塞以便调试（使CUDA操作同步执行）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 设置随机种子以确保结果可复现
set_seed(config.seed)

# ==================== 数据路径配置 ====================
# 数据类型：'unica'表示经过预处理的EEG数据
data_type = "unica"

# 8个被试的数据文件路径
# 论文第4.1节：实验数据来自8名健康被试，每人完成多次试验
file_paths = [
    fr"D:\machine learning\EEG_video\{data_type}_data\cw_1.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\ghr_2.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\gr_3.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\kx_4.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\pbl_5.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\sjt_6.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\wxc_7.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\xxb_8.mat"
]

# ==================== 模型初始化 ====================
# 选择计算设备（优先使用GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化PSAFNet模型
model = PSAFNet(
    stage_timepoints=config.stage_timepoints,  # 每个相位的时间点数（150）
    lead=config.num_channels,  # EEG通道数（59）
    time=config.num_timepoints  # 总时间点数（200）
).to(device)

# 显示模型结构摘要（参数量、层结构等）
summary(model)

# ==================== 训练配置 ====================
# 定义损失函数：交叉熵损失（用于二分类）
criterion = torch.nn.CrossEntropyLoss()

# 定义优化器：Adam优化器
# 论文第4.2节：使用Adam优化器，学习率0.001
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# ==================== 交叉验证循环 ====================
# 初始化结果存储变量
fold_results = []
k_folds = 8  # 折数（被试数）
fold_accuracies = []  # 每折的准确率
fold_hit_rates = []  # 每折的命中率
fold_false_alarm_rates = []  # 每折的虚警率

# 对每个被试进行测试（留一被试交叉验证）
for i in range(k_folds):
    print(f"Testing on Subject {i + 1}")

    # 创建跨被试数据加载器
    # 当前被试i作为测试集，另一个被试作为验证集，其余被试作为训练集
    train_loader, val_loader, test_loader = create_cross_subject_dataloaders(
        file_paths,
        window_length=(config.num_timepoints / config.fs),  # 窗口长度：1秒
        batch_size=config.batchsize,
        test_index=i  # 当前测试被试的索引
    )

    # 训练模型
    # 注意：每折都会重新训练模型（不使用前一折的权重）
    train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=config.num_epochs)

    # 测试模型
    test_accuracy, hit_rate, false_alarm_rate = test(model, device, test_loader)

    # 转换为百分比并存储结果
    test_accuracy *= 100
    fold_accuracies.append(test_accuracy)
    fold_hit_rates.append(hit_rate * 100)
    fold_false_alarm_rates.append(false_alarm_rate * 100)

    # 打印当前被试的结果
    print(
        f'Subject {i + 1}: Accuracy = {test_accuracy:.6f}%, Hit Rate = {hit_rate * 100:.6f}%, False Alarm Rate = {false_alarm_rate * 100:.6f}%')

# ==================== 结果汇总 ====================
# 打印每个被试的详细结果
print('\nIndividual Subject Results:')
for i in range(k_folds):
    print(
        f'Subject {i + 1}: Accuracy = {fold_accuracies[i]:.6f}%, Hit Rate = {fold_hit_rates[i]:.6f}%, False Alarm Rate = {fold_false_alarm_rates[i]:.6f}%')

# 计算所有被试的平均性能和标准差
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
mean_hit_rate = np.mean(fold_hit_rates)
std_hit_rate = np.std(fold_hit_rates)
mean_false_alarm_rate = np.mean(fold_false_alarm_rates)
std_false_alarm_rate = np.std(fold_false_alarm_rates)

# 打印总体结果（均值 ± 标准差）
# 论文表2：跨被试评估结果
print('\nOverall Results (Mean ± Std):')
print(f'Accuracy: {mean_accuracy:.6f}% ± {std_accuracy:.6f}%')
print(f'Hit Rate: {mean_hit_rate:.6f}% ± {std_hit_rate:.6f}%')
print(f'False Alarm Rate: {mean_false_alarm_rate:.6f}% ± {std_false_alarm_rate:.6f}%')