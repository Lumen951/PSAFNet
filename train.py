import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=100, save_path='best_model.pth', show=True):
    """
    训练PSAFNet模型

    该函数实现完整的训练循环，包括前向传播、损失计算、反向传播和模型保存。
    使用组合损失函数：交叉熵损失 + MMD对齐损失。

    论文参考：第4.2节训练细节 - "使用Adam优化器，学习率0.001，组合损失函数包括
    分类损失和MMD对齐损失，权重系数alpha=1.5"

    参数:
        model: PSAFNet模型
        device: 计算设备（CPU或CUDA）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数（通常为CrossEntropyLoss）
        optimizer: 优化器（通常为Adam）
        num_epochs: 训练轮数，默认100
        save_path: 模型保存路径，默认'best_model.pth'
        show: 是否显示训练进度，默认True

    返回:
        无（模型权重保存到文件）
    """
    best_val_loss = float('inf')  # 记录最佳验证损失
    with tqdm(total=num_epochs, desc='Training Progress', unit='epoch') as pbar:
        for epoch in range(num_epochs):
            model.train()  # 设置为训练模式
            running_loss = 0.0
            alpha = 1.5  # MMD损失的权重系数（论文第4.2节）

            # 训练一个epoch
            for inputs, labels in train_loader:
                inputs = inputs.to(device)  # [batch, channels, timepoints]
                labels = labels.to(device)  # [batch]
                inputs = inputs.unsqueeze(1)  # 添加特征通道维度：[batch, 1, channels, timepoints]

                # 前向传播
                optimizer.zero_grad()
                outputs, similarity_loss = model(inputs)  # outputs: [batch, num_classes], similarity_loss: 标量

                # 计算组合损失
                CE_loss = criterion(outputs, labels)  # 交叉熵损失
                loss = CE_loss + alpha * similarity_loss  # 总损失 = 分类损失 + alpha * MMD损失

                # 反向传播和优化
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # 验证
            val_loss, val_accuracy = validate(model, device, val_loader, criterion)

            # 更新进度条
            if show:
                pbar.set_postfix({
                    'Train Loss': running_loss / len(train_loader),
                    'Val Loss': val_loss,
                    'Val Accuracy': val_accuracy
                })
            pbar.update(1)

            # 保存最佳模型（基于验证损失）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f'Saving model with val_loss {val_loss:.4f} and val_accuracy {val_accuracy} at epoch {epoch + 1}')

def validate(model, device, val_loader, criterion):
    """
    验证模型性能

    在验证集上评估模型，计算验证损失和准确率。不进行梯度计算以节省内存。

    参数:
        model: PSAFNet模型
        device: 计算设备（CPU或CUDA）
        val_loader: 验证数据加载器
        criterion: 损失函数

    返回:
        val_loss: 平均验证损失
        val_accuracy: 验证准确率
    """
    model.eval()  # 设置为评估模式（关闭dropout等）
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():  # 不计算梯度
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)  # 添加特征通道维度

            # 前向传播
            outputs, _ = model(inputs)  # 忽略MMD损失
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 获取预测结果
            _, preds = torch.max(outputs, 1)  # 取最大概率的类别
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算准确率
    val_accuracy = accuracy_score(all_labels, all_preds)
    return val_loss / len(val_loader), val_accuracy

def test(model, device, test_loader, load_path='best_model.pth'):
    """
    测试模型性能并计算详细指标

    加载最佳模型权重，在测试集上评估性能，计算多个评估指标。

    论文参考：第4.3节评估指标 - "使用准确率、命中率（TPR）和虚警率（FPR）
    作为主要评估指标，这些指标在目标检测任务中具有重要意义"

    评估指标说明：
    - Accuracy: 总体准确率 = (TP + TN) / (TP + TN + FP + FN)
    - Hit Rate (TPR): 真正例率 = TP / (TP + FN)，衡量检测到目标的能力
    - False Alarm Rate (FPR): 假正例率 = FP / (FP + TN)，衡量误报率

    参数:
        model: PSAFNet模型
        device: 计算设备（CPU或CUDA）
        test_loader: 测试数据加载器
        load_path: 模型权重文件路径，默认'best_model.pth'

    返回:
        accuracy: 测试准确率
        hit_rate: 命中率（真正例率）
        false_alarm_rate: 虚警率（假正例率）
    """
    # 加载最佳模型权重
    model.load_state_dict(torch.load(load_path))
    model.eval()  # 设置为评估模式

    all_preds = []
    all_labels = []
    prediction_times = []  # 记录推理时间

    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)  # 添加特征通道维度

            # 测量推理时间
            start_time = time.time()
            outputs, _ = model(inputs)
            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)

            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 计算混淆矩阵并提取各项指标
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    hit_rate = tp / (tp + fn) if (tp + fn) != 0 else 0  # 真正例率（召回率）
    false_alarm_rate = fp / (fp + tn) if (fp + tn) != 0 else 0  # 假正例率

    # 打印测试结果
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Hit Rate (TPR): {hit_rate:.4f}')
    print(f'False Alarm Rate (FPR): {false_alarm_rate:.4f}')
    print(f'Average Prediction Time per Sample: {np.mean(prediction_times):.6f} seconds')

    return accuracy, hit_rate, false_alarm_rate
