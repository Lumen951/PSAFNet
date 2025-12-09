class Config():
    """
    PSAFNet配置类 - 集中管理所有超参数

    该类包含训练参数、模型架构参数和数据参数。所有参数都在此处定义，
    便于实验管理和参数调优。

    论文参考：第4节实验设置 - 详细说明了各参数的选择依据
    """
    def __init__(self):

        # ==================== 训练参数 ====================
        self.seed = 42  # 随机种子，用于结果可复现性
        self.num_channels = 59  # EEG通道数（导联数）
        self.num_timepoints = 200  # 每个样本的时间点数（1秒 @ 200Hz采样率）
        self.num_epochs = 5  # 训练轮数
        self.learning_rate = 0.001  # Adam优化器的学习率
        self.batchsize = 16  # 批次大小
        self.fs = 200  # 采样频率（Hz）
        self.num_class = 2  # 分类类别数（0: 无目标, 1: 有目标）

        # ==================== PSAFNet架构参数 ====================
        # 相位分割参数
        self.stage_timepoints = 150  # 每个相位的时间点数（0.75秒 @ 200Hz）
                                     # 论文第3.1节：两个相位重叠50个时间点（0.25秒）

        # 相位编码器参数
        self.init_conv_layers = 12  # 初始卷积层的输出通道数
                                    # 论文第3.2节：多尺度卷积后拼接得到12个特征图
        self.conv_depth = 2  # 深度卷积的深度乘数
                            # 最终特征通道数 = init_conv_layers * conv_depth = 24

        # 注意力机制参数
        self.SE_spatial_size = 2  # 空间SE层的瓶颈维度
                                  # 论文第3.2.1节：用于EEG通道注意力
        self.SE_channels_size = 1  # 通道SE层的瓶颈维度
                                   # 论文第3.2.1节：用于特征通道注意力

        # 归一化和正则化参数
        self.GN_groups = 3  # GroupNorm的组数（12个通道分为3组，每组4个通道）
        self.dropout_rate = 0.2  # Dropout概率，防止过拟合

        # 动态融合模块参数
        self.dilation_expand = 2  # TCN膨胀率的扩展因子
                                  # 论文第3.3节：膨胀率按2的幂次增长（1, 2, 4, ...）
        self.mmd_sigma = 1.0  # MMD损失的高斯核带宽参数
                              # 论文第3.3节：控制特征分布对齐的敏感度
        self.TCN_hidden_dim = 24  # TCN隐藏层维度
                                  # 论文第3.3节：与编码器输出维度一致


        # 打印所有参数（用于实验记录）
        self.print_config()

    def print_config(self):
        """打印所有配置参数，便于实验记录和调试"""
        print("Configuration parameters:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")



# 创建全局配置对象，供所有模块导入使用
config = Config()