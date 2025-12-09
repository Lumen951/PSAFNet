import torch.nn.functional as F
import torch
from torch import nn
from my_config import config

class SELayer(nn.Module):
    """
    空间压缩-激励层（Spatial Squeeze-and-Excitation Layer），用于EEG通道注意力机制

    该模块实现了跨EEG通道（导联）的空间注意力，受SE-Net架构启发。通过显式建模通道间的
    相互依赖关系来重新校准通道特征响应。

    论文参考：第3.2.1节 - "空间注意力用于强调重要的EEG通道，同时抑制不太相关的通道"

    参数:
        leads (int): EEG通道/导联数量（默认：59）
        core_size (int): 用于降维的瓶颈维度（默认：2）
    """
    def __init__(self, leads=59, core_size=2):
        super(SELayer, self).__init__()
        # 全局平均池化，用于压缩空间信息
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 瓶颈层：降维以高效捕获通道依赖关系
        self.fc1 = nn.Linear(leads, core_size, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 扩展回原始通道维度
        self.fc2 = nn.Linear(core_size, leads, bias=False)
        # Sigmoid激活函数，生成[0,1]范围的注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        空间注意力的前向传播

        参数:
            x: 输入张量，形状为 [batch, channels, leads, timepoints]

        返回:
            应用空间注意力后的重新校准张量
        """
        b, c, leads, times = x.size()
        # 转置为 [batch, leads, channels, timepoints] 以便在空间维度上池化
        y = self.avg_pool(x.transpose(1, 2))  # [batch, leads, 1, 1]
        y = y.view(b, leads)  # 展平为 [batch, leads]
        # 压缩：降维到瓶颈维度
        y = self.fc1(y)  # [batch, core_size]
        y = self.relu(y)
        # 激励：扩展回通道维度
        y = self.fc2(y)  # [batch, leads]
        # 生成注意力权重
        y = self.sigmoid(y)  # [batch, leads]
        # 重塑并转置回匹配输入维度
        y = y.view(b, leads, 1, 1)  # [batch, leads, 1, 1]
        y = y.transpose(1, 2)  # [batch, 1, leads, 1] -> [batch, channels, leads, 1]
        # 通过逐元素乘法应用注意力权重
        return x * y


class SE_channels_Block(nn.Module):
    """
    特征通道压缩-激励块（Channel-wise SE Block），带残差连接

    该模块对特征通道（非空间通道）应用注意力机制，以自适应地重新校准特征图响应。
    包含残差连接以保留原始信息。

    论文参考：第3.2.1节 - "在深度卷积后使用通道注意力来自适应地重新校准特征通道"

    参数:
        feature_channel (int): 特征通道数（例如 init_conv_layers * conv_depth）
        core_num (int): 用于压缩的瓶颈维度（默认：1）
    """
    def __init__(self, feature_channel, core_num=1):
        super(SE_channels_Block, self).__init__()
        self.feature_channel = feature_channel
        # 全局平均池化，聚合空间信息
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 瓶颈层，用于通道注意力
        self.fc1 = nn.Linear(feature_channel, core_num)
        self.fc2 = nn.Linear(core_num, feature_channel)
        # Sigmoid激活函数，生成注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        带通道注意力和残差连接的前向传播

        参数:
            x: 输入张量，形状为 [batch, feature_channel, time_point, channels]

        返回:
            应用通道注意力并添加残差连接后的输出
        """
        batch_size, feature_channel, time_point, channels = x.size()
        # 重塑并池化：独立处理每个特征通道
        squeezed = self.global_avg_pool(x.view(batch_size * feature_channel, 1, time_point, channels))
        squeezed = squeezed.view(batch_size, feature_channel)  # [batch, feature_channel]
        # 压缩-激励：压缩和扩展
        excitation = self.fc1(squeezed)  # [batch, core_num]
        excitation = self.fc2(excitation)  # [batch, feature_channel]
        # 生成注意力权重并重塑
        excitation = self.sigmoid(excitation).view(batch_size, feature_channel, 1, 1)
        # 应用注意力权重
        scale = x * excitation
        # 残差连接：保留原始特征
        return scale + x


class Phased_Encoder(nn.Module):
    """
    相位编码器（Phased Encoder）- PSAFNet的核心空间特征提取模块

    该编码器模拟人脑视觉系统的多尺度时间感受野，通过并行的多尺度卷积核捕获不同时间
    尺度的EEG特征。结合深度可分离卷积和双重注意力机制（空间+通道）进行高效特征提取。

    论文参考：第3.2节 - "相位编码器使用多尺度时间卷积（32、64、96）来模拟大脑中
    不同神经元的时间感受野差异，然后通过深度可分离卷积提取空间-时间特征"

    参数:
        num_channels (int): EEG通道数（空间维度）
        o1 (int): 初始卷积层输出通道数（默认从config读取）
        d (int): 深度卷积的深度乘数（默认从config读取）
    """
    def __init__(self, num_channels, o1=config.init_conv_layers, d=config.conv_depth):
        super(Phased_Encoder, self).__init__()
        # 第一阶段：多尺度时间卷积（模拟不同时间感受野）
        # 论文图3：三个并行的时间卷积分支，捕获短、中、长时间依赖
        self.conv1_1 = nn.Conv2d(1, o1 // 3, (1, 32), padding=(0, 16), bias=False)  # 短时间尺度：32个时间点
        self.conv1_2 = nn.Conv2d(1, o1 // 3, (1, 64), padding=(0, 32), bias=False)  # 中时间尺度：64个时间点
        self.conv1_3 = nn.Conv2d(1, o1 // 3, (1, 96), padding=(0, 48), bias=False)  # 长时间尺度：96个时间点

        # 空间注意力：强调重要的EEG通道
        self.se1 = SELayer(core_size=config.SE_spatial_size)
        self.batchnorm1 = nn.GroupNorm(num_groups=config.GN_groups, num_channels=o1)

        # 第二阶段：深度卷积（Depthwise Convolution）
        # 论文第3.2.1节：跨所有EEG通道的深度卷积，提取空间特征
        self.depthwiseConv = nn.Conv2d(o1, o1 * d, (num_channels, 1), groups=o1, bias=False)

        # 通道注意力：自适应重新校准特征通道
        self.sec = SE_channels_Block(o1 * d, core_num=config.SE_channels_size)
        self.batchnorm2 = nn.GroupNorm(num_groups=config.GN_groups, num_channels=o1 * d)
        self.elu = nn.GELU()  # GELU激活函数，平滑的非线性变换
        self.dropout1 = nn.Dropout(config.dropout_rate)

        # 第三阶段：可分离卷积（Separable Convolution）进行时间下采样
        # 论文第3.2.1节：两个连续的深度卷积 + 逐点卷积，降低时间分辨率
        self.depthwiseConv2_1 = nn.Conv2d(o1 * d, o1 * d, (1, 32), padding=(0, 16), stride=(1, 2), groups=o1 * d, bias=False)
        self.depthwiseConv2_2 = nn.Conv2d(o1 * d, o1 * d, (1, 32), padding=(0, 16), stride=(1, 2), groups=o1 * d, bias=False)
        self.pointwiseConv2 = nn.Conv2d(o1 * d, o1 * d, (1, 1), bias=False)  # 1x1卷积，混合通道信息
        self.batchnorm3 = nn.GroupNorm(num_groups=config.GN_groups, num_channels=o1 * d)
        self.elu2 = nn.GELU()
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        """
        相位编码器的前向传播

        参数:
            x: 输入EEG张量，形状为 [batch, 1, channels, timepoints]

        返回:
            提取的空间-时间特征，形状为 [batch, o1*d, 1, timepoints//4]
        """
        # 阶段1：多尺度时间特征提取
        x1 = self.conv1_1(x)  # 短时间尺度特征
        x2 = self.conv1_2(x)  # 中时间尺度特征
        x3 = self.conv1_3(x)  # 长时间尺度特征
        x = torch.cat((x1, x2, x3), dim=1)  # 拼接多尺度特征

        # 应用空间注意力和归一化
        x = self.se1(x)
        x = self.batchnorm1(x)

        # 阶段2：深度卷积提取空间特征
        x = self.depthwiseConv(x)  # 跨所有EEG通道
        x = self.sec(x)  # 通道注意力
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.dropout1(x)

        # 阶段3：可分离卷积进行时间下采样（降低4倍）
        x = self.depthwiseConv2_1(x)  # 第一次下采样（stride=2）
        x = self.depthwiseConv2_2(x)  # 第二次下采样（stride=2）
        x = self.pointwiseConv2(x)  # 逐点卷积混合通道
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        return x


class TCNLayer(nn.Module):
    """
    时间卷积网络层（Temporal Convolutional Network Layer）

    TCN使用因果卷积（causal convolution）和膨胀卷积（dilated convolution）来捕获长期
    时间依赖关系，同时保持时间顺序。包含残差连接以促进梯度流动。

    论文参考：第3.3节 - "TCN通过膨胀因果卷积捕获长期时间依赖，膨胀率指数增长以
    扩大感受野"

    参数:
        input_dim (int): 输入特征维度
        output_dim (int): 输出特征维度
        kernel_size (int): 卷积核大小
        dilation (int): 膨胀率（控制感受野大小）
    """
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super(TCNLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # 膨胀因果卷积：只看过去的时间点，不看未来
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, dilation=dilation, padding=0)
        # 残差连接：1x1卷积用于维度匹配
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        """
        TCN层的前向传播

        参数:
            x: 输入张量，形状为 [batch, input_dim, time]

        返回:
            输出张量，形状为 [batch, output_dim, time]
        """
        # 因果填充：只在左侧（过去）填充，保证不使用未来信息
        padding = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (padding, 0), mode='constant', value=0)
        # 膨胀卷积
        out = self.conv(x_padded)
        out = F.relu(out)
        # 残差连接
        residual = self.residual(x)
        out = out + residual
        return out


class CrossAttention(torch.nn.Module):
    """
    交叉注意力模块（Cross-Attention Module）

    实现两个特征序列之间的交叉注意力机制，用于对齐和融合来自不同时间相位的特征。
    使用标准的Query-Key-Value注意力机制。

    论文参考：第3.3节 - "交叉注意力用于对齐两个相位的特征，通过计算一个相位对另一个
    相位的注意力权重，实现特征的自适应融合"

    参数:
        input_dim (int): 输入特征维度
    """
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        # Query、Key、Value的线性变换矩阵
        self.W_Q = torch.nn.Linear(input_dim, input_dim)  # Query投影
        self.W_K = torch.nn.Linear(input_dim, input_dim)  # Key投影
        self.W_V = torch.nn.Linear(input_dim, input_dim)  # Value投影

    def forward(self, A, B):
        """
        交叉注意力的前向传播

        参数:
            A: 查询序列（Query），形状为 [batch, time, features]
            B: 键值序列（Key-Value），形状为 [batch, time, features]

        返回:
            注意力输出，A对B的加权聚合，形状为 [batch, time, features]
        """
        # 生成Query、Key、Value
        Q_A = self.W_Q(A)  # A作为Query
        K_B = self.W_K(B)  # B作为Key
        V_B = self.W_V(B)  # B作为Value

        # 计算注意力分数：Q * K^T
        attention_scores = torch.matmul(Q_A, K_B.transpose(-2, -1))

        # 缩放注意力分数（scaled dot-product attention）
        d_k = Q_A.size(-1)
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float))

        # Softmax归一化得到注意力权重
        attention_weights = F.softmax(scaled_attention_scores, dim=-1)

        # 加权聚合Value
        attention_output = torch.matmul(attention_weights, V_B)
        return attention_output


class Dynamic_Fusion(nn.Module):
    """
    动态融合模块（Dynamic Fusion Module）- PSAFNet的核心融合组件

    该模块通过交叉注意力对齐两个相位的特征，然后使用时间卷积网络（TCN）进行动态融合。
    同时计算MMD（Maximum Mean Discrepancy）损失以促进两个相位特征分布的对齐。

    论文参考：第3.3节 - "动态融合模块首先使用双向交叉注意力对齐两个相位的特征，
    然后通过时间填充模拟相位间的时间偏移，最后使用TCN进行融合并输出分类结果"

    关键创新：
    1. 双向交叉注意力：feature1关注feature2，feature2关注feature1
    2. 时间对齐填充：模拟两个相位之间的时间偏移（论文图4）
    3. MMD损失：促进两个相位特征分布的一致性

    参数:
        input_dim (int): 输入特征维度
        hidden_dim (int): TCN隐藏层维度
        output_dim (int): 输出类别数
        kernel_size (int): TCN卷积核大小（默认：3）
        num_layers (int): TCN层数（默认：3）
    """
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, num_layers=3):
        super(Dynamic_Fusion, self).__init__()
        # 构建多层TCN，膨胀率指数增长
        self.tcn_layers = nn.ModuleList()
        dilation = 1
        for i in range(num_layers):
            self.tcn_layers.append(TCNLayer(input_dim, hidden_dim, kernel_size, dilation))
            input_dim = hidden_dim
            dilation *= config.dilation_expand  # 膨胀率指数增长（默认×2）

        # 全连接层：从TCN特征到分类输出
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 双向交叉注意力模块
        input_dim = config.init_conv_layers * config.conv_depth
        self.cross_attention_A = CrossAttention(input_dim=input_dim)  # feature1关注feature2
        self.cross_attention_B = CrossAttention(input_dim=input_dim)  # feature2关注feature1

        # 推理模式标志和权重记录（用于可视化分析）
        self.inference_mode = False
        self.temporal_weights_1 = []
        self.temporal_weights_2 = []

    def forward(self, feature1, feature2):
        """
        动态融合的前向传播

        参数:
            feature1: 第一个相位的特征，形状为 [batch, feature_layers, 1, time_points]
            feature2: 第二个相位的特征，形状为 [batch, feature_layers, 1, time_points]

        返回:
            output: 分类输出，形状为 [batch, num_classes]
            loss: MMD对齐损失
        """
        batch_size, feature_layers, _, time_points = feature1.size()

        # 重塑特征：[batch, features, 1, time] -> [batch, time, features]
        feature1 = feature1.squeeze(2).permute(0, 2, 1)
        feature2 = feature2.squeeze(2).permute(0, 2, 1)

        # 双向交叉注意力对齐（论文第3.3节）
        # feature1通过关注feature2来增强自身，使用残差连接
        feature1 = self.cross_attention_A(feature2, feature1) * feature1 + feature1
        # feature2通过关注feature1来增强自身，使用残差连接
        feature2 = self.cross_attention_B(feature1, feature2) * feature2 + feature2

        # 时间对齐填充（论文图4）
        # 模拟两个相位之间的时间偏移：feature1在前，feature2在后
        stage_timepoints = config.stage_timepoints
        padding_length = round(time_points * (200 / stage_timepoints - 1))

        # feature1右侧填充（模拟早期相位）
        padded_feature1 = torch.cat([feature1, torch.zeros(batch_size, padding_length, feature_layers).to(feature1.device)], dim=1)
        # feature2左侧填充（模拟晚期相位）
        padded_feature2 = torch.cat([torch.zeros(batch_size, padding_length, feature_layers).to(feature2.device), feature2], dim=1)

        # 融合两个对齐后的特征
        dynamic_input = padded_feature1 + padded_feature2

        # 转换为TCN输入格式：[batch, time, features] -> [batch, features, time]
        dynamic_input = dynamic_input.permute(0, 2, 1)

        # 通过多层TCN提取时间依赖
        for tcn_layer in self.tcn_layers:
            dynamic_input = tcn_layer(dynamic_input)

        # 时间维度平均池化
        dynamic_input = dynamic_input.mean(dim=2)

        # 全连接层输出分类结果
        output = self.fc(dynamic_input)

        # 计算MMD损失以对齐两个相位的特征分布
        self.mmd_sigma = config.mmd_sigma
        loss = self.compute_mmd_loss(feature1, feature2)

        return output, loss

    def compute_mmd_loss(self, feature1, feature2):
        """
        计算最大均值差异（Maximum Mean Discrepancy, MMD）损失

        MMD是一种衡量两个分布差异的非参数方法，通过核技巧在再生核希尔伯特空间（RKHS）
        中比较两个分布的均值。

        论文参考：第3.3节 - "使用MMD损失促进两个相位特征分布的对齐，减少相位间的
        分布差异，提高融合效果"

        参数:
            feature1: 第一个相位的特征 [batch, time, features]
            feature2: 第二个相位的特征 [batch, time, features]

        返回:
            mmd_loss: MMD损失值（标量）
        """
        # 计算feature1内部的成对距离
        diff1 = feature1.unsqueeze(1) - feature1.unsqueeze(0)  # [batch, batch, time, features]
        # 计算feature2内部的成对距离
        diff2 = feature2.unsqueeze(1) - feature2.unsqueeze(0)
        # 计算feature1和feature2之间的成对距离
        diff_cross = feature1.unsqueeze(1) - feature2.unsqueeze(0)

        # 计算欧氏距离的平方
        dist1 = torch.sum(diff1 ** 2, dim=-1)  # [batch, batch, time]
        dist2 = torch.sum(diff2 ** 2, dim=-1)
        dist_cross = torch.sum(diff_cross ** 2, dim=-1)

        # 使用高斯核（RBF核）计算核矩阵
        K_XX = torch.exp(-dist1 / (2 * self.mmd_sigma ** 2))  # feature1内部的核矩阵
        K_YY = torch.exp(-dist2 / (2 * self.mmd_sigma ** 2))  # feature2内部的核矩阵
        K_XY = torch.exp(-dist_cross / (2 * self.mmd_sigma ** 2))  # feature1和feature2之间的核矩阵

        # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        mmd_loss = torch.mean(K_XX + K_YY - 2 * K_XY)

        return mmd_loss


class PSAFNet(nn.Module):
    """
    PSAFNet: Phase Segment and Aligned Fusion Network
    相位分割与对齐融合网络

    这是论文的主模型，实现了基于脑启发的EEG视频目标检测深度学习架构。核心思想是：
    1. 将EEG信号分割为两个重叠的时间相位（模拟大脑的时间处理机制）
    2. 使用两个独立的相位编码器提取各自的空间-时间特征
    3. 通过动态融合模块对齐并融合两个相位的特征
    4. 使用MMD损失促进相位间特征分布的一致性

    论文参考：
    - 第3节整体架构图（图2）
    - 第3.1节：相位分割策略 - "将1秒EEG信号分为两个重叠的相位，模拟大脑对视觉刺激
      的早期和晚期响应"
    - 第3.2节：相位编码器 - "使用多尺度时间卷积和注意力机制提取特征"
    - 第3.3节：对齐融合 - "通过交叉注意力和TCN融合两个相位的特征"

    参数:
        stage_timepoints (int): 每个相位的时间点数（默认150，对应0.75秒@200Hz）
        lead (int): EEG导联数（空间维度，默认59）
        time (int): 总时间点数（默认200，对应1秒@200Hz）
    """

    def __init__(self, stage_timepoints, lead, time):
        super(PSAFNet, self).__init__()

        self.lead = lead  # EEG导联数（通道数）
        self.time = time  # 总时间点数
        self.stage_time = stage_timepoints  # 每个相位的时间点数

        # 两个独立的相位编码器，用于处理分割后的时间相位
        # 论文第3.2节：使用权重独立的编码器以捕获不同相位的特异性特征
        self.time_model1 = Phased_Encoder(lead)  # 早期相位编码器
        self.time_model2 = Phased_Encoder(lead)  # 晚期相位编码器

        # 动态融合模块：使用TCN进行时间建模和特征融合
        self.TCN_fuse = Dynamic_Fusion(
            config.init_conv_layers * config.conv_depth,  # 输入特征维度
            config.TCN_hidden_dim,  # TCN隐藏层维度
            config.num_class  # 输出类别数（2：目标/非目标）
        )

    def forward(self, x):
        """
        PSAFNet的前向传播

        参数:
            x: 输入EEG张量，形状为 [batch_size, 1, channels, timepoints]
               - batch_size: 批次大小
               - 1: 特征通道数（单通道输入）
               - channels: EEG导联数（59）
               - timepoints: 时间点数（200）

        返回:
            y_fuse: 融合后的分类输出，形状为 [batch_size, num_classes]
            similarity_loss: 两个相位特征集之间的MMD损失（用于训练时的正则化）

        数据流程：
            输入EEG [B,1,59,200]
                ↓ 相位分割
            相位1 [B,1,59,150]  相位2 [B,1,59,150]
                ↓                    ↓
            编码器1              编码器2
                ↓                    ↓
            特征1 [B,24,1,T]    特征2 [B,24,1,T]
                ↓                    ↓
                    动态融合模块
                         ↓
                输出 [B,2] + MMD损失
        """

        # 相位分割（论文第3.1节）
        # 相位1：取前stage_time个时间点（例如前150个点，0-0.75秒）
        x1 = x[:, :, :, :self.stage_time]
        # 相位2：取后stage_time个时间点（例如后150个点，0.25-1秒）
        # 注意：两个相位有重叠区域，这模拟了大脑处理视觉信息时的时间连续性
        x2 = x[:, :, :, self.time - self.stage_time:]

        # 使用独立的编码器提取各相位的空间-时间特征
        y1 = self.time_model1(x1)  # 早期相位特征
        y2 = self.time_model2(x2)  # 晚期相位特征

        # 动态融合：对齐并融合两个相位的特征，同时计算MMD损失
        y_fuse, similarity_loss = self.TCN_fuse(y1, y2)

        return y_fuse, similarity_loss
