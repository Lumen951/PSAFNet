import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    设置所有随机数生成器的种子以确保结果可复现

    在深度学习实验中，为了确保结果的可复现性，需要固定所有随机数生成器的种子。
    这包括Python内置的random模块、NumPy和PyTorch（CPU和GPU）。

    论文参考：第4.2节实验设置 - "为确保实验的可复现性，所有随机种子设置为42"

    参数:
        seed (int): 随机种子值，默认42

    注意:
        - 即使设置了种子，某些CUDA操作仍可能是非确定性的
        - 如需完全确定性，还需设置：
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False
    """
    random.seed(seed)  # 设置Python内置random模块的种子
    np.random.seed(seed)  # 设置NumPy随机数生成器的种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机数生成器种子
