import numpy as np
from keras.utils import Sequence
class DataLoaderSequence(Sequence):
    """
    自定义数据加载器类，继承自 Keras 的 Sequence 类。
    这种方式适用于生成批次数据，尤其是对于需要按批次加载大规模数据集的场景，常用于训练深度学习模型。
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        初始化 DataLoaderSequence 类。
        
        :param x_set: 输入数据（特征），通常是一个 NumPy 数组。
        :param y_set: 目标数据（标签），通常是一个 NumPy 数组。
        :param batch_size: 每个批次的数据量。
        """
        self.x, self.y = x_set, y_set  # 分配输入数据和目标数据
        self.batch_size = batch_size    # 设置每个批次的大小

    def __len__(self):
        """
        返回每个 epoch 中的批次数量。
        Keras 会调用此方法来计算训练过程中每个 epoch 需要多少个步骤。
        
        :return: 该数据集的批次数量。
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))
        # 返回总样本数除以批次大小后的向上取整值，表示总批次数

    def __getitem__(self, idx):
        """
        获取给定索引 `idx` 对应的一个批次的数据。
        
        :param idx: 批次的索引。
        
        :return: 一个元组 (batch_x, batch_y)，其中：
                 - batch_x: 当前批次的输入数据（特征）。
                 - batch_y: 当前批次的目标数据（标签）。
        """
        # 计算当前批次的起始和结束索引
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size, :, :]  # 获取当前批次的输入数据
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size, :]      # 获取当前批次的目标数据

        # 返回批次数据，转换为 NumPy 数组
        return np.array(batch_x), np.array(batch_y)
