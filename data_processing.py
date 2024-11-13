import import_ipynb
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import json
from data_generator import DataLoaderSequence
import math
import random
from PyEMD import EMD

class DataLoader():
    """A class for loading and transforming data for the EMD_lstm model"""

    def __init__(self, filename, split1, split2, cols, pre_len, input_timesteps, seq_len, **EMD_para):
        """
        :param filename: the name of the file contains the data, type: .csv
        :param split1: split the data into 2 parts: training, (validation, test)
        :param split2: split the data into 2 parts: validation, test
        |-------------------------------------------|-------------|--------------|
                                                 split1(0.7)   split2(0.85)     
        :param cols: the features 
        :param pre_len: the prediction length (24, 48....)
        :param input_timesteps: the length of looking back (1 month or 1 year), unit: hours
        :param seq_len: the sum of input_timesteps and pre_len
        :param **EMD_para: if apply EMD before LSTM
        """
        # 读取 CSV 文件到 DataFrame
        self.dataframe = pd.read_csv(filename, sep=',')
        if not isinstance(self.dataframe.index, pd.DatetimeIndex):
            # 如果数据框的索引不是日期时间索引，转换为日期时间索引
            self.dataframe['Date_Time'] = pd.to_datetime(self.dataframe['Date_Time'])
            self.dataframe = self.dataframe.set_index('Date_Time')
        
        # 初始化变量
        self.cols = cols  # 需要使用的特征列
        self.split1 = split1  # 训练数据的比例
        self.split2 = split2  # 验证集与测试集的比例
        self.len_train_windows = None
        self.pre_len = pre_len  # 预测长度
        self.input_timesteps = input_timesteps  # 输入时间步数
        self.seq_len = seq_len  # 输入时间步和预测长度之和
        
        # 输出输入的特征列
        print('the input cols are:', self.cols)
        
        # 进行数据预处理，包括归一化和EMD
        self.Normalization(**EMD_para)
    
    def scale_EMD(self, activate_EMD=False):
        '''
        使用对数对数据进行缩放，以突出数据的峰值
        
        param activate_EMD: 决定是否在进行 LSTM 之前使用 EMD 进行预处理
        '''
        # 对 'Consumption' 列进行对数缩放
        for col in self.cols:
            if col == 'Consumption':
                self.dataframe['Consumption'] = self.dataframe.set_index('Consumption').index.map(lambda x: math.log(x))
                print('scaling Consumption is done!')
        
        # 如果激活 EMD，执行经验模态分解
        if activate_EMD == True:
            # 对 'Consumption' 列应用 EMD 分解
            self.IMFs = EMD().emd(self.dataframe['Consumption'].values)
            print('the signal is decomposed into ' + str(self.IMFs.shape[0]) + ' parts')
            
            # 保存每个 IMF 结果
            self.df_names_IMF = locals()
            
            # 将每个 IMF 分量和其他列合并
            for ind, IMF in enumerate(self.IMFs):
                IMF_name = 'IMF' + str(ind) + '_consumption'
                data = {IMF_name: self.IMFs[ind]}
                IMF_i = pd.DataFrame(data=data)
                self.df_names_IMF['IMF' + str(ind)] = pd.concat([IMF_i[IMF_name], self.dataframe.get(self.cols[1:])], axis=1)

    def Normalization(self, **EMD_para):
        '''
        调用函数 scale_EMD()，决定是否使用 EMD 进行预处理
        对训练数据进行归一化，并应用相同的缩放方法处理验证和测试数据
        '''
        # 按照比例将数据分为训练集、验证集和测试集
        i_split1 = int(len(self.dataframe) * self.split1)
        i_split2 = int(len(self.dataframe) * self.split2)
        
        if len(EMD_para) == 0:
            # 如果没有传入 EMD 参数，则直接进行普通缩放
            self.scale_EMD()
            
            # 划分训练、验证和测试集
            self.data_train_original = self.dataframe.get(self.cols)[:i_split1]
            self.data_val_original = self.dataframe.get(self.cols)[i_split1:i_split2]
            self.data_test_original = self.dataframe.get(self.cols)[i_split2:]
        else:
            # 如果传入 EMD 参数，则应用 EMD 进行数据预处理
            self.scale_EMD(activate_EMD=True)
            IMF_number = EMD_para['IMF_num']
            
            print('processing the data of IM' + str(IMF_number))
            
            # 如果 IMF 参数有效，使用相应的 IMF 分量
            if IMF_number in range(self.IMFs.shape[0]):
                self.data_train_original = self.df_names_IMF['IMF' + str(IMF_number)][:i_split1]
                self.data_val_original = self.df_names_IMF['IMF' + str(IMF_number)][i_split1:i_split2]
                self.data_test_original = self.df_names_IMF['IMF' + str(IMF_number)][i_split2:]
            else:
                print("Oops! IMF_number was no valid number. it must between 0 and " + str(self.IMFs.shape[0] - 1))

        # 使用 MinMaxScaler 对数据进行归一化
        self.min_max_scaler = preprocessing.MinMaxScaler().fit(self.data_train_original.values)

        # 对训练、验证和测试集进行归一化
        self.data_train = self.min_max_scaler.transform(self.data_train_original.values)
        self.data_val = self.min_max_scaler.transform(self.data_val_original.values)
        self.data_test = self.min_max_scaler.transform(self.data_test_original.values)

        # 获取训练、验证和测试集的长度
        self.len_train = len(self.data_train_original)
        self.len_val = len(self.data_val_original)
        self.len_test = len(self.data_test_original)
    
    def get_pre_time(self):
        data_windows = []
        
        # 获取预测时间的窗口
        for i in range((self.len_test - self.input_timesteps) // self.pre_len):
            data_windows.append(self.data_test_original.index[i * self.pre_len:i * self.pre_len + self.seq_len])
        
        pre_time = np.array([p[self.input_timesteps:] for p in data_windows])
        return pre_time
    
    def get_test_data(self):
        '''
        创建测试数据集的 x, y 窗口
        警告：该方法使用批量方式加载数据，确保有足够的内存来加载数据，否则减少训练集的大小。
        '''
        data_windows = []

        # 创建测试数据的窗口
        for i in range((self.len_test - self.input_timesteps) // self.pre_len):
            data_windows.append(self.data_test[i * self.pre_len:i * self.pre_len + self.seq_len])

        # 获取 X 和 Y 数据
        x = np.array([p[:self.input_timesteps, :] for p in data_windows])
        y = np.array([p[self.input_timesteps:, 0] for p in data_windows])
        return x, y
    
    def get_train_data(self):
        '''
        创建训练数据集的 x, y 窗口
        警告：该方法使用批量方式加载数据，确保有足够的内存来加载数据，否则使用 generate_training_window() 方法。
        '''
        train_x = []
        train_y = []
        
        # 创建训练数据的窗口
        for i in range(self.len_train - self.seq_len):
            data_window = self.data_train[i:i + self.seq_len]
            train_x.append(data_window[:self.input_timesteps, :])
            train_y.append(data_window[self.input_timesteps:, 0])
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        
        # 随机打乱训练数据
        sfl = list(range(len(train_x)))
        random.shuffle(sfl)
        train_x = train_x[sfl]
        train_y = train_y[sfl]
        
        return train_x, train_y
            
    def get_val_data(self):
        '''
        创建验证数据集的 x, y 窗口
        '''
        val_x = []
        val_y = []
        
        # 创建验证数据的窗口
        for i in range(self.len_val - self.seq_len):
            data_window = self.data_val[i:i + self.seq_len]
            val_x.append(data_window[:self.input_timesteps, :])
            val_y.append(data_window[self.input_timesteps:, 0])
        
        val_x = np.array(val_x)
        val_y = np.array(val_y)
        
        # 随机打乱验证数据
        sfl = list(range(len(val_x)))
        random.shuffle(sfl)
        val_x = val_x[sfl]
        val_y = val_y[sfl]
        
        return val_x, val_y
    
    def training_batch_generator(self, batch_size):
        '''
        生成训练数据批次的生成器
        '''
        train_x, train_y = self.get_train_data()
        return DataLoaderSequence(train_x, train_y, batch_size)
    
    def val_batch_generator(self, batch_size):
        '''
        生成验证数据批次的生成器
        '''
        val_x, val_y = self.get_val_data()
        return DataLoaderSequence(val_x, val_y, batch_size)

# 示例：如何使用 DataLoader 类加载数据
# configs = json.load(open('config.json', 'r'))
# data = DataLoader(
#     filename=os.path.join('data', configs['data']['filename']),
#     split1=configs['data']['train_test_split1'],
#     split2=configs['data']['train_test_split2'],
#     cols=configs['data']['columns'],
#     pre_len=configs['model']['layers'][4]['neurons'],
#     input_timesteps=configs['model']['layers'][0]['input_timesteps'],
#     seq_len=configs['data']['sequence_length'],
#     IMF_num=10
# )
# data.dataframe
