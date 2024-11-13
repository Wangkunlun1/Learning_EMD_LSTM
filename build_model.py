import import_ipynb
import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from utils import Timer  # 引入自定义的计时器类
# 引入 Keras 所需的各种模块，进行深度学习模型的构建、训练、保存和评估。
 
class Model():
    """用于构建、训练和推理 LSTM 模型的类"""
    
    def __init__(self):
        self.model = Sequential()  # 初始化一个 Keras 顺序模型（Sequential），用于逐层堆叠神经网络
 
    def load_model(self, filepath):
        """加载一个已经训练的模型"""
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)  # 从指定路径加载保存的模型文件
 
    def build_model(self, configs):
        """根据配置文件构建模型"""
        timer = Timer()  # 创建一个计时器对象，用于计算构建模型的时间
        timer.start()
        
        # 根据配置文件中的层定义，逐层构建神经网络
        for layer in configs['model']['layers']:  # 遍历每一层的配置
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            
            # 根据不同的层类型添加相应的层
            if layer['type'] == 'dense':  # 如果是全连接层
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':  # 如果是 LSTM 层
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':  # 如果是 Dropout 层
                self.model.add(Dropout(dropout_rate))
        
        # 编译模型，指定损失函数和优化器
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        print('[Model] Model Compiled')  # 输出模型已编译完成的信息
        timer.stop()  # 结束计时并输出耗时
 
    def model_to_json(self, save_dir):
        """将模型结构保存为 JSON 文件"""
        model_json = self.model.to_json()  # 将模型的结构转化为 JSON 格式
        fname = os.path.join(save_dir, 'model.json')  # 构建保存路径
        with open(fname, "w") as json_file:
            json_file.write(model_json)  # 将模型结构保存到指定路径
        print('[Model] Serialize model to JSON at %s' % fname)
 
    def train(self, x, y, epochs, batch_size, save_dir):
        """训练模型并保存最佳模型"""
        timer = Timer()  # 创建计时器对象，记录训练时间
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))  # 打印训练信息
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))  # 生成保存模型的文件名
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),  # 如果验证集损失不再下降，提前停止训练
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)  # 仅保存损失最小的模型
        ]
        
        # 使用给定的训练数据训练模型
        self.model.fit(
            x,  # 输入数据
            y,  # 输出数据
            epochs=epochs,  # 训练轮数
            batch_size=batch_size,  # 批量大小
            callbacks=callbacks  # 设置回调函数
        )
        self.model.save(save_fname)  # 保存最终训练好的模型
        print('[Model] Training Completed. Model saved as %s' % save_fname)  # 输出训练完成并保存模型的信息
        timer.stop()  # 结束计时并输出耗时
 
    def train_generator(self, data_gen, val_gen, epochs, batch_size, steps_per_epoch, log_fname, save_fname):
        """使用数据生成器训练模型"""
        timer = Timer()  # 创建计时器对象，记录训练时间
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))  # 打印训练信息

        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),  # 监控验证集损失并保存最好的模型
            TensorBoard(log_dir=log_fname, histogram_freq=0, write_graph=True, write_images=True)  # 记录 TensorBoard 日志
        ]
        
        # 使用数据生成器进行训练
        self.model.fit_generator(
            data_gen,  # 训练数据生成器
            validation_data=val_gen,  # 验证数据生成器
            validation_steps=1,  # 验证步骤数
            steps_per_epoch=steps_per_epoch,  # 每个 epoch 的步数
            epochs=epochs,  # 训练轮数
            callbacks=callbacks,  # 设置回调函数
            workers=1  # 工作线程数，通常根据 CPU 核心数调整
        )
        print('[Model] Training Completed. Model saved as %s' % save_fname)  # 输出训练完成并保存模型的信息
        timer.stop()  # 结束计时并输出耗时
 
    def predict_point_by_point(self, data):
        """逐点预测每个时间步的值"""
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)  # 使用模型预测
        predicted = np.reshape(predicted, (predicted.size,))  # 将预测结果展平为一维数组
        return predicted  # 返回预测结果
 
    def evaluate(self, x_test, y_test):
        """评估模型的表现"""
        score = self.model.evaluate(x=x_test, y=y_test)  # 使用测试数据评估模型性能
        return score  # 返回评估结果
 
    def predict_sequences_multiple(self, data, window_size, prediction_len):
        """预测多个时间步长的序列"""
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []  # 用于存储多个预测序列
        
        # 遍历数据并预测每一个序列
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]  # 获取当前时间窗
            
            predicted = []  # 存储每个序列的预测结果
            for j in range(prediction_len):  # 对每个时间步进行预测
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])  # 预测并将结果存入 predicted
                curr_frame = curr_frame[1:]  # 更新当前时间窗
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)  # 插入新的预测值
            prediction_seqs.append(predicted)  # 将预测序列添加到 prediction_seqs 中
        return prediction_seqs  # 返回所有预测的序列
 
    def predict_sequence_full(self, data, window_size):
        """通过滑动窗口预测完整的时间序列"""
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]  # 获取初始时间窗
        predicted = []  # 存储预测结果
        
        # 遍历数据并预测每个时间步的值
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])  # 预测当前时间窗的下一个时间步
            curr_frame = curr_frame[1:]  # 更新当前时间窗
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)  # 插入新的预测值
        return predicted  # 返回完整的预测序列
