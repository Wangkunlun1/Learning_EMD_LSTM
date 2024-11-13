import os
import json
import time
import datetime as dt
import math
import import_ipynb
from data_processing import DataLoader  # 导入数据处理模块
from build_model import Model  # 导入模型构建模块
import pandas as pd
import numpy as np
import keras
import tensorflow
# from plotly.offline import iplot
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)  # 初始化Plotly离线模式

# 绘制预测结果与真实数据对比图
def plot_results(predicted_data, true_data, pre_time):
    pre_time = pd.to_datetime(pre_time)  # 转换时间戳为日期时间格式

    # 绘制真实数据曲线
    trace1 = go.Scatter(x=pre_time,
                        y=true_data,
                        mode='lines',
                        name='True',
                        hoverinfo='name',
                        line=dict(
                                    shape='spline'  # 平滑曲线
                                 )
                        )
    
    # 绘制预测数据曲线
    trace2 = go.Scatter(x=pre_time,
                        y=predicted_data,
                        mode='lines',
                        name='Prediction',
                        hoverinfo='name',
                        line=dict(
                                    shape='spline'  # 平滑曲线
                                 )
                        )
    # 组合绘图数据
    data = [trace1, trace2]
    layout = go.Layout(title = 'Prediction & True',
                       xaxis = dict(title = 'timestamp')  # X轴为时间戳
                  )
    # 绘制图形
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig)  # 显示图形

    # 读取配置文件
    configs = json.load(open('config.json', 'r'))

# 如果模型保存目录不存在，则创建该目录
if not os.path.exists(configs['model']['save_dir']): 
    os.makedirs(configs['model']['save_dir'])

# 输出使用的数据集文件名
print('import dataset:', configs['data']['filename'])

# 加载数据
data = DataLoader(
    filename=os.path.join('data', configs['data']['filename']),
    split1=configs['data']['train_test_split1'],
    split2=configs['data']['train_test_split2'],
    cols=configs['data']['columns'],
    pre_len=24,  # 使用24个小时作为预测长度
    input_timesteps=configs['model']['layers'][0]['input_timesteps'],
    seq_len=configs['data']['sequence_length']
)

# 获取训练数据
train_x, train_y = data.get_train_data()
train_x.shape  # 输出训练数据的形状
val_x, val_y = data.get_val_data()  # 获取验证数据
# train_x, train_y, val_x, val_y = data.get_train_val_data()
# train_x.shape, x_test, y_test = data.get_test_data()
pre_time = data.get_pre_time()  # 获取预测时间

# 计算每个训练周期的步数
steps_per_epoch = math.ceil((len(train_x) - configs['data']['sequence_length']) / 
                            configs['training']['batch_size'])

# 模型保存目录
save_dir = configs['model']['save_dir']

y_test.shapescore = {}  # 初始化测试集的score
models_dir = []  # 用于保存模型目录

# 训练模型的循环（这里只训练一个模型）
for i in range(1):
    print('Starting training %s Model' % (i+1))
    model = Model()  # 创建模型实例
    model.build_model(configs)  # 构建模型
    
    # 获取当前时间，用于保存模型的文件夹名称
    time_now = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = configs['model']['save_dir']
    save_dir = os.path.join(save_dir, '%s-e%s' % (time_now, str(i)))
    models_dir.append(save_dir)
    os.makedirs(save_dir)  # 创建保存模型的目录
    
    save_fname = os.path.join(save_dir, 'e%s.h5' % (str(i)))  # 定义模型保存路径
    log_fname = save_dir  # 日志保存路径
    
    # 将模型结构保存为JSON文件
    model.model_to_json(save_dir)
    
    # 定义模型信息文件
    fname = os.path.join(save_dir, 'model_information.json')
    
    # 训练模型
    model.train_generator(
        data_gen=data.training_batch_generator(
            batch_size=configs['training']['batch_size']
        ),
        val_gen=data.val_batch_generator(
            batch_size=configs['training']['batch_size']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        log_fname=log_fname,
        save_fname=save_fname
    )
    
    # 在验证集上评估模型
    score_ = model.evaluate(val_x, val_y)
    print("loss:", score_)
    score[save_fname] = score_
    
    # 保存模型训练信息
    with open(fname, "w") as to:
        with open("./config.json", 'r') as original:
            m = json.loads(original.read())  # 读取原始配置文件
            m['loss'] = score_  # 将训练损失写入配置
            json_str = json.dumps(m)
            to.write(json_str)
            
    print('[Model] Store model_information at %s' % fname)
    
# 找到表现最好的模型
filename_best = min(score, key=score.get)
print(filename_best)

# 加载最佳模型
model.load_model(filename_best)

# 使用最佳模型进行预测
predictions = model.predict_point_by_point(x_test)

# 绘制预测结果与真实结果的对比图
plot_results(predicted_data=predictions, 
             true_data=np.reshape(y_test, (y_test.size,)),
             pre_time=np.reshape(pre_time, (pre_time.size,)))
