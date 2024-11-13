import datetime as dt  # 导入 datetime 模块，用于处理日期和时间

class Timer():
    def __init__(self):
        self.start_dt = None  # 初始化时，没有设置开始时间
        
    def start(self):
        # 启动计时器，记录当前时间
        self.start_dt = dt.datetime.now()  # 获取当前的日期和时间，精确到秒
    
    def stop(self):
        # 停止计时器，记录结束时间
        end_dt = dt.datetime.now()  # 获取当前时间作为结束时间
        # 计算并打印程序执行的时间差（结束时间 - 开始时间）
        print('Time taken: %s' % (end_dt - self.start_dt))  # 输出时间差
