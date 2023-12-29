import numpy as np

np.random.seed(2)


class Task:
    """
    定义任务类型
    """

    def __init__(self, vehicle=None, createTime=0):
        # 产生任务的车辆
        self.vehicle = vehicle
        # 完成该任务所消耗的资源
        self.aim = None  # 传送对象

        self.size = np.random.uniform(0.5, 1) * 126 # np.random.uniform(0.5, 1)  # Mb
        self.cycle = np.random.randint(30, 50)  # cycle/bit
        self.need_trans_size = self.size  # Kb 还剩余多少未传输完成
        self.need_precess_cycle = self.cycle * self.size  # Mb * cycle/byte =M cycle 还剩余多少轮次未完成（10^6)

        self.need_time = 0  # 需要计算时间
        self.trans_time = 0  # 需要传输的时间
        self.hold_time = 0  # 任务在计算等待队列中得等待时间
        self.wait_time = 0  # 需要等待传输的时间

        self.rate = 0  # 当前速率
        self.compute_resource = 0  # 被分配的资源

        self.create_time = createTime  # 任务产生时间
        self.pick_time = 0  # 被选择的时间（出队列时间）
