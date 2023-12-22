# -*- coding: utf-8 -*-
import numpy as np

RANGE_MEC = 1000  # MEC通信范围 /m
RESOURCE = 10000  # 可用资源  MHz
BrandWidth = 500  # MHz
AVERAGE = 5


# 边缘服务器
class MEC:
    def __init__(self, position, resources=RESOURCE):
        self.loc_x = position[0]
        self.loc_y = position[1]
        self.loc = position
        # 当前总资源
        self.cur_resource = resources
        # 当前可用资源 MHz
        self.resources = resources
        self.brand = BrandWidth / AVERAGE
        self.state = []
        # 通信范围 m
        self.range = RANGE_MEC
        # 当前接到需要处理的任务信息(最多同时处理10个任务)
        self.accept_task = []
        # 最多处理任务量
        self.max_task = 10
        # 接受任务的数量
        self.sum_needDeal_task = 0
        # 用于奖励计算的任务队列
        self.task_queue_for_reward = []
        # 当前选择此目标的车辆
        self.task_vehicle = []
        # 当前状态
        self.get_state()
        # 范围内的车辆
        self.range_vehicle = []
        # 每段频谱选择的车辆
        self.pick_vehicle = [[] for _ in range(AVERAGE)]

    @property
    def get_x(self):
        return self.loc_x

    @property
    def get_y(self):
        return self.loc_y

    @property
    def get_location(self):
        return self.loc

    def clear_vehicle(self):
        # self.resources = np.random.randint(2000, 10000)
        # 范围内的车辆
        self.range_vehicle = []
        self.task_vehicle = []
        # 每段频谱选择的车辆
        self.pick_vehicle = [[] for _ in range(AVERAGE)]
        # 当前接到需要处理的任务信息(最多同时处理10个任务)
        self.accept_task = []
        self.cur_resource = self.resources

    def re(self):
        # 范围内的车辆
        self.range_vehicle = []
        self.task_vehicle = []
        # 每段频谱选择的车辆
        self.pick_vehicle = [[] for _ in range(AVERAGE)]
        # 当前总资源
        self.cur_resource = RESOURCE
        # 当前可用资源 MHz
        self.resources = RESOURCE

    """
        获得状态
    """

    def get_state(self):
        """
        :return:state 维度：[loc_x,loc_y,sum_needDeal_task,resources]
        """
        self.state = []
        self.state.extend(self.loc)
        self.state.append(len(self.accept_task))
        self.state.append(self.resources)
        return self.state
