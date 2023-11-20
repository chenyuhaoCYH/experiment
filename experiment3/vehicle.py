# -*- coding: utf-8 -*-

import numpy as np
from memory import ExperienceBuffer, PPOMemory
from task import Task

Dv = 100  # 车的最大通信范围
Fv = 2000  # 车最大计算能力  MHZ

CAPACITY = 50000  # 缓冲池大小
TASK_SOLT = 20  # 任务产生时隙

np.random.seed(2)


class Vehicle:
    # 位置：x，y 速度、方向：-1左，1右
    def __init__(self, id, position, direction, velocity=20):
        self.id = id
        # 车的位置信息
        self.loc_x = position[0]
        self.loc_y = position[1]
        self.position = position
        self.velocity = velocity  # m/s
        self.direction = direction
        # 通信范围
        self.range = Dv
        # 通信功率
        self.power = 0
        # 信噪比
        self.fade = 0
        # 邻居表
        self.neighbor = []
        # mec
        self.Mec = None
        # 当前时间
        self.cur_frame = 0
        # 接受的任务的列表
        self.accept_task = []
        # 当前选择此目标的车辆
        self.task_vehicle = []
        # 接受任务的数量(包括处理的任务和正在等待的任务)
        self.sum_needDeal_task = 0
        # 当前可用资源
        self.resources = 2000  # round((1 - np.random.randint(1, 3) / 10) * Fv, 2)  # MHz
        # 当前总资源
        self.cur_resource = self.resources
        # 表示当前是否有任务正在传输给邻居车辆（0：没有，1：有）
        self.trans_task_for_vehicle = 0
        # 当前选择的信道
        self.band = -1
        # 当前处理的任务（用于计算奖励，不用于状态信息）
        self.cur_task = None
        # 判断是否需要重新计算
        self.flag = False
        # 卸载成功率
        self.success_rate = 0
        self.success_task = 0
        self.sum_create_task = 0

        # 当前状态信息
        self.self_state = []
        # 邻居车状态
        self.neighbor_state = []
        # 去除邻居的状态信息用于邻居车观察和全局critic的处理
        self.excludeNeighbor_state = []
        # 缓冲池
        self.buffer = ExperienceBuffer(capacity=CAPACITY)
        # 总奖励
        self.reward = []
        # 任务溢出的数量
        self.overflow = 0
        # 上一个任务产生的时间
        self.lastCreatWorkTime = 0

        self.timeSolt = TASK_SOLT  # * (id % 2 + 1)
        self.memory = ExperienceBuffer(CAPACITY)
        # 产生任务
        self.create_work()

    # 获得位置
    @property
    def get_location(self):
        return self.position

    # 设置位置
    def set_location(self, loc_x, loc_y):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.position = [self.loc_x, self.loc_y]

    # 获得x
    @property
    def get_x(self):
        return self.loc_x

    # 获得y
    @property
    def get_y(self):
        return self.loc_y

    # 产生任务 传入当前时间
    def create_work(self):
        # if self.id % 4 == 0:
        #     return
        # 每隔一段时间进行一次任务产生
        if self.cur_frame - self.lastCreatWorkTime >= self.timeSolt and self.cur_task is None:
            # # 每次有0.8的概率产生任务
            if np.random.random() < 0.5:
                self.cur_task = Task(self, self.cur_frame % 20000)
                self.lastCreatWorkTime = self.cur_frame
            else:
                self.cur_task = None

    def clear(self):
        self.cur_resource = self.resources
        self.task_vehicle = []
        self.overflow = 0

    """
    获得状态
    """

    def get_state(self):

        self.self_state = []

        # 位置信息  4
        self.self_state.extend(self.position)
        self.self_state.append(self.velocity)
        self.self_state.append(self.direction)

        # 资源信息（可用资源）
        self.self_state.append(self.resources)

        # 当前处理的任务量
        self.self_state.append(len(self.accept_task))

        # 当前选择的信道
        self.self_state.append(self.band)

        # 当前是否有任务在传输
        # self.self_state.append(self.queue_for_trans_vehicle.size())

        # 任务状态
        if self.cur_task is not None:
            self.self_state.append(self.cur_task.size)
            self.self_state.append(self.cur_task.cycle)
        else:
            self.self_state.append(0)
            self.self_state.append(0)

        # MEC状态
        self.self_state.extend(self.Mec.get_state())

        return self.self_state

    def get_neighbor_states(self):

        self.neighbor_state = []
        # 邻居状态
        for neighbor in self.neighbor:
            self.neighbor_state.append(neighbor.get_state())
        return self.neighbor_state
