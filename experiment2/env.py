import math

import numpy as np
from mec import MEC
from vehicle import Vehicle

N = 20  # 车的数量
MAX_NEIGHBOR = 5  # 最大邻居数
CAPACITY = 20000  # 缓冲池大小
time_solt = 10

sigma = -114  # 噪声dbm
sig2 = 10 ** (sigma / 10)
POWER = 23  # 功率 dbm
POWER_V2V = 46

gama = 1.25  # * (10 ** -27)  # 能量系数 J/M cycle
a = 0.6  # 奖励中时间占比
b = 0.2  # 奖励中能量占比
c = 0.2  # 奖励中支付占比
T1 = -0.5
T2 = -0.7
T3 = 0.05
PUNISH = -5

# 价格系数（MEC、VEC、local）
MEC_Price = 0.15
VEC_Price = 0.1
LOC_Price = 0.3

Ki = -10  # 非法动惩罚项(会导致任务直接失败，所以惩罚力度大)
Kq = 0.25  # 任务队列长度系数
ko = 0.5  # 溢出任务系数
Ks = 0.5  # 奖励占比

np.random.seed(2)


class Env:
    def __init__(self, num_Vehicles=N):
        self.avg_trans_time = [[] for _ in range(N)]
        self.avg_compute_time = [[] for _ in range(N)]
        self.avg_reward = [[] for _ in range(N)]
        self.avg_time = [[] for _ in range(N)]
        self.avg_energy = [[] for _ in range(N)]
        self.avg_price = [[] for _ in range(N)]

        # 高速公路车道宽3.75m，隔离带1m，应急车道2.5m,mec部署位置为上下交叉放置 总宽21m
        self.vehicles = []
        # 下面两条右车道
        self.right_lanes = [3.75 / 2 + 2.5, 3.75 / 2 + 3.75 + 2.5]
        # 上面两条左车道
        self.left_lanes = [3.75 / 2 + 3.75 * 2 + 1 + 2.5, 3.75 / 2 + 3.75 * 3 + 1 + 2.5]
        # 环境总长度
        self.width = 3000

        self.hwy_large_scale_channel_sampler = HwyChannelLargeScaleFadingGenerator(8, 3, 1.5, 25, 2, 3, 8, 5, 9)
        # 基站天线高度
        self.bsHeight = 25
        # 车辆天线高度
        self.vehHeight = 1.5
        self.stdV2I = 8
        self.stdV2V = 3
        self.freq = 2
        self.vehAntGain = 3
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9

        # 环境内所有车辆
        self.vehicles = []
        # 环境内的mec位于最中间
        self.MECs = [MEC([21, 500]), MEC([0, 1500]), MEC([21, 2500])]

        # 车辆数以及mec数
        self.num_Vehicles = num_Vehicles

        # 所有需要传输的任务
        self.need_trans_task = []
        # 当前平均奖励奖励数
        self.Reward = 0
        # 记录每辆车的历史奖励
        self.vehicleReward = []
        # 当前奖励
        self.reward = [.0] * self.num_Vehicles
        # 当前时间
        self.cur_frame = 0
        # 所有车的卸载动作
        self.offloadingActions = [0] * num_Vehicles
        # 功率选择动作
        self.bandActions = [0] * num_Vehicles
        # 计算占比动作
        self.freqActions = [0] * num_Vehicles
        # 当前全局的状态信息
        self.vehicleState = []
        self.mecState = []
        self.neighborState = []
        # 系统的缓冲池
        self.buffer = []

    # 添加车辆
    def add_new_vehicles(self, id, position, direction, velocity):
        vehicle = Vehicle(id=id, position=position, direction=direction, velocity=velocity)
        self.vehicles.append(vehicle)

    # 初始化/重置环境
    def reset(self):
        self.Reward = 0
        self.vehicleState = []
        self.cur_frame = 0
        self.offloadingActions = [0] * self.num_Vehicles
        self.reward = [0] * self.num_Vehicles

        for i in range(self.num_Vehicles):
            self.vehicleReward.append([])

        i = 0
        while i < self.num_Vehicles:  # 初始化车子
            # 靠近隔离带的快速车道速度为[100，120]km/h-[28,33]m/s，靠近应急车道的车道速度为[60，100]km/h-[17,27]m/s
            start_position = [self.right_lanes[0], np.random.randint(0, self.width)]
            start_direction = 1
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(28, 33))
            i += 1

            start_position = [self.left_lanes[0], np.random.randint(0, self.width)]
            start_direction = -1
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(17, 27))
            i += 1

            start_position = [self.right_lanes[1], np.random.randint(0, self.width)]
            start_direction = 1
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(17, 27))
            i += 1

            start_position = [self.left_lanes[1], np.random.randint(0, self.width)]
            start_direction = -1
            self.add_new_vehicles(i, start_position, start_direction, np.random.randint(28, 33))
            i += 1
        # 初始化邻居信息
        self.renew_neighbor()
        # 初始化邻居mec
        self.renew_mec()
        # 初始化状态信息
        for vehicle in self.vehicles:
            # 全局状态
            self.vehicleState.append(vehicle.get_state())
        for vehicle in self.vehicles:
            # 邻居状态
            self.neighborState.append(vehicle.get_neighbor_states())
        for mec in self.MECs:
            self.mecState.append(mec.get_state())

    # 更新每辆车邻居表
    def renew_neighbor(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbor = []
        z = np.array([[complex(vehicle.get_x, vehicle.get_y) for vehicle in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(MAX_NEIGHBOR):
                self.vehicles[i].neighbor.append(self.vehicles[sort_idx[j + 1]])

    # 更新每辆车最近的mec
    def renew_mec(self):
        for mec in self.MECs:
            mec.clear_vehicle()
        for i in range(len(self.vehicles)):
            vehicle = self.vehicles[i]
            cur = -1
            min_distance = 5000
            for j in range(len(self.MECs)):
                mec = self.MECs[j].get_location
                distance = np.sqrt(
                    np.power(mec[0] - vehicle.get_x, 2) + np.power(mec[1] - vehicle.get_y, 2))
                if distance < min_distance:
                    min_distance = distance
                    cur = j
            vehicle.Mec = self.MECs[cur]
            self.MECs[cur].range_vehicle.append(vehicle)

    # 获得卸载对象
    @staticmethod
    def get_aim(vehicle: Vehicle, action):
        if action == 0:
            return vehicle
        elif action == 1:
            return vehicle.Mec
        else:
            return vehicle.neighbor[action - 2]

    # 判断分配比率的合法性
    def process_fracActions(self):
        # 找到目标车辆
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.cur_task is not None:
                # 获得卸载对象
                aim = self.get_aim(vehicle, self.offloadingActions[i])
                vehicle.cur_task.aim = aim
                aim.task_vehicle.append(vehicle)
        # 每辆车判断资源分配
        for vehicle in self.vehicles:
            ratio = 0
            for task_vehicle in vehicle.task_vehicle:
                ratio += self.freqActions[task_vehicle.id]
            if ratio >= 1:
                for task_vehicle in vehicle.task_vehicle:
                    self.freqActions[task_vehicle.id] = 0.9 * round(self.freqActions[task_vehicle.id] / ratio, 2)
        # 每个mec判断资源分配
        for mec in self.MECs:
            ratio = 0
            for task_vehicle in mec.task_vehicle:
                ratio += self.freqActions[task_vehicle.id]
            if ratio > 1:
                for task_vehicle in mec.task_vehicle:
                    self.freqActions[task_vehicle.id] = 0.9 * round(self.freqActions[task_vehicle.id] / ratio, 2)
        # 分配资源
        for vehicle in self.vehicles:
            resource = vehicle.resources
            sum_resource = 0
            for task_vehicle in vehicle.task_vehicle:
                task = task_vehicle.cur_task
                compute_resource = self.freqActions[task_vehicle.id] * resource
                task.compute_resource = compute_resource
                # 没有资源给予惩罚
                if task.compute_resource == 0:
                    self.reward[task_vehicle.id] += PUNISH
                    # task_vehicle.flag = True
                    task_vehicle.cur_task = None
                    continue
                # print("车{}获得{}服务器{}资源".format(task_vehicle.id, vehicle.id, task.compute_resource))
                task.need_time = task.need_precess_cycle / task.compute_resource  # 记录任务需要计算时间(ms)
                if task.need_time >= 100:
                    # print("车{}任务超时".format(task_vehicle.id))
                    self.reward[task_vehicle.id] += PUNISH
                    task_vehicle.cur_task = None
                    continue
                # print("车{}需要计算时间{}".format(task_vehicle.id, task.need_time))
                self.avg_compute_time[task_vehicle.id].append(task.need_time)
                sum_resource += compute_resource
            vehicle.resources -= sum_resource
            if vehicle.resources < 0:
                print("出现错误")

        for mec in self.MECs:
            resource = mec.resources
            sum_resource = 0
            for task_vehicle in mec.task_vehicle:
                task = task_vehicle.cur_task
                compute_resource = self.freqActions[task_vehicle.id] * resource
                task.compute_resource = compute_resource
                # 没有资源给予惩罚
                if task.compute_resource == 0:
                    self.reward[task_vehicle.id] += PUNISH
                    task_vehicle.cur_task = None
                    # task.wait_time += time_solt
                    # task_vehicle.flag = True
                    continue
                # print("车{}获得{}服务器{}资源".format(task_vehicle.id, mec, task.compute_resource))
                task.need_time = task.need_precess_cycle / task.compute_resource  # 记录任务需要计算时间(ms)
                if task.need_time >= 100:
                    self.reward[task_vehicle.id] += PUNISH
                    # print("车{}任务超时".format(task_vehicle.id))
                    task_vehicle.cur_task = None
                    continue
                # print("车{}需要计算时间{}".format(task_vehicle.id, task.need_time))
                self.avg_compute_time[task_vehicle.id].append(task.need_time)
                sum_resource += compute_resource
            mec.resources -= sum_resource
            if mec.resources < 0:
                print("出现错误")

    def process_taskActions(self):
        """
        处理选择卸载目标和信道选择，计算占比选择
        """
        for i, vehicle in enumerate(self.vehicles):
            # 没有任务，无需执行卸载,也不给惩罚
            if vehicle.cur_task is None:
                continue
            bandAction = self.bandActions[i]
            task = vehicle.cur_task
            vehicle.Mec.pick_vehicle[bandAction].append(vehicle)
            task.pick_time = self.cur_frame % 1000

        # 计算路损
        self.compute_fade()
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.cur_task is not None and vehicle.cur_task.compute_resource > 0:  # and not vehicle.flag:
                task = vehicle.cur_task
                aim = task.aim
                band = self.bandActions[i]

                # 计算实时速率，用作奖励函数计算
                task.rate = self.compute_rate(vehicle, aim, POWER, band)
                if task.rate == 0:
                    task.trans_time = 0
                else:
                    task.trans_time = task.need_trans_size / task.rate
                if task.trans_time > 50:
                    self.reward[task.vehicle.id] += PUNISH
                    vehicle.cur_task = None
                    continue
                aim.accept_task.append(task)
                # print("车{}需要传输时间{}".format(vehicle.id, task.trans_time))
                self.avg_trans_time[task.vehicle.id].append(task.trans_time)
                # 卸载给本地 直接放到任务队列中
                # if vehicle == aim:
                #     vehicle.accept_task.append(task)
                # else:
                #     vehicle.queue_for_trans_vehicle.push(task)
                #     # 有任务在传输 放到传输等待队列中
                #     if vehicle.trans_task_for_vehicle == 0:
                #         vehicle.trans_task_for_vehicle = 1
                #         task.wait_time = 0
                #     else:
                #         last_task = vehicle.queue_for_trans_vehicle.getLast()
                #         task.wait_time = last_task.wait_time + last_task.need_trans_size / last_task.rate

    # 计算距离(车到车或者车到MEC)  aim：接受任务的目标
    @staticmethod
    def compute_distance(taskVehicle: Vehicle, aim):
        return round(np.sqrt(np.abs(taskVehicle.get_x - aim.get_x) ** 2 + np.abs(taskVehicle.get_y - aim.get_y) ** 2),
                     2)

    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight) ** 2 + dist_veh2bs ** 2
        pathLoss = 128.1 + 37.6 * np.log10(np.sqrt(dist2) / 1000)  # 路损公式中距离使用km计算
        combinedPL = -(np.random.randn() * self.stdV2I + pathLoss)
        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure

    def generate_fading_V2V(self, dist_DuePair):
        pathLoss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(self.freq)
        combinedPL = -(np.random.randn() * self.stdV2V + pathLoss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure

    def compute_fade(self):
        for vehicle in self.vehicles:
            if vehicle.cur_task is not None:
                task = vehicle.cur_task
                aim = task.aim
                if type(aim) == MEC:
                    distance = self.compute_distance(vehicle, aim)
                    fade = self.hwy_large_scale_channel_sampler.generate_fading_V2I(distance)
                else:
                    fade = self.hwy_large_scale_channel_sampler.generate_fading_V2V(vehicle.position, aim.position)
                fast_component = np.abs(np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
                vehicle.fade = fade + 20 * np.log10(fast_component)

    def compute_rate(self, vehicle: Vehicle, aim, power, band):
        """
        计算实时传输速率（在一个时隙内假设不变）
        """
        # print("vehicle:{} aim:{} ".format(vehicle.id, aim.id))
        if aim == vehicle:  # 本地计算
            return 0
        interference = 0
        for vehicle_i in vehicle.Mec.pick_vehicle[band]:
            if vehicle_i != vehicle:
                interference += 10 ** (vehicle_i.fade / 10)
        interference += sig2
        signals = 10 ** ((power + vehicle.fade + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        rate = round(vehicle.Mec.brand * np.log2(1 + np.divide(signals, interference)), 3) / 8
        print("第{}辆车速率:{} kb/ms".format(vehicle.id, rate))
        return rate  # kb/ms

    @staticmethod
    def compute_energy(trans_time, aim):
        return trans_time

    def get_reward(self, task):
        """
        计算此时任务的奖励
        """
        vehicle = task.vehicle
        aim = task.aim
        trans_time = task.trans_time

        # 计算时间
        compute_time = task.need_time
        sum_time = trans_time + compute_time
        self.avg_time[vehicle.id].append(sum_time)

        # if task.aim != vehicle:
        #     # 考虑通信消耗的能量（非本地卸载）
        #     energy = self.compute_energy(trans_time, aim)
        #     # print("传输需要{}ms".format(trans_time))
        #     # print("传输消耗{} J".format(energy))
        #     # 支付价格
        #     if type(task.aim) == MEC:
        #         price = MEC_Price * task.size
        #     else:
        #         price = VEC_Price * task.size
        #         # print("{}卸载给了邻居车".format(task.vehicle.id))
        # else:
        #     # 计算任务消耗的能量（本地卸载）
        #     price = 0
        #     energy = round(gama * task.need_time * task.compute_resource / 1000, 2)
        #     # print("本地计算消耗{} J".format(energy))
        # # if energy > 100:
        # #     energy /= 10
        # self.avg_price[vehicle.id].append(price)
        # self.avg_energy[vehicle.id].append(energy)
        reward = -sum_time / (task.size / 20)

        # if sum_time > task.max_time:
        #     reward += T2 * (sum_time - task.max_time) / 10
        # else:
        #     vehicle.success_task += 1
        return round(reward, 2)  # + 2.7

    def compute_rewards(self):
        """
        计算此时环境所有车辆的奖励
        """
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.cur_task is not None and vehicle.cur_task.compute_resource > 0:  # and not vehicle.flag:
                self.reward[i] = self.get_reward(vehicle.cur_task)
                vehicle.cur_task = None
            if self.reward[i] != 0:
                self.avg_reward[i].append(self.reward[i])

    # def distribute_task(self, cur_frame):
    #     """
    #     根据动作将任务添加至对应的列表当中  分配任务
    #     """
    #     for i, vehicle in enumerate(self.vehicles):
    #         time = cur_frame - self.cur_frame
    #         while not vehicle.queue_for_trans_vehicle.is_empty():
    #             task = vehicle.queue_for_trans_vehicle.peek()
    #             aim = task.aim
    #             needTime = task.need_trans_size / task.rate
    #             if time >= needTime:
    #                 time -= needTime
    #                 vehicle.queue_for_trans_vehicle.pop()
    #                 aim.accept_task.append(task)
    #             else:
    #                 task.need_trans_size -= time * task.rate
    #                 break
    #         if vehicle.queue_for_trans_vehicle.is_empty():
    #             self.vehicles[vehicle.id].trans_task_for_vehicle = 0

    # 更新资源信息并为处理完的任务计算奖励
    def renew_resources(self, cur_frame):
        """
        更新资源信息
        """
        time = cur_frame - self.cur_frame
        # 更新车的资源信息
        for i, vehicle in enumerate(self.vehicles):
            total_task = vehicle.accept_task

            if len(total_task) > 0:  # 此时有任务并且有剩余资源
                # 记录这个时隙能够处理完的任务
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    # 遍历此车的所有任务列表
                    precessed_time = task.need_precess_cycle / f
                    if precessed_time > time:
                        # 不能处理完
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                    else:
                        vehicle.resources += f
                vehicle.accept_task = retain_task

        # 更新mec的资源信息
        for mec in self.MECs:
            total_task = mec.accept_task
            if len(total_task) > 0:
                retain_task = []
                for task in total_task:
                    f = task.compute_resource
                    precessed_time = task.need_precess_cycle / f
                    if precessed_time > time:
                        task.need_precess_cycle -= f * time
                        retain_task.append(task)
                    else:
                        mec.resources += f
                mec.accept_task = retain_task
        # 分配任务信息（在计算之后执行是因为将一个时隙看作为原子操作，因此这个时隙接受到的任务不能进行计算）
        # self.distribute_task(cur_frame=cur_frame)

    # 更新车辆位置：renew_position(无)，遍历每辆车，根据其方向和速度更新位置，
    def renew_positions(self, cur_frame):
        time = (cur_frame - self.cur_frame)  # ms
        for i in range(len(self.vehicles)):
            vehicle = self.vehicles[i]
            vehicle.clear()
            s = vehicle.velocity * time / 1000
            vehicle.loc_y += s * vehicle.direction
            # 到达出口  掉头
            if vehicle.loc_y >= 3000 or vehicle.loc_y <= 0:
                if vehicle.loc_y >= 3000:
                    vehicle.loc_y = 3000
                else:
                    vehicle.loc_y = 0
                index = np.random.randint(0, 2)
                if vehicle.direction == 1:
                    vehicle.loc_x = self.left_lanes[index]
                else:
                    vehicle.loc_x = self.right_lanes[index]
                vehicle.direction = -vehicle.direction
            vehicle.position = [vehicle.loc_x, vehicle.loc_y]

    def renew_state(self, cur_frame):
        """
        更新状态
        """
        self.vehicleState = []
        self.mecState = []

        # 更新车位置信息
        self.renew_positions(cur_frame)
        # 更新邻居表
        self.renew_neighbor()
        # 更新mec
        self.renew_mec()
        for vehicle in self.vehicles:
            # 更新时间
            vehicle.cur_frame = cur_frame
            # 产生任务
            vehicle.create_work()
            # 更新资源已经接受的任务信息
            self.vehicleState.append(vehicle.get_state())
        for vehicle in self.vehicles:
            self.neighborState.append(vehicle.get_neighbor_states())

        for mec in self.mecState:
            self.mecState.append(mec.get_state())

    # 执行动作
    def step(self, bandActions, offloadingActions, freqActions):
        cur_frame = self.cur_frame + time_solt  # ms
        # 分配动作
        # 卸载动作
        self.offloadingActions = offloadingActions
        self.bandActions = bandActions
        self.freqActions = freqActions

        # 重置奖励
        self.reward = [0] * self.num_Vehicles

        # 判断比率分配动作
        self.process_fracActions()
        # 处理选取任务动作
        self.process_taskActions()

        # 计算奖励
        self.compute_rewards()

        # 记录当前状态
        vehicleState = self.vehicleState
        mecState = self.mecState
        neighbor_state = self.neighborState

        # 更新资源信息以及车辆任务信息
        self.renew_resources(cur_frame)

        # 更新状态
        self.renew_state(cur_frame)

        # 更新时间
        self.cur_frame = cur_frame
        # print("当前有{}个任务没传输完成".format(len(self.need_trans_task)))

        # 平均奖励
        self.Reward = np.mean([reward for i, reward in enumerate(self.reward) if i % 4 != 0])
        return vehicleState, neighbor_state, self.vehicleState, self.neighborState, self.Reward, self.reward


class HwyChannelLargeScaleFadingGenerator:
    def __init__(self, stdV2I, stdV2V, vehHeight, bsHeight, freq, vehAntGain, bsAntGain, bsNoiseFigure,
                 vehNoiseFigure):
        self.stdV2I = stdV2I
        self.stdV2V = stdV2V
        self.vehHeight = vehHeight
        self.bsHeight = bsHeight
        self.freq = freq
        self.vehAntGain = vehAntGain
        self.bsAntGain = bsAntGain
        self.bsNoiseFigure = bsNoiseFigure
        self.vehNoiseFigure = vehNoiseFigure

    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight) ** 2 + dist_veh2bs ** 2
        pathLoss = 128.1 + 37.6 * np.log10(np.sqrt(dist2) / 1000)  # 路损公式中距离使用km计算
        combinedPL = -pathLoss
        return combinedPL

    def generate_fading_V2V(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        # d_bp = 4 * (1.5 - 1) * (1.5 - 1) * self.freq * (10 ** 9) / (3 * 10 ** 8)

        # 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(2)
        Pro_LOS = min(1, 1.05 * np.exp(-0.0114 * d))
        if np.random.uniform(0, 1) < Pro_LOS:
            PL_LOS = 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(self.freq)
            combinedPL = -PL_LOS
            # combinedPL = - PL_LOS
            return combinedPL
            # + self.vehAntGain * 2 - self.vehNoiseFigure
        else:
            PL_NOS = 36.85 + 30 * np.log10(d) + 18.9 * np.log10(self.freq)
            combinedPL = -PL_NOS
            # combinedPL = -PL_NOS
            return combinedPL
