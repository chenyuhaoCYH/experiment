# -*- coding: utf-8 -*-
import os
import time
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pylab import mpl
import matplotlib.font_manager as fm
import netron

from env import Env
from model import *

np.random.seed(2)

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 加载 Times New Roman 字体
font_path = 'C:/Windows/Fonts/times.ttf'
prop = fm.FontProperties(fname=font_path, size=8)

Experience = namedtuple('Transition',
                        field_names=['cur_otherState', "cur_NeighborState",  # 状态
                                     'bandAction', 'aimAction', 'freqAction',  # 动作
                                     'reward',  # 奖励
                                     'next_otherState', 'next_NeighborState'])  # Define a transition tuple
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 1000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 100  # 更新目标网络频率

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.8
EPSILON_FINAL = 0.01
EPSILON = 200000

RESET = 1000  # 重置游戏次数

BandNum = 5  # 带宽选择
FreqNum = 6  # 比率选择

momentum = 0.005

RESOURCE = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]


@torch.no_grad()
def play_step(env, epsilon, models):
    vehicles = env.vehicles
    old_otherState = []
    old_neighborState = []

    actionFreq = []
    actionBand = []
    actionAim = []
    # 贪心选择动作
    for i, model in enumerate(models):
        old_otherState.append(vehicles[i].self_state)
        old_neighborState.append(vehicles[i].neighbor_state)
        if np.random.random() < epsilon:
            # 随机动作
            actionBand.append(np.random.randint(0, 5))
            actionFreq.append(RESOURCE[np.random.randint(0, len(RESOURCE))])
            actionAim.append(np.random.randint(0, 7))  # local+mec+neighbor
        else:
            state_v = torch.tensor([vehicles[i].self_state], dtype=torch.float32)
            neighborState_v = torch.tensor([[vehicles[i].neighbor_state]], dtype=torch.float32)
            bandActionValue, aimActionValue, freqActionValue = model(state_v, neighborState_v)

            bandAction = np.array(bandActionValue, dtype=np.float32).reshape(-1)
            aimAction = np.array(aimActionValue, dtype=np.float32).reshape(-1)
            freqAction = np.array(freqActionValue, dtype=np.float32).reshape(-1)

            actionAim.append(np.argmax(aimAction))
            actionBand.append(np.argmax(bandAction))
            actionFreq.append(np.argmax(freqAction))
    # print("action:", action)
    _, _, otherState, neighborState, Reward, reward = env.step(actionBand, actionAim, actionFreq)
    # print("reward:", reward)

    for i, vehicle in enumerate(vehicles):
        exp = Experience(old_otherState[i], [old_neighborState[i]],
                         actionBand[i], actionAim[i], actionFreq[i],
                         reward[i],
                         otherState[i], [neighborState[i]])
        vehicle.buffer.append(exp)
    return round(Reward, 2)  # 返回总的平均奖励


# 计算一个智能体的损失
def calc_loss(batch, net: DQNCNN, tgt_net: DQNCNN, device="cpu"):
    cur_otherState, curNeighborState, bandAction, aimAction, freqAction, rewards, next_otherState, next_NeighborState = batch  #

    otherStates_v = torch.tensor(np.array(cur_otherState, copy=False), dtype=torch.float32).to(device)
    neighborStates_v = torch.tensor(np.array(curNeighborState, copy=False), dtype=torch.float32).to(device)
    # print("states_v:", states_v)  # batch状态
    aimActions_v = torch.tensor(np.array(aimAction), dtype=torch.int64).to(device)
    bandAction_v = torch.tensor(np.array(bandAction), dtype=torch.int64).to(device)
    freqAction_v = torch.tensor(np.array(freqAction), dtype=torch.int64).to(device)
    # print("actions_v", actions_v)  # batch动作
    rewards_v = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    # print("rewards_v", rewards_v)  # batch奖励
    next_otherStates_v = torch.tensor(np.array(next_otherState, copy=False), dtype=torch.float32).to(device)
    next_NeighborState_v = torch.tensor(np.array(next_NeighborState, copy=False), dtype=torch.float32).to(device)
    # print("next_states_v", next_states_v)  # batch下一个状态

    # 计算当前网络q值
    bandActionValue, aimActionValue, freqActionValue = net(otherStates_v,
                                                           neighborStates_v)  # .gather(1, aimActions_v.unsqueeze(-1)).squeeze(-1)
    aimActionValues = aimActionValue.gather(1, aimActions_v.unsqueeze(-1)).squeeze(-1)
    bandActionValues = bandActionValue.gather(1, bandAction_v.unsqueeze(-1)).squeeze(-1)
    freqActionValues = freqActionValue.gather(1, freqAction_v.unsqueeze(-1)).squeeze(-1)

    # 计算目标网络q值
    next_bandActionValue, next_aimActionValue, next_freqActionValue = tgt_net(next_otherStates_v,
                                                                              next_NeighborState_v)  # .max(1)[0]  # 得到最大的q值

    next_bandActionValues = next_bandActionValue.max(1)[0].detach()
    next_aimActionValues = next_aimActionValue.max(1)[0].detach()
    next_freqActionValues = next_freqActionValue.max(1)[0].detach()

    # 防止梯度流入用于计算下一状态q近似值得NN
    # next_states_values = next_aimActionValues.detach()
    # print("next_states_values", next_states_values)
    expected_band_values = next_bandActionValues * GAMMA + rewards_v
    expected_aim_values = next_aimActionValues * GAMMA + rewards_v
    expected_freq_values = next_freqActionValues * GAMMA + rewards_v
    # print(" expected_state_values", expected_state_values)

    return nn.MSELoss()(bandActionValues, expected_band_values), \
           nn.MSELoss()(aimActionValues, expected_aim_values), \
           nn.MSELoss()(freqActionValues, expected_freq_values)


if __name__ == '__main__':
    env = Env()
    env.reset()

    frame_idx = 0
    # writer = SummaryWriter(comment="-" + env.__doc__)
    agents = env.vehicles
    models = []
    tgt_models = []
    optimizers = []
    neighbor_shape = np.array([agents[0].neighbor_state]).shape
    for agent in agents:
        # print(agent.get_location, agent.velocity)

        # print(task_shape)
        model = DQNCNN(len(agent.self_state), neighbor_shape, BandNum, len(agent.neighbor) + 2, FreqNum)
        models.append(model)
        optimer = optim.RMSprop(params=model.parameters(), lr=LEARNING_RATE, momentum=momentum)
        optimizers.append(optimer)
    for agent in agents:
        # print(agent.get_location, agent.velocity)
        # task_shape = np.array([agent.task_state]).shape
        # print(task_shape)
        model = DQNCNN(len(agent.self_state), neighbor_shape, BandNum, len(agent.neighbor) + 2, FreqNum)
        model.load_state_dict(models[agent.id].state_dict())
        tgt_models.append(model)

    # 打印网络结构
    # model = models[0]
    # state_v = torch.tensor([env.vehicles[0].self_state], dtype=torch.float32)
    # taskState_v = torch.tensor([[env.vehicles[0].task_state]], dtype=torch.float32)
    # neighbor_v = torch.tensor([[env.vehicles[0].neighbor_state]], dtype=torch.float32)
    # # 针对有网络模型，但还没有训练保存 .pth 文件的情况
    # modelpath = "./netStruct/demo.onnx"  # 定义模型结构保存的路径
    # torch.onnx.export(model, (state_v, taskState_v, neighbor_v), modelpath)  # 导出并保存
    # netron.start(modelpath)

    total_reward = []
    recent_reward = []
    loss_aim_list = []
    reward_1 = []

    epsilon = EPSILON_START
    eliposde = EPSILON
    while eliposde > 0:
        frame_idx += 1
        # 重置游戏
        # if frame_idx % RESET == 0:
        #     print("游戏重置")
        #     # memory = []
        #     # for vehicle in env.vehicles:
        #     #     memory.append(vehicle.buffer)
        #     env.reset()
        #     agents = env.vehicles
        #     # for i, vehicle in enumerate(agents):
        #     #     vehicle.buffer = memory[i]
        print("the {} steps".format(frame_idx))
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, epsilon, models)
        total_reward.append(reward)
        print("current reward:", reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        recent_reward.append(-np.mean(total_reward[-100:]))
        # if np.mean(total_reward[-100:]) > 0.7:
        #     break

        for i, agent in enumerate(agents):
            # print("length of {} buffer".format(agent.id), len(agent.buffer))
            if len(agent.buffer) < REPLAY_SIZE:  # 缓冲池要足够大
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:  # 更新目标网络
                tgt_models[i].load_state_dict(models[i].state_dict())
            optimizers[i].zero_grad()
            batch = agent.buffer.sample(BATCH_SIZE)
            loss_band, loss_aim, loss_freq = calc_loss(batch, models[i], tgt_models[i])
            if i == 2:
                print("loss:", loss_band, " ", loss_aim, " ", loss_freq)
            # loss_t.backward()
            torch.autograd.backward([loss_band, loss_aim, loss_freq])
            # total_loss = 0.6 * loss_aim + 0.4 * loss_task
            optimizers[i].step()
        eliposde -= 1
        if frame_idx % 10000 == 0 and frame_idx != 0:
            cur_time = time.strftime("%Y-%m-%d-%H", time.localtime(time.time())) + "-" + str(frame_idx)
            # 创建文件夹
            os.makedirs("D:/pycharm/Project/VML/Experience/experiment/result/" + cur_time)
            for i, vehicle in enumerate(env.vehicles):
                # 保存每个网络模型
                torch.save(tgt_models[i].state_dict(),
                           "D:/pycharm/Project/VML/Experience/experiment/result/" + cur_time + "/vehicle" + str(
                               i) + ".pkl")
    cur_time = time.strftime("%Y-%m-%d-%H", time.localtime(time.time())) + "-" + str(frame_idx)
    array = np.array(recent_reward)
    np.save('data/' + cur_time, array)
    plt.plot(range(len(recent_reward) - 500), recent_reward[500:])
    # plt.plot(range(len(total_reward)), total_reward)
    plt.ylabel("Average Reward", fontproperties=prop)
    plt.xlabel("Episode", fontproperties=prop)
    plt.show()
