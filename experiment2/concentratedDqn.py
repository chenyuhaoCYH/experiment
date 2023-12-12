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
from model import DQN, DQNCNN

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
BATCH_SIZE = 32
REPLAY_SIZE = 1000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 100  # 更新目标网络频率

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.8
EPSILON_FINAL = 0.01
EPSILON = 400000

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
