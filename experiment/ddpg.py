# -*- coding: utf-8 -*-
import os
import time
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pylab import mpl
import matplotlib.font_manager as fm
import netron

from env import Env
import model

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

REPLAY_SIZE = 100000
LEARNING_RATE_Actor = 1e-5
LEARNING_RATE_Critic = 1e-4
GAMMA = 0.9
BATCH_SIZE = 64
REPLAY_INITIAL = 10000
TARGET_STEPS = 10

EPSILON = 400000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 0.6
EPSILON_FINAL = 0.01

RESOURCE = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]


@torch.no_grad()
def play_step(env: Env, epsilon, models: list):
    vehicles = env.vehicles
    old_otherState = []
    old_neighborState = []
    old_all_vehicle_state = env.vehicleState

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

        vehicleState, neighbor_state, vehicleState, neighborState, Reward, reward = env.step(actionBand, actionAim,
                                                                                             actionFreq)
