# -*- coding: utf-8 -*-
import os
import time
from collections import namedtuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pylab import mpl

import model
from env import Env

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 加载 Times New Roman 字体
font_path = 'C:/Windows/Fonts/times.ttf'
prop = fm.FontProperties(fname=font_path, size=8)

Experience = namedtuple('Transition',
                        field_names=['cur_otherState', "cur_NeighborState", "all_vehicle_state",  # 状态
                                     'bandAction', 'aimAction', 'freqAction',  # 动作
                                     'reward',  # 奖励
                                     'next_otherState', 'next_NeighborState',
                                     "next_all_vehicle_state"])  # Define a transition tuple

REPLAY_SIZE = 100000
LEARNING_RATE_Actor = 1e-5
LEARNING_RATE_Critic = 1e-4
GAMMA = 0.9
BATCH_SIZE = 64
REPLAY_INITIAL = 1000
TARGET_STEPS = 10

EPSILON = 200000

EPSILON_DECAY_LAST_FRAME = 100000
EPSILON_START = 0.6
EPSILON_FINAL = 0.01

RESOURCE = [0.2, 0.4, 0.5, 0.6, 0.8]
BandNum = 5  # 带宽选择
FreqNum = len(RESOURCE)  # 比率选择


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
            actionFreq.append(RESOURCE[np.argmax(freqAction)])

    vehicleState, neighbor_state, all_vehicleState, neighborState, Reward, reward = env.step(actionBand, actionAim,
                                                                                             actionFreq)
    # 加入各自的缓存池【当前其他状态、当前任务状态、目标动作、任务动作，下一其他状态、下一任务状态】
    for j, vehicle in enumerate(vehicles):
        exp = Experience(old_otherState[j], [old_neighborState[j]], [old_all_vehicle_state],
                         [actionBand[j]], [actionAim[j]], [actionFreq[j]], Reward,
                         vehicleState[j], [neighbor_state[j]], [all_vehicleState])
        vehicle.buffer.append(exp)
    return round(Reward, 5)  # 返回总的平均奖励


# 将经验转换成torch
def unpack_batch_ddpg(batch, device='cpu'):
    vehicle_state, neighbor_state, all_vehicle_state, \
    band_action, aim_action, freq_action, reward, \
    next_vehicle_state, next_neighbor_state, next_all_vehicle_state = batch
    vehicle_state_v = torch.tensor(vehicle_state, dtype=torch.float32).to(device)
    neighbor_state_v = torch.tensor(neighbor_state, dtype=torch.float32).to(device)
    all_vehicle_state_v = torch.tensor(all_vehicle_state, dtype=torch.float32).to(device)
    band_action_v = torch.tensor(band_action, dtype=torch.float32).to(device)
    aim_action_v = torch.tensor(aim_action, dtype=torch.float32).to(device)
    freq_action_v = torch.tensor(freq_action, dtype=torch.float32).to(device)
    reward_v = torch.tensor(reward, dtype=torch.float32).to(device)
    next_vehicle_state_v = torch.tensor(next_vehicle_state, dtype=torch.float32).to(device)
    next_neighbor_state_v = torch.tensor(next_neighbor_state, dtype=torch.float32).to(device)
    next_all_vehicle_state_v = torch.tensor(next_all_vehicle_state, dtype=torch.float32).to(device)

    return vehicle_state_v, neighbor_state_v, all_vehicle_state_v, \
           band_action_v, aim_action_v, freq_action_v, reward_v, \
           next_vehicle_state_v, next_neighbor_state_v, next_all_vehicle_state_v


if __name__ == '__main__':
    env = Env()
    env.reset()

    vehicles = env.vehicles
    actor_models = []
    actor_target_models = []
    actor_optimizers = []
    critic_models = []
    critic_target_models = []
    critic_optimizers = []

    # 初始化网络
    AIM_DIM = len(vehicles[0].neighbor) + 2
    vehicle_shape = len(vehicles[0].self_state)
    neighbor_shape = np.array([vehicles[0].neighbor_state]).shape
    all_vehicles_shape = np.array([env.vehicleState]).shape
    for vehicle in vehicles:
        actor_model = model.ModelActor(vehicle_shape, neighbor_shape, BandNum, AIM_DIM, FreqNum)
        target_actor_model = model.TargetNet(actor_model)
        actor_optimizer = optim.Adam(actor_model.parameters(), LEARNING_RATE_Actor)

        critic_model = model.ModelCritic(all_vehicles_shape, 1, 1, 1)
        target_critic_model = model.TargetNet(critic_model)
        critic_optimizer = optim.Adam(critic_model.parameters(), LEARNING_RATE_Critic)

        actor_models.append(actor_model)
        actor_target_models.append(target_actor_model)
        actor_optimizers.append(actor_optimizer)
        critic_models.append(critic_model)
        critic_target_models.append(target_critic_model)
        critic_optimizers.append(critic_optimizer)

    time_solt = 0
    epsilon = EPSILON

    total_reward = []
    recent_reward = []
    while epsilon > 0:
        time_solt += 1
        print("the {} step".format(time_solt))
        # 执行一步
        eps = max(EPSILON_FINAL, EPSILON_START - time_solt / EPSILON_DECAY_LAST_FRAME)
        reward = play_step(env, eps, actor_models)
        total_reward.append(reward)
        print("current reward:", reward)
        print("current 100 times total rewards:", np.mean(total_reward[-100:]))
        recent_reward.append(np.mean(total_reward[-100:]))
        # if epsilon % 100000 == 0:
        #     env.reset()
        # if np.mean(total_reward[-100:]) > 0.7:
        #     break

        for i, vehicle in enumerate(vehicles):
            if len(vehicle.buffer) < REPLAY_INITIAL:
                continue
            # 从经验池中选取经验
            batch = vehicle.buffer.sample(BATCH_SIZE)
            vehicle_state_v, neighbor_state_v, all_vehicle_state_v, \
            band_action_v, aim_action_v, freq_action_v, reward_v, \
            next_vehicle_state_v, next_neighbor_state_v, next_all_vehicle_state_v = unpack_batch_ddpg(batch=batch)

            # train critic
            critic_optimizers[i].zero_grad()
            # 计算q
            band_q_v, aim_q_v, freq_q_v = critic_models[i](all_vehicle_state_v, band_action_v, aim_action_v,
                                                           freq_action_v)
            next_band_action, next_aim_action, next_freq_action = actor_target_models[i].target_model(
                next_vehicle_state_v, next_neighbor_state_v)
            # next_task_action = next_task_a_dist.sample()
            # next_task_action = next_task_action.unsqueeze(1)
            # next_aim_action = next_aim_a_v_dist.sample()
            # next_aim_action = next_aim_action.unsqueeze(1)
            next_band_action = torch.argmax(next_band_action, dim=1).unsqueeze(1)
            next_aim_action = torch.argmax(next_aim_action, dim=1).unsqueeze(1)
            next_freq_action = torch.argmax(next_freq_action, dim=1).unsqueeze(1)
            # 计算q‘
            next_band_q_v, next_aim_q_v, next_freq_q_v = critic_target_models[i].target_model(next_all_vehicle_state_v,
                                                                                              next_band_action,
                                                                                              next_aim_action,
                                                                                              next_freq_action)
            band_q_ref_v = reward_v.unsqueeze(dim=-1) + next_band_q_v * GAMMA
            aim_q_ref_v = reward_v.unsqueeze(dim=-1) + next_aim_q_v * GAMMA
            freq_q_ref_v = reward_v.unsqueeze(dim=-1) + next_freq_q_v * GAMMA
            band_loss_v = F.mse_loss(band_q_v, band_q_ref_v.detach())
            aim_loss_v = F.mse_loss(aim_q_v, aim_q_ref_v.detach())
            freq_loss_v = F.mse_loss(freq_q_v, freq_q_ref_v.detach())
            torch.autograd.backward([band_loss_v, aim_loss_v, freq_loss_v])
            critic_optimizers[i].step()

            # train actor
            actor_optimizers[i].zero_grad()
            cur_band_action_v, cur_aim_action_v, cur_freq_action_v = actor_models[i](vehicle_state_v, neighbor_state_v)
            band_action = torch.argmax(cur_band_action_v, dim=1).unsqueeze(1)
            aim_action = torch.argmax(cur_aim_action_v, dim=1).unsqueeze(1)
            freq_action = torch.argmax(cur_freq_action_v, dim=1).unsqueeze(1)
            actor_band_loss, actor_aim_loss, actor_freq_loss = critic_models[i](all_vehicle_state_v, band_action,
                                                                                aim_action,
                                                                                freq_action)
            actor_band_loss = -actor_band_loss.mean()
            actor_aim_loss = -actor_aim_loss.mean()
            actor_freq_loss = -actor_freq_loss.mean()
            torch.autograd.backward([actor_band_loss, actor_aim_loss, actor_freq_loss])
            actor_optimizers[i].step()

            # 目标网络软更新
            if time_solt % TARGET_STEPS == 0:
                critic_target_models[i].alpha_sync(alpha=1 - 1e-3)
                actor_target_models[i].alpha_sync(alpha=1 - 1e-3)
        epsilon -= 1

        if time_solt % 50000 == 0 and time_solt != 0:
            cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())) + "-" + str(time_solt)
            # 创建文件夹
            os.makedirs("D:/pycharm/Project/VML/Experience/experiment3/result/" + cur_time)
            for i, vehicle in enumerate(env.vehicles):
                # 保存每个网络模型
                torch.save(actor_target_models[i].target_model.state_dict(),
                           "D:/pycharm/Project/VML/Experience/experiment3/result/" + cur_time + "/vehicle" + str(
                               i) + ".pkl")
    cur_time = time.strftime("%Y-%m-%d-%H", time.localtime(time.time())) + "-" + str(epsilon)
    array = np.array(recent_reward)
    np.save('data/' + cur_time, array)

    plt.plot(range(len(recent_reward)), recent_reward)
    plt.title("当前最近100次奖励曲线")
    plt.show()

    plt.plot(range(len(total_reward)), total_reward)
    plt.title("奖励曲线")
    plt.show()
