import os

import numpy as np
import matplotlib.pyplot as plt

from env import Env

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.random.seed(2)
if __name__ == '__main__':
    print()
    frac = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    env = Env()
    env.reset()

    # 测试网络节点数
    # task = np.array(env.taskState)
    # print(task.shape)
    vehicles = env.vehicles

    # for vehicle in vehicles:
    #     print("第{}车状态：{}".format(vehicle.id, vehicle.self_state))
    #     print("该车邻居:")
    #     for i in vehicle.neighbor:
    #         print(i.id, end="  ")
    #     print()

    # 测试环境运行
    reward = []
    models = []
    position = [[] for i in vehicles]

    for step in range(1000):
        offloadingActions = []
        bandActions = []
        freqActions = []
        for i in range(env.num_Vehicles):
            offloadingActions.append(np.random.randint(0, 7))
            bandActions.append(np.random.randint(0, 5))
            freqActions.append(frac[np.random.randint(0, 6)])
        Reward, _, _, _, _, _ = env.step(bandActions, offloadingActions, freqActions)
        for vehicle in vehicles:
            position[vehicle.id].append(vehicle.position)
        # reward.append(Reward)
        # print("第{}次平均奖励{}".format(step, Reward))
        # print("当前状态:", state)
        # print("下一状态:", next_state)
        # print("车状态:", vehicleState)
        # print("任务状态", taskState)
        # print("当前奖励:", reward)
        # print("每个奖励,", vehicleReward)
        # print("当前有{}任务没有传输完成".format(len(env.need_trans_task)))
        # print("average reward:", env.Reward)

    # for i, p in enumerate(position):
    #     print("id: ", i)
    #     print(p)

    # plt.figure()
    # fix, ax = plt.subplots(5, 4)
    #
    # for i in range(5):
    #     for j in range(4):
    #         number = i * 4 + j
    #         ax[i, j].plot(x[number], y[number])
    #         ax[i, j].set_title('vehicle {}'.format(number))
    # plt.plot(range(len(reward)), reward)
    # plt.ylabel("Reward")
    # plt.show()

    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_time)]
    plt.ylabel("sumTime")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()

    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_trans_time)]
    plt.ylabel("transTime")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()

    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_compute_time)]
    plt.ylabel("computeTime")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()
    #
    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_reward)]
    plt.ylabel("avg_reward")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()
    #
    plt.figure()
    avg = [np.mean(avg_energy) for i, avg_energy in enumerate(env.avg_energy)]
    plt.ylabel("Energy")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()
    # #
    plt.figure()
    avg = [np.mean(sum_time) for i, sum_time in enumerate(env.avg_price)]
    plt.ylabel("Price")
    plt.bar(range(len(avg)), avg, color="blue")
    plt.show()
    #
    # plt.figure()
    # avg = [vehicle.success_task / vehicle.sum_create_task for i, vehicle in enumerate(env.vehicles)]
    # plt.ylabel("successRate")
    # plt.bar(range(len(avg)), avg, color="blue")
    # plt.show()
