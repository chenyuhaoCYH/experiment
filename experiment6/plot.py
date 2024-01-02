import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # array = np.load("data/2023-12-23-23-200000.npy")
    # plt.plot(-array[500:])
    # ppo = np.load("data/2023-12-26-21-200000.npy")
    # plt.plot(-ppo[500:], color='r')
    dpn = np.load("data/2024-01-02-16-0.npy")
    plt.plot(dpn, color='y')
    plt.show()
