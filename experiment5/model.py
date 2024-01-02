# import ptan
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

HID_SIZE = 64
HID_SIZE_MIN = 32


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 软更新
    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class ModelActor(nn.Module):
    def __init__(self, obs_dim, neighbor_dim, bankAction_dim, aimAction_dim, freqActions_dim):
        super(ModelActor, self).__init__()

        self.cnn_neighbor = CNNLayer(neighbor_dim, HID_SIZE_MIN)
        self.same = nn.Sequential(
            nn.Linear(HID_SIZE_MIN + obs_dim, 2 * HID_SIZE),
            nn.ReLU(),
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 2 * HID_SIZE),
            nn.ReLU(),
        )
        self.bank = nn.Sequential(
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, bankAction_dim),
        )
        self.aim = nn.Sequential(
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, aimAction_dim),
        )
        self.freq = nn.Sequential(
            nn.Linear(2 * HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, freqActions_dim),
        )
        self.logstd_bank = nn.Parameter(torch.zeros(bankAction_dim))
        self.logstd_freq = nn.Parameter(torch.zeros(freqActions_dim))
        self.logstd_aim = nn.Parameter(torch.zeros(aimAction_dim))

    def forward(self, obs, neighbor, is_train=True):
        neighbor_out = self.cnn_neighbor(neighbor)
        x = torch.cat((neighbor_out, obs), -1)
        same_out = self.same(x)
        bank_out = self.bank(same_out)
        aim_out = self.aim(same_out)
        freq_out = self.freq(same_out)
        if is_train:
            rnd_bank = torch.tensor(np.random.normal(size=bank_out.shape))
            rnd_aim = torch.tensor(np.random.normal(size=aim_out.shape))
            rnd_freq = torch.tensor(np.random.normal(size=freq_out.shape))
            bank_out = bank_out + torch.exp(self.logstd_bank) * rnd_bank
            aim_out = aim_out + torch.exp(self.logstd_aim) * rnd_aim
            freq_out = freq_out + torch.exp(self.logstd_freq) * rnd_freq

        # act_out = F.gumbel_softmax(act_out)
        bank_pro = F.softmax(bank_out, dim=-1)
        aim_pro = F.softmax(aim_out, dim=-1)
        freq_pro = F.softmax(freq_out, dim=-1)
        # print(act_pro)
        # print(torch.sum(act_pro))
        # print(task_pro)
        # return act_pro, task_pro  # 打印网络结构用
        return bank_pro, aim_pro, freq_pro  # 真实使用


class ModelCritic(nn.Module):
    def __init__(self, obs_size, bank_action, aim_action, freq_action):
        super(ModelCritic, self).__init__()

        self.cnn = CNNLayer(obs_size, HID_SIZE)

        self.bank_value = nn.Sequential(
            nn.Linear(HID_SIZE + bank_action, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )
        self.aim_value = nn.Sequential(
            nn.Linear(HID_SIZE + aim_action, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )
        self.freq_value = nn.Sequential(
            nn.Linear(HID_SIZE+ freq_action, HID_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HID_SIZE * 2, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE_MIN),
            nn.ReLU(),
            nn.Linear(HID_SIZE_MIN, 1),
        )

    def forward(self, states_v, bank_actions_v, aim_action_v, freq_action_v):
        cnn_out = self.cnn(states_v)

        bank_value = self.bank_value(torch.cat((cnn_out, bank_actions_v), -1))
        aim_value = self.aim_value(torch.cat((cnn_out, aim_action_v), -1))
        freq_value = self.freq_value(torch.cat((cnn_out, freq_action_v), -1))
        return bank_value, aim_value, freq_value  # , aim_value


class ModelSACTwinQ(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelSACTwinQ, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)


"""

class AgentDDPG(ptan.agent.BaseAgent):
    """"""
    Agent implementing Orstein-Uhlenbeck exploration process
    """"""

    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(
                    size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states

"""


class DQNCNN(nn.Module):
    def __init__(self, obs_dim, neighbor_dim, bankAction_dim, aimAction_dim, freqActions_dim):
        super(DQNCNN, self).__init__()
        self.input_layer = nn.Linear(obs_dim + 32, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.cnn2 = CNNLayer(neighbor_dim, 32)
        self.output_layer1 = self.common(64, bankAction_dim)
        self.output_layer2 = self.common(64, aimAction_dim)
        self.output_layer3 = self.common(64, freqActions_dim)

    def common(self, input_dim, action_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            self.hidden1,
            nn.ReLU(),
            self.hidden2,
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, neighbor):
        """

        :param x: batch_size*state_n
        :return: batch_size*actions_n  输出每个动作对应的q值
        """
        # 任务卷积层
        cnn_out2 = self.cnn2(neighbor)
        x = torch.cat((x, cnn_out2), -1)

        # 公共层
        x1 = F.relu(self.input_layer(x))
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))

        bandActionValue = self.output_layer1(x3)
        aimActionValue = self.output_layer2(x3)
        freqActionValue = self.output_layer3(x3)

        return bandActionValue, aimActionValue, freqActionValue


class DQN(nn.Module):
    def __init__(self, obs_dim, task_dim, taskAction_dim, aimAction_dim):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(obs_dim + 32, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 128)
        self.cnn = CNNLayer(task_dim, 32)
        self.output_layer1 = self.common(64, taskAction_dim)
        self.output_layer2 = self.common(64, aimAction_dim)

    def common(self, input_dim, action_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            self.hidden1,
            nn.ReLU(),
            self.hidden2,
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x, task):
        """

        :param x: batch_size*state_n
        :return: batch_size*actions_n  输出每个动作对应的q值
        """
        # 任务卷积层
        cnn_out = self.cnn(task)
        x = torch.cat((x, cnn_out), -1)

        # 公共层
        x1 = F.relu(self.input_layer(x))
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))

        taskActionValue = self.output_layer1(x3)
        aimActionValue = self.output_layer2(x3)

        return taskActionValue, aimActionValue


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal=True, use_ReLU=True, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):  # 权重使用正交初始化，激活函数使用relu
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            nn.Flatten(),
            init_(nn.Linear(
                hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                hidden_size)
            ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        x = self.cnn(x)

        return x


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
