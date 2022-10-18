from py_interface import *
from ctypes import *
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--output_dir', type=str,
                    default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')


N_STATES = 3    # 三种流量状态
ACTIONS = ['FTP', 'VOICE', 'UDP']   # 三个模型
EPSILON = 0.7   # 随机选取率
ALPHA = 0.01     # 学习率
GAMMA = 0.9     # 抽取
MAX_EPISODES = 20   # 最大轮次
FRESH_TIME = 0.5    # 更新时间


#   从ns-3中获取数据  其运用接口参照ns3-ai接口或者ns3-gym
class RlEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ('test_packet', c_uint32),
        ('packetsize', c_uint32),
        ('node_a', c_uint32),
        ('node_b', c_uint32),
        ('num_cell', c_uint32),
        ('palce', c_uint32),
        ('speed', c_uint32),
        ('bytesInFlight', c_uint32),
        ('id', c_uint32),
        ('nodeId', c_uint32),
        ('socketUid', c_uint32),
        ('envType', c_uint8),
        ('simTime_us', c_int64),
        ('ssThresh', c_uint32),
        ('cWnd', c_uint32),
        ('segmentSize', c_uint32),
        ('segmentsAcked', c_uint32),
    ]


class RlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('id', c_uint32),
        ('nexts', c_uint32)
    ]



def build_q_table(n_state, actions):
    table = pd.DataFrame(
        np.zeros((n_state, len(actions))),
        columns=actions,
    )
    # print(table)
    return table

# ------------------------------------
#     根据状态和q-table选择下一步的动作
# ------------------------------------


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 取一个状态的一个列表
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()

    return action_name

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
        )

    def forward(self, x):
        return self.layers(x)

# 强化学习算法模型
class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.batchsize = 32
        self.observer_shape = 5
        self.target_replace = 100
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((2000, 2*5+2))    # s, a, r, s'
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > 0.99 ** self.memory_counter:    # choose best
            action = self.eval_net.forward(x)
            action = torch.argmax(action, 0).numpy()
        else:    # explore
            action = np.random.randint(0, 4)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        self.learn_step += 1
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape+1])
        r = torch.Tensor(
            sample[:, self.observer_shape+1:self.observer_shape+2])
        s_ = torch.Tensor(sample[:, self.observer_shape+2:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + 0.8 * q_next.max(1, True)[0].data

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def rl():
    q_table = build_q_table(RlEnv, RlAct)
    for episode in range(MAX_EPISODES):
        for i, row in env_data.iterrows():
            s = int(list(row)[0])
            # print(s)
            is_terminate = False
            while not is_terminate:
                a = choose_action(s, q_table)
                s_, r = get_env_feedback(s, a)
                q_predict = q_table.loc[s, a]
                if s_ == 1:
                    q_target = r + GAMMA * q_table.iloc[s, :].max()
                else:
                    q_target = r
                    is_terminate = True

                q_table.loc[s, a] += ALPHA * (q_target - q_predict)
                # print(q_table.loc[s, a])
            if s == 0:
                list_ftp_acc.append(q_table.loc[s, a])

            elif s == 1:
                list_voice_acc.append(q_table.loc[s, a])
            else:
                list_udp_acc.append(q_table.loc[s, a])
    return q_table

Init(1000, 4096)
var = Ns3AIRL(1000, RlEnv, RlAct)
res_list = ['test_packet', 'id']
args = parser.parse_args()

# 调用内存中的数据划分数据
if args.result:
    for res in res_list:
        globals()[res] = []
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

if args.use_rl:
    dqn = DQN()
exp = Experiment(1234, 4096, 'lte_wifi_test', '../../')
exp.run(show_output=1)

# 训练过程
try:
    while not var.isFinish():
        with var as data:
            if not data:
                break
    #         print(var.GetVersion())
            test_packet = data.env.test_packet
            id = data.env.id
            segmentsAcked = data.env.segmentsAcked
            packetSize = data.env.packetSize
            bytesInFlight = data.env.bytesInFlight

            if args.result:
                for res in res_list:
                    globals()[res].append(globals()[res[:-2]])
                    #print(globals()[res][-1])

            if not args.use_rl:
                new_id = 1
                new_test_packet = 1
                # IncreaseWindow
                if (test_packet < id):
                    # slow start
                    if (test_packet >= 1):
                        new_cWnd = test_packet + packetSize
                if (test_packet >= id):
                    # congestion avoidance
                    if (id > 0):
                        adder = 1.0 * (packetSize * packetSize) / test_packet
                        adder = int(max(1.0, adder))
                        new_test_packet= test_packet+ adder
                # GetSsThresh
                new_id = int(max(2 * packetSize, bytesInFlight / 2))
                data.act.new_test_packet = new_test_packet
                data.act.new_id = new_id
            else:
                s = [test_packet, id, packetSize, bytesInFlight]
                a = dqn.choose_action(s)
                if a & 1:
                    new_test_packet = id + packetSize
                else:
                    if(id > 0):
                        new_test_packet = test_packet + int(max(1, (packetSize * bytesInFlight) / test_packet))
                if a < 3:
                    new_id = 2 * packetSize
                else:
                    new_nexts = int(bytesInFlight / 2)
                data.act.test_id = new_id
                data.act.new_nexts = new_nexts

                packetSize = data.env.packetSize
                test_packet = data.env.test_packet
                bytesInFlight = data.env.bytesInFlight

                # modify the reward
                r = packetSize - bytesInFlight - test_packet
                s_ = [test_packet, id, packetSize, bytesInFlight]

                dqn.store_transition(s, a, r, s_)

                if dqn.memory_counter > dqn.memory_capacity:
                    dqn.learn()
except KeyboardInterrupt:
    exp.kill()
    del exp

if args.result:
    for res in res_list:
        y = globals()[res]
        x = range(len(y))
        plt.clf()
        plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
        plt.xlabel('Step Number')
        plt.title('Information of {}'.format(res[:-2]))
        plt.savefig('{}.png'.format(os.path.join(args.output_dir, res[:-2])))


ftp = joblib.load('./choose_model/RadomForest_ftp.pkl')
voice = joblib.load('./choose_model/RadomForest_voice.pkl')
udp = joblib.load('./choose_model/RadomForest_udp.pkl')
list_f_y = []
list_v_y = []
list_u_y = []
list_f_predict = []
list_v_predict = []
list_u_predict = []

#  将识别完业务的的数据对应模型采用信道接入
q_table.to_csv('./result/q_table.csv', encoding='utf_8_sig')

for i, row in test_data.iterrows():
    # print(row)
    s = int(list(row)[0])
    # print(s)
    a = choose_action(s, q_table)
    if a == "FTP":
        x = preprocessing.scale(row[1:7])
        x = np.array(x).reshape(1, -1)
        y = np.array(row[7:8])
        # print(x)

        list_f_predict.append(ftp.predict(x))
        list_f_y.append(y)

    elif a == "VOICE":
        x = preprocessing.scale(row[1:7])
        x = np.array(x).reshape(1, -1)
        y = np.array(row[7:8])
        # print(x)

        list_v_predict.append(voice.predict(x))
        list_v_y.append(y)

    else:
        x = row[1:7]
        x = preprocessing.scale(x)
        x = np.array(x).reshape(1, -1)
        y = np.array(row[7:8])

        list_u_predict.append(udp.predict(x))
        list_u_y.append(y)


def acc(list_y, list_predict):
    k = 0
    for i in range(len(list_y)):
        # print(list_y[i], list_predict[i])
        if list_y[i] == list_predict[i]:
            k += 1
        else:
            continue
    return k/len(list_y)