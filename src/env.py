import numpy as np
import copy, random
import os
from subprocess import Popen, PIPE
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import pandas as pd
import math
from collections import defaultdict, OrderedDict

m_type_dicts = {1:"CONV", 2:"DSCONV", 3:"CONV", 4:"TRCONV"}


class MaestroEnvironment(object):
    def __init__(self, dimension, slevel, action_space, fitness="latency", batch_size=10):
        super(MaestroEnvironment,self).__init__()
        self.dimension = dimension
        self.norm_dimension = np.zeros((batch_size, 9), dtype=np.float)
        self.norm_dimension[:, 0:9] = 1. * np.array(dimension) / np.array([256, 2048, 2048, 224, 224, 7, 7, 2, 2])
        self.dim2id = {"N": 1, "K": 2, "C": 3, "Y": 4, "X": 5, "R": 6, "S": 7}
        self.id2dim = {1: "N", 2: "K", 3: "C", 4: "Y", 5: "X", 6: "R", 7: "S"}
        self.slevel = slevel
        self.action_space = action_space
        self.action_bound = [self.action_space['pe_x'][-1], self.action_space['l2_kb'][-1],
                             self.action_space['l1_byte'][-1], self.action_space['banks'][-1],
                             5, self.dimension[0], self.dimension[1], self.dimension[2],
                             self.dimension[3], self.dimension[4], self.dimension[5], self.dimension[6],
                             5, self.dimension[0], self.dimension[1], self.dimension[2],
                             self.dimension[3], self.dimension[4], self.dimension[5], self.dimension[6],
                             ]
        print(self.action_bound)
        self.action_bound = 1. * np.expand_dims(np.array(self.action_bound), axis=0)

        dst_path = "../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        self.out_repr = set(["N", "K", "C", "R", "S"])
        self.fitness = fitness
        self.best_reward = float('-inf')
        self.min_reward = None
        self.last_reward = 0.
        self.mode = 0
        self.total_steps = 12
        self.batch_size = batch_size
        self.mode2action = {0: 'pe_x', 1: 'l2_kb', 2: 'l1_byte', 3: 'banks', 4: 'unroll_space', 5: 'N', 6: 'K', 7: 'C',
                            8: 'Y', 9: 'X', 10: 'R', 11: 'S'}
        self.mode_sequence = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]

        pe_x = self.action_space['pe_x'][0]
        l2_kb = self.action_space['l2_kb'][0]
        l1_byte = self.action_space['l1_byte'][0]
        banks = self.action_space['banks'][0]

        # NTile = self.action_space['split_space']['N'][0]
        # KTile = self.action_space['split_space']['K'][0]
        # CTile = self.action_space['split_space']['C'][0]
        # YTile = self.action_space['split_space']['Y'][0]
        # XTile = self.action_space['split_space']['X'][0]
        # RTile = self.action_space['split_space']['R'][0]
        # STile = self.action_space['split_space']['S'][0]
        # NTile = self.action_space['split_space']['N'][0]
        # KTile = self.action_space['split_space']['K'][2]
        # CTile = self.action_space['split_space']['C'][2]
        # YTile = self.action_space['split_space']['Y'][2]
        # XTile = self.action_space['split_space']['X'][2]
        # RTile = self.action_space['split_space']['R'][2]
        # STile = self.action_space['split_space']['S'][2]
        NTile = self.action_space['split_space']['N'][-1]
        KTile = self.action_space['split_space']['K'][-1]
        CTile = self.action_space['split_space']['C'][-1]
        YTile = self.action_space['split_space']['Y'][-1]
        XTile = self.action_space['split_space']['X'][-1]
        RTile = self.action_space['split_space']['R'][-1]
        STile = self.action_space['split_space']['S'][-1]

        parallel = self.action_space['unroll_space'][0]
        # print(pe_x, l2_kb, l1_byte, banks, NTile, KTile, CTile, YTile, XTile, RTile, STile, parallel)
        self.state = np.zeros((self.batch_size, 4 + 7 * (self.slevel - 1) + (self.slevel - 1)), dtype=np.int32)
        self.state[:, 0:] = np.array([pe_x, l2_kb, l1_byte, banks,
                                      parallel[0], NTile[1] * NTile[2], KTile[1] * KTile[2], CTile[1] * CTile[2],
                                      YTile[1] * YTile[2], XTile[1] * XTile[2], RTile[1] * RTile[2], STile[1] * STile[2],
                                      parallel[1], NTile[2], KTile[2], CTile[2], YTile[2], XTile[2], RTile[2], STile[2]])

        # pool = Pool(min(self.batch_size, cpu_count()))
        # return_list = pool.map(self.get_reward, self.state)
        # self.initial_reward = np.zeros(self.batch_size)
        # for i in range(self.batch_size):
        #     self.initial_reward[i] = return_list[i][2]
        # self.min_reward = self.initial_reward.min()
        self.info = np.ones(self.batch_size)
        # print(self.min_reward)
        # self.min_reward = None

    def epoch_reset(self, dimension, fitness):

        # self.last_reward = copy.deepcopy(self.initial_reward)
        self.best_reward = float('-inf')
        self.dimension = dimension
        self.fitness = fitness
        self.mode = 0
        self.info = np.ones(self.batch_size)

        pe_x = self.action_space['pe_x'][0]
        l2_kb = self.action_space['l2_kb'][0]
        l1_byte = self.action_space['l1_byte'][0]
        banks = self.action_space['banks'][0]

        NTile = self.action_space['split_space']['N'][-1]
        KTile = self.action_space['split_space']['K'][-1]
        CTile = self.action_space['split_space']['C'][-1]
        YTile = self.action_space['split_space']['Y'][-1]
        XTile = self.action_space['split_space']['X'][-1]
        RTile = self.action_space['split_space']['R'][-1]
        STile = self.action_space['split_space']['S'][-1]

        parallel = self.action_space['unroll_space'][0]
        self.state = np.zeros((self.batch_size, 4 + 7 * (self.slevel - 1) + (self.slevel - 1)), dtype=np.int32)
        self.state[:, 0:] = np.array([pe_x, l2_kb, l1_byte, banks,
                                      parallel[0], NTile[1] * NTile[2], KTile[1] * KTile[2], CTile[1] * CTile[2],
                                      YTile[1] * YTile[2], XTile[1] * XTile[2], RTile[1] * RTile[2], STile[1] * STile[2],
                                      parallel[1], NTile[2], KTile[2], CTile[2], YTile[2], XTile[2], RTile[2], STile[2]])

        norm_state = self.state / self.action_bound

        # return np.concatenate([self.norm_dimension, norm_state], axis=1)
        return norm_state

    def get_out_repr(self, x):
        if x in self.out_repr:
            return x
        else:
            return x + "'"

    def get_reward(self, state):
        resource = state[0:4]
        sol = state[4:]
        reward, latency, power, energy, area, l1_size, l2_size = self.oberserve_maestro(resource, sol)
        constraint = (l1_size, l2_size)
        return resource, sol, reward, latency, power, energy, area, constraint

    def step(self, action):
        done = 0

        state_length = 4 + 7*(self.slevel - 1) + self.slevel - 1
        action = action.cpu().numpy()

        if 5 <= self.mode <= 11:
            action = self.action_space['split_space'][self.mode2action[self.mode]][action]
            self.state[:, self.mode] = action[:, 1] * action[:, 2]
            self.state[:, self.mode + 8] = action[:, 2]
        else:
            action = self.action_space[self.mode2action[self.mode]][action]
            if self.mode <= 3:
                self.state[:, self.mode] = action
                # self.state[:, self.mode] = self.action_space[self.mode2action[self.mode]][np.zeros(self.batch_size, dtype=np.int32)]
            else:
                self.state[:, self.mode] = action[:, 0]
                self.state[:, self.mode + 8] = action[:, 1]

        # state = np.stack([pe_x, l2_kb, l1_byte, banks,
        #                   parallel[:, 0], NTile[:, 1]*NTile[:, 2], KTile[:, 1]*KTile[:, 2], CTile[:, 1]*CTile[:, 2],
        #                   YTile[:, 1]*YTile[:, 2], XTile[:, 1]*XTile[:, 2], RTile[:, 1]*RTile[:, 2], STile[:, 1]*STile[:, 2],
        #                   parallel[:, 1], NTile[:, 2], KTile[:, 2], CTile[:, 2], YTile[:, 2], XTile[:, 2],
        #                   RTile[:, 2], STile[:, 2]], axis=1)

        norm_state = self.state / self.action_bound

        self.mode += 1

        if self.mode == self.total_steps:

            # reward_saved = copy.deepcopy(reward)
            # reward_saved[reward_saved == float('-inf')] = float('inf')
            # self.min_reward = min(self.min_reward, reward_saved.min())
            # reward_saved[reward_saved == float('inf')] = self.min_reward * 2
            # reward = reward_saved - self.last_reward
            # self.last_reward = reward_saved

            done = 1
            pool = Pool(min(self.batch_size, cpu_count()))
            return_list = pool.map(self.get_reward, self.state)
            resource = []
            sol = []
            reward = np.zeros(self.batch_size)
            latency = np.zeros(self.batch_size)
            power = np.zeros(self.batch_size)
            energy = np.zeros(self.batch_size)
            area = np.zeros(self.batch_size)
            for i in range(self.batch_size):
                resource.append(return_list[i][0])
                sol.append(return_list[i][1])
                reward[i] = return_list[i][2]
                latency[i] = return_list[i][3]
                power[i] = return_list[i][4]
                energy[i] = return_list[i][5]
                area[i] = return_list[i][6]
            self.info[reward == float('-inf')] = 0
            # print(self.info.sum())
            reward_saved = copy.deepcopy(reward)
            reward_saved[reward_saved == float('-inf')] = float('inf')
            if self.min_reward is None:
                self.min_reward = reward_saved.min()
            self.min_reward = min(self.min_reward, reward_saved.min())
            reward_saved[reward_saved == float('inf')] = self.min_reward * 2
            reward = reward_saved - self.min_reward
            print(self.min_reward, self.info)
        else:
            done = 0
            reward = np.zeros(self.batch_size)
            reward_saved = np.zeros(self.batch_size).fill(self.min_reward)
            latency = None
            power = None
            energy = None
            area = None
            resource = None
            sol = None

        # print(self.mode, self.total_steps, reward_saved, reward)
        # return np.concatenate([self.norm_dimension, norm_state], axis=1), resource, sol, reward, reward_saved, latency, power, energy, done, self.info
        return norm_state, resource, sol, reward, reward_saved, latency, power, energy, area, done, self.info

    def write_maestro(self, num_pe, sol, layer_id=0, m_file=None):
        m_type = m_type_dicts[int(self.dimension[-1])]
        stride = int(self.dimension[-2])
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            fo.write("Layer {} {{\n".format(m_type))
            fo.write("Type: {}\n".format(m_type))
            # fo.write("Stride { X: {}, Y: {} }\n".format(stride, stride))
            fo.write("Stride { X: " + str(stride) + ", Y: " + str(stride) + " }\n")
            dim = self.dimension[0:7]
            # dim[2] *= stride
            # dim[3] *= stride
            fo.write("Dimensions {{ N: {:.0f}, K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*dim))
            fo.write("Dataflow {\n")

            for i in range(1, 6):
                if sol[0] == i:
                    # if self.id2dim[i] == "Y" or self.id2dim[i] == "X":
                    #     fo.write(f"SpatialMap({sol[i]}, 1) {self.get_out_repr(self.id2dim[i])};\n")
                    # else:
                    fo.write(f"SpatialMap({sol[i]}, {sol[i]}) {self.get_out_repr(self.id2dim[i])};\n")
                else:
                    # if self.id2dim[i] == "Y" or self.id2dim[i] == "X":
                    #     fo.write(f"TemporalMap({sol[i]}, 1) {self.get_out_repr(self.id2dim[i])};\n")
                    # else:
                    fo.write(f"TemporalMap({sol[i]}, {sol[i]}) {self.get_out_repr(self.id2dim[i])};\n")

            fo.write(f"TemporalMap({sol[6]}, {sol[6]}) R;\n")
            fo.write(f"TemporalMap({sol[7]}, {sol[7]}) S;\n")

            fo.write(f"Cluster({min(num_pe, sol[sol[0]])}, P);\n")
            for i in range(1, 6):
                if sol[0+8] == i:
                    # if self.id2dim[i] == "Y" or self.id2dim[i] == "X":
                    #     fo.write(f"SpatialMap({sol[i+8]}, 1) {self.get_out_repr(self.id2dim[i])};\n")
                    # else:
                    fo.write(f"SpatialMap({sol[i+8]}, {sol[i+8]}) {self.get_out_repr(self.id2dim[i])};\n")
                else:
                    # if self.id2dim[i] == "Y" or self.id2dim[i] == "X":
                    #     fo.write(f"TemporalMap({sol[i + 8]}, 1) {self.get_out_repr(self.id2dim[i])};\n")
                    # else:
                    fo.write(f"TemporalMap({sol[i + 8]}, {sol[i + 8]}) {self.get_out_repr(self.id2dim[i])};\n")

            fo.write(f"TemporalMap({sol[6+8]}, {sol[6+8]}) R;\n")
            fo.write(f"TemporalMap({sol[7+8]}, {sol[7+8]}) S;\n")

            fo.write("}\n")
            fo.write("}\n")
            fo.write("}")

        # with open("{}.m".format(m_file), "r") as fo:
        #     lines = fo.readlines()
        #     print(lines, '\n')

    def oberserve_maestro(self, resource, sol):

        num_pe = resource[0]*resource[0]
        l2_size = resource[1] * 1024
        l1_size = resource[2]
        # l1_size = resource[2] * 1024
        bandwidth = 2 * resource[3] * 8  # dual ports

        # print(num_pe, l1_size, l2_size, bandwidth, sol)

        m_file = "{}".format(random.randint(0, 2 ** 32))
        self.write_maestro(num_pe, sol, m_file=m_file)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None

        # command = [self._executable,
        #            "--Mapping_file={}.m".format(m_file),
        #            "--full_buffer=false", "--noc_bw={}".format(bandwidth),
        #            "--noc_hops=1", "--noc_hop_latency=1",
        #            "--noc_mc_support=true", "--num_pes={}".format(num_pe),
        #            "--num_simd_lanes=1", "--l1_size={}".format(l1_size),
        #            "--l2_size={}".format(l2_size), "--print_res=false", "--print_res_csv_file=true",
        #            "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw={}".format(bandwidth),
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(num_pe),
                   "--num_simd_lanes=1", "--l1_size={}".format(81920000),
                   "--l2_size={}".format(81920000), "--print_res=false", "--print_res_csv_file=true",
                   "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        os.remove("./{}.m".format(m_file)) if os.path.exists("./{}.m".format(m_file)) else None
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            used_l1_size = np.array(df[" L1 SRAM Size (Bytes)"]).reshape(-1, 1)
            used_l2_size = np.array(df["  L2 SRAM Size (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            observation = [np.mean(x) for x in [runtime, throughput, energy, area, used_l1_size, used_l2_size, mac, power]]
            # print(sol, used_l2_size, used_l1_size)
            def catch_exception():
                if used_l1_size > l1_size or used_l2_size > l2_size or runtime < 1 or used_l1_size < 0 or used_l2_size < 0 or power > 1E+4:
                    # print(used_l2_size, l2_size, used_l1_size, l1_size, power)
                    return True
                else:
                    return False

            if len(str(stdout)) > 3 or catch_exception():
                # print(stdout)
                return float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), np.sum(l1_size), np.sum(l2_size)
            return self.judge(observation)
        except Exception as e:
            # print(e, sol)
            return float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf'), -1, -1

    def judge(self, observation):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = observation
        # print(observation)
        # values = []
        # for term in [self.fitness]:
        #     if term == "energy":
        #         reward = -energy
        #     elif term == "thrpt_ave":
        #         reward = throughput
        #     elif term == "EDP":
        #         reward = -energy * runtime * 1E-6
        #     elif term == "LAP":
        #         reward = -area * runtime
        #     elif term == "EAP":
        #         reward = -area * energy
        #     elif term == "thrpt" or term == "thrpt_naive":
        #         reward = throughput
        #     elif term == "thrpt_btnk":
        #         reward = throughput
        #     elif term == "latency":
        #         reward = -runtime
        #     elif term == "area":
        #         reward = -area
        #     elif term == "l1_size":
        #         reward = - l1_size
        #     elif term == "l2_size":
        #         reward = -l2_size
        #     elif term == "power":
        #         reward = -power
        #     else:
        #         raise NameError('Undefined fitness type')
        #     values.append(reward)
        # values.append(l1_size)
        # values.append(l2_size)
        if self.fitness == 'lpp':
            reward = -1 * runtime * power * 1e-6
        elif self.fitness == 'edp':
            reward = -1 * runtime * energy * 1e-6
        elif self.fitness == 'lpa':
            reward = -1 * runtime * power * area
        return reward, -runtime, -power, -energy, -area, l1_size, l2_size
