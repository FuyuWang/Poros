from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.LSTMCell:
        nn.init.orthogonal_(m.weight_hh)
        nn.init.orthogonal_(m.weight_ih)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, action_space, slevel=3, h_size=128, hidden_dim=10, batch_size=10):
        super(Actor, self).__init__()

        self.slevel = slevel
        self.h_size = h_size
        self.batch_size = batch_size
        # self.prime2idx = prime2idx
        # self.idx2prime = {value: key for key, value in prime2idx.items()}
        # self.num_primes = len(self.prime2idx.keys())

        # dim_length = 9 + 4 + 7*(slevel-1) + slevel-1
        dim_length = 4 + 7*(slevel-1) + slevel-1
        self.dim_encoder = nn.Sequential(
            nn.Linear(dim_length, dim_length*hidden_dim),
            nn.ReLU(),
            nn.Linear(dim_length*hidden_dim, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            # nn.Linear(h_size, h_size),
            # nn.ReLU(),
        )

        self.x_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['pe_x'])),
        )

        self.l2_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['l2_kb'])),
        )

        self.l1_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['l1_byte'])),
        )

        self.bank_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['banks'])),
        )

        self.NTile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['N'])),
        )
        self.KTile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['K'])),
        )
        self.CTile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['C'])),
        )
        self.YTile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['Y'])),
        )
        self.XTile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['X'])),
        )
        self.RTile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['R'])),
        )
        self.STile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['split_space']['S'])),
        )

        self.parallel_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, len(action_space['unroll_space'])),
        )

        self.critic = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1),
        )

        self.lstm = torch.nn.LSTMCell(h_size, h_size)

        # self.parallel_temperature = 1.
        # self.order_temperature = 1.
        # self.tile_temperature = 1.
        self.lstm_value = None

        self.init_weight()

    def reset(self):
        self.lstm_value = self.init_hidden()

    def init_weight(self):
        self.apply(init_weights)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.batch_size, self.h_size),
                weight.new_zeros(self.batch_size, self.h_size))

    # def set_tile_temperature(self, temp):
    #     self.tile_temperature = temp
    #
    # def set_order_temperature(self, temp):
    #     self.order_temperature = temp
    #
    # def set_parallel_temperature(self, temp):
    #     self.parallel_temperature = temp

    def forward(self, state, instruction):
        '''
        :param state dim_info if instruction == 0, origin tile state if instruction in [1,6], origin tile and order state if instruction in [7, 12]
        :param instruction int parallel action, order action or tile action
        :param remaining_budgets  next level tile <= last level tile
        :return: parallel dim action
        '''
        dim_feat = self.dim_encoder(state)

        # h, x = self.lstm(dim_feat, self.lstm_value)
        # self.lstm_value = (h, x)
        h=dim_feat

        if instruction == 0:
            x_score = self.x_decoder(dim_feat)
            x_prob = F.softmax(x_score, dim=-1)
            x_density = Categorical(x_prob)
            x_action = x_density.sample()
            x_log_prob = x_density.log_prob(x_action)
            action = x_action
            log_prob = x_log_prob
            entropy = x_density.entropy()
        elif instruction == 1:
            l2_score = self.l2_decoder(h)
            l2_prob = F.softmax(l2_score, dim=-1)
            l2_density = Categorical(l2_prob)
            l2_action = l2_density.sample()
            l2_log_prob = l2_density.log_prob(l2_action)
            action = l2_action
            log_prob = l2_log_prob
            entropy = l2_density.entropy()
        elif instruction == 2:
            l1_score = self.l1_decoder(h)
            l1_prob = F.softmax(l1_score, dim=-1)
            l1_density = Categorical(l1_prob)
            l1_action = l1_density.sample()
            l1_log_prob = l1_density.log_prob(l1_action)
            action = l1_action
            log_prob = l1_log_prob
            entropy = l1_density.entropy()
        elif instruction == 3:
            bank_score = self.bank_decoder(h)
            bank_prob = F.softmax(bank_score, dim=-1)
            bank_density = Categorical(bank_prob)
            bank_action = bank_density.sample()
            bank_log_prob = bank_density.log_prob(bank_action)
            action = bank_action
            log_prob = bank_log_prob
            entropy = bank_density.entropy()
        elif instruction == 4:
            parallel_score = self.parallel_decoder(h)
            parallel_prob = F.softmax(parallel_score, dim=-1)
            parallel_density = Categorical(parallel_prob)
            parallel_action = parallel_density.sample()
            parallel_log_prob = parallel_density.log_prob(parallel_action)
            action = parallel_action
            log_prob = parallel_log_prob
            entropy = parallel_density.entropy()
        elif instruction == 5:
            NTile_score = self.NTile_decoder(h)
            NTile_prob = F.softmax(NTile_score, dim=-1)
            NTile_density = Categorical(NTile_prob)
            NTile_action = NTile_density.sample()
            NTile_log_prob = NTile_density.log_prob(NTile_action)
            action = NTile_action
            log_prob = NTile_log_prob
            entropy = NTile_density.entropy()
        elif instruction == 6:
            KTile_score = self.KTile_decoder(h)
            KTile_prob = F.softmax(KTile_score, dim=-1)
            KTile_density = Categorical(KTile_prob)
            KTile_action = KTile_density.sample()
            KTile_log_prob = KTile_density.log_prob(KTile_action)
            action = KTile_action
            log_prob = KTile_log_prob
            entropy = KTile_density.entropy()
        elif instruction == 7:
            CTile_score = self.CTile_decoder(h)
            CTile_prob = F.softmax(CTile_score, dim=-1)
            CTile_density = Categorical(CTile_prob)
            CTile_action = CTile_density.sample()
            CTile_log_prob = CTile_density.log_prob(CTile_action)
            action = CTile_action
            log_prob = CTile_log_prob
            entropy = CTile_density.entropy()
        elif instruction == 8:
            YTile_score = self.YTile_decoder(h)
            YTile_prob = F.softmax(YTile_score, dim=-1)
            YTile_density = Categorical(YTile_prob)
            YTile_action = YTile_density.sample()
            YTile_log_prob = YTile_density.log_prob(YTile_action)
            action = YTile_action
            log_prob = YTile_log_prob
            entropy = YTile_density.entropy()
        elif instruction == 9:
            XTile_score = self.XTile_decoder(h)
            XTile_prob = F.softmax(XTile_score, dim=-1)
            XTile_density = Categorical(XTile_prob)
            XTile_action = XTile_density.sample()
            XTile_log_prob = XTile_density.log_prob(XTile_action)
            action = XTile_action
            log_prob = XTile_log_prob
            entropy = XTile_density.entropy()
        elif instruction == 10:
            RTile_score = self.RTile_decoder(h)
            RTile_prob = F.softmax(RTile_score, dim=-1)
            RTile_density = Categorical(RTile_prob)
            RTile_action = RTile_density.sample()
            RTile_log_prob = RTile_density.log_prob(RTile_action)
            action = RTile_action
            log_prob = RTile_log_prob
            entropy = RTile_density.entropy()
        else:
            STile_score = self.STile_decoder(h)
            STile_prob = F.softmax(STile_score, dim=-1)
            STile_density = Categorical(STile_prob)
            STile_action = STile_density.sample()
            STile_log_prob = STile_density.log_prob(STile_action)
            action = STile_action
            log_prob = STile_log_prob
            entropy = STile_density.entropy()
        value = self.critic(h)

        return action, log_prob, entropy, value.squeeze()
