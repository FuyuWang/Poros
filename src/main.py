import copy
import argparse
import random
import pandas as pd
import glob
import os, sys
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from space import get_hw_space, get_sw_space
from env import MaestroEnvironment
from a2c import Actor
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 1e-3  # learning rate of the actor
GAMMA = 0.95  # discount factor
CLIPPING_LSTM = 10
CLIPPING_MODEL = 5
EPISIOLON = 2**(-12)


def compute_policy_loss(rewards, log_probs, entropies, values, info, filter=False):
    dis_rewards = []
    batch_size = log_probs.size(1)

    batch_masks = torch.from_numpy(info).to(log_probs.device)
    fail_idx = []
    for i in range(batch_size):
        if info[i] < 0:
            fail_idx.append(i)
    if len(fail_idx) > 4:
        fail_idx = random.sample(fail_idx, 4)
    batch_masks[fail_idx] = 1.
    # if filter:
    #     batch_masks = torch.from_numpy(info).to(log_probs.device)
    # else:
    #     success_idx = []
    #     fail_idx = []
    #     for i in range(batch_size):
    #         if rewards[-1, i] > 0:
    #             success_idx.append(i)
    #         else:
    #             fail_idx.append(i)
    #     # if len(fail_idx) > 2 * len(success_idx):
    #     #     fail_idx = random.sample(fail_idx, 2 * len(success_idx))
    #     if len(fail_idx) > 4:
    #         fail_idx = random.sample(fail_idx, 4)
    #     # print(len(success_idx), len(fail_idx), rewards[-1, :])
    #     batch_masks = log_probs.new_zeros(batch_size)
    #     batch_masks[success_idx] = 1.
    #     batch_masks[fail_idx] = 1.
    #
    # else:
    #     batch_masks = log_probs.new_ones(batch_size)

    # rewards = rewards[7:]
    # log_probs = log_probs[:-7]
    # log_prob_masks = log_prob_masks[:-7]

    R = np.zeros(batch_size)
    for r in rewards[::-1]:
        R = r + GAMMA * R
        dis_rewards.insert(0, R)
    dis_rewards = torch.from_numpy(np.array(dis_rewards)).to(log_probs.device)
    # print(dis_rewards.size(), log_prob_masks[:,7,:], log_prob_masks[:,6,:])

    advantage = dis_rewards - values
    policy_loss = (-log_probs * advantage.detach()).mean(dim=0)
    value_loss = advantage.pow(2).mean(dim=0)

    value_coeff = 0.5
    entropy_coeff = 0.02
    # loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropies.sum(dim=0)
    loss = policy_loss + value_coeff * value_loss
    loss = (loss * batch_masks).sum() / batch_masks.sum()
    # policy_loss = dis_rewards * (-1 * log_probs)
    # policy_loss = policy_loss.sum(dim=0) * batch_masks
    # policy_loss = policy_loss.sum() / batch_masks.sum()
    # return policy_loss
    return loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def train(dimension, actor_state_dict=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LR_ACTOR = 5e-4  # learning rate of the actor
    CLIPPING_MODEL = 100

    agent_chkpt = {}
    agent_chkpt['dimension'] = dimension
    agent_chkpt['best_reward_record'] = []
    agent_chkpt['best_latency_record'] = []
    agent_chkpt['best_power_record'] = []
    agent_chkpt['best_energy_record'] = []
    agent_chkpt['best_area_record'] = []
    agent_chkpt['best_sols'] = []
    agent_chkpt['best_reward'] = float("-Inf")
    agent_chkpt['best_latency'] = float("-Inf")
    agent_chkpt['best_power'] = float("-Inf")
    agent_chkpt['best_energy'] = float("-Inf")
    agent_chkpt['best_area'] = float("-Inf")
    agent_chkpt['best_resource'] = float("-Inf")
    agent_chkpt['best_sol'] = float("-Inf")
    # agent_chkpt['best_state'] = None

    action_space = get_hw_space()
    # if dimension[1] == 1:
    #     action_space.update(get_sw_space(dimension, opt.slevel, par_k=False))
    # else:
    action_space.update(get_sw_space(dimension, opt.slevel, par_k=True))
    num_episodes = 32
    env = MaestroEnvironment(dimension, opt.slevel, action_space, opt.fitness, num_episodes)
    actor = Actor(action_space, opt.slevel, batch_size=num_episodes).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR, betas=(0.9, 0.999))
    best_reward = float('-inf')
    thres_epoch = 10
    if actor_state_dict is not None:
        actor.load_state_dict(state_dict=actor_state_dict, strict=False)
    for ep in range(opt.epochs):
        # env.epoch_reset(dimension, opt.fitness)
        #
        # if ep > 0 and ep % thres_epoch == 0:
        #     for param_group in actor_optimizer.param_groups:
        #         for param in param_group['params']:
        #             if param.requires_grad:
        #                 param_group['lr'] = param_group['lr'] * 0.8
        #                 break
        #         print(param_group['lr'])
        # print(datetime.now().time())
        rewards = []
        log_probs = []
        log_prob_masks = []
        values = []
        entropies = []
        norm_state = env.epoch_reset(dimension, opt.fitness)
        actor.reset()
        for t in range(env.total_steps):
            norm_state = torch.from_numpy(norm_state).type(torch.FloatTensor).to(device)
            action, log_prob, entropy, value = actor(norm_state, t)
            norm_state, resource, sol, reward, reward_saved, latency, power, energy, area, done, info = env.step(action)
            # if t > 3:
            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            # print(reward)
        # print(reward, (reward > 0).sum(), (reward < 0).sum())
        if info.sum() == 0:
            continue
        rewards = np.stack(rewards, axis=0)
        log_probs = torch.stack(log_probs, dim=0)
        entropies = torch.stack(entropies, dim=0)
        values = torch.stack(values, dim=0)
        policy_loss = compute_policy_loss(rewards, log_probs, entropies, values, info, ep >= 2)
        best_idx = np.argmax(reward_saved)
        if reward_saved[best_idx] > best_reward:
            best_reward = reward_saved[best_idx]
            agent_chkpt['best_actor'] = actor.state_dict()
            agent_chkpt['best_reward'] = best_reward
            agent_chkpt['best_latency'] = latency[best_idx]
            agent_chkpt['best_power'] = power[best_idx]
            agent_chkpt['best_energy'] = energy[best_idx]
            agent_chkpt['best_area'] = area[best_idx]
            agent_chkpt['best_resource'] = resource[best_idx]
            agent_chkpt['best_sol'] = sol[best_idx]
            # agent_chkpt['best_state'] = state[best_idx]
            # print("Epoch {}, Best Reward: {}, Best Sol: {}".format(ep, best_reward, latency[best_idx], power[best_idx], sol[best_idx]))
            print(f"Epoch {ep}, Best Reward: { best_reward, latency[best_idx], power[best_idx], area[best_idx]}, "
                  f"Best Sol: {resource[best_idx], sol[best_idx]}")

        agent_chkpt['best_reward_record'].append(agent_chkpt['best_reward'])
        agent_chkpt['best_latency_record'].append(agent_chkpt['best_latency'])
        agent_chkpt['best_power_record'].append(agent_chkpt['best_power'])
        agent_chkpt['best_energy_record'].append(agent_chkpt['best_energy'])
        agent_chkpt['best_area_record'].append(agent_chkpt['best_area'])
        agent_chkpt['best_sols'].append(agent_chkpt['best_sol'])
        log_str = f"Epoch {ep}, Best Reward: {best_reward, agent_chkpt['best_latency'], agent_chkpt['best_power'], agent_chkpt['best_area']}, " \
                  f"Best Sol: {agent_chkpt['best_resource'], agent_chkpt['best_sol']}\n"
        print(log_str)
        epf.write(log_str)
        epf.flush()
        # policy_loss /= num_episodes
        actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), CLIPPING_MODEL)
        actor_optimizer.step()

    new_state_dict = {}
    for key, value in actor.state_dict().items():
        if 'Tile' not in key:
            new_state_dict[key] = value
    return new_state_dict, agent_chkpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness', type=str, default="lpp", help='objective fitness')
    parser.add_argument('--learning', default=False, action='store_true', help='whether to execute learning procedure')
    parser.add_argument('--input_size', type=int, default=1, help='number of inputs')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    # parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--slevel', type=int, default=3, help='parallelization level min')
    parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    parser.add_argument('--seed', type=int, default=42)

    opt = parser.parse_args()
    m_file_path = "../model/"
    new_state_dict = None
    # for input_size in [1, 4, 16, 64, 128, 256, 512]:
    for input_size in [1, 16]:
        # for model in ['ResNet50', 'UNet', 'MobileNetV2']:
        for model in ['PaLM']:
        # for model in ['ResNet50', 'MobileNetV2', 'UNet', 'Transformer']:
            m_file = os.path.join(m_file_path, model + ".csv")
            df = pd.read_csv(m_file)
            model_defs = df.to_numpy()

            num_layers, dim_size = model_defs.shape
            dim2chkpt = {}
            for i in range(0, num_layers):
            # for i in [3, 12]:
                dimension = np.zeros(len(model_defs[i]) + 1)
                dimension[0] = input_size
                dimension[1:] = model_defs[i]
                dimension = dimension.astype(np.int32).tolist()
                outdir = opt.outdir
                outdir = os.path.join("../", outdir)

                exp_name = "A2C_{}_{}_inputs-{}_EPOCH-{}/layer-{}".format(model, opt.fitness, input_size, opt.epochs, i)

                outdir_exp = os.path.join(outdir, exp_name)
                os.makedirs(outdir, exist_ok=True)
                os.makedirs(outdir_exp, exist_ok=True)

                dimension_to_key = ','.join(str(j) for j in dimension)
                if dimension_to_key in dim2chkpt:
                    agent_chkpt = dim2chkpt[dimension_to_key]
                    pickle.dump(agent_chkpt, open(os.path.join(outdir_exp, 'agent_chkpt.plt'), 'wb'))
                    print("repeated")
                else:
                    chkpt_file_t = "{}".format("result")
                    log_file = os.path.join(outdir_exp, chkpt_file_t + ".log")
                    epf = open(log_file, 'a')
                    # dimension[2] /= dimension[6]
                    # dimension[3] /= dimension[6]
                    print(dimension, dimension_to_key)
                    try:
                        set_seed(opt.seed)
                        if opt.learning:
                            print(opt.learning)
                            new_state_dict, agent_chkpt = train(dimension, new_state_dict)
                        else:
                            new_state_dict, agent_chkpt = train(dimension)
                        dim2chkpt[dimension_to_key] = agent_chkpt
                        torch.save(new_state_dict, os.path.join(outdir_exp, 'state_dict.plt'))
                        pickle.dump(agent_chkpt, open(os.path.join(outdir_exp, 'agent_chkpt.plt'), 'wb'))
                    finally:
                        for f in glob.glob("*.m"):
                            os.remove(f)
                        for f in glob.glob("*.csv"):
                            os.remove(f)

