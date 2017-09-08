#!/usr/bin/env python
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import os, logging, gym
from two_stock_portfolio.equity_environment import EquityEnvironment
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.policies import CnnPolicy
import json
import ipdb

with open('../../../config.json') as json_data_file:
    data = json.load(json_data_file)
print(data)

assets = data["assets"]
#assets_qunantity = [1, 1]
look_back_reinforcement = data["look_back_reinforcement"]
# len(assets) #each asset have 3 options, TODO: I can use second high
# probablity neuron to find out output for other assets and to decrease
# action
# action_dim = 1
assets_qunantity = [data["quantity_buy"]] * len(assets)


SUPERVISED_THREAD = data["SUPERVISED_THREAD"]
SUPERVISED_OPTIMIZER = data["SUPERVISED_OPTIMIZER"]
SUPERVISED_RUNTIME = data["SUPERVISED_RUNTIME"]
TEST_THREAD = data["TEST_THREAD"]
TEST_RUN_TIME = data["TEST_RUN_TIME"]
THREAD_DELAY = data["THREAD_DELAY"]
network = data["network"]
n_threads = data["n_threads"]

supervised_learning = data["supervised_learning"]
test = data["test"]
noisy_net = data["noisy_net"]
reward_config= data["reward_config"]
file = data["file"]
env_config = data["env_config"]
episode_length = data["episode_length"]
test_running_time = data["test_running_time"]
train_running_time = data["train_running_time"]
input_stocks = data["input_stocks"]
env = EquityEnvironment(assets, look_back_reinforcement, input_stocks, episode_length, 0, assets_qunantity, network, supervised=False, file=file, reward_config=reward_config, config = env_config)

#ipdb.set_trace();

def train(env_id, num_frames, seed, num_cpu):
    num_timesteps = int(num_frames / 4 * 1.1) 
    def make_env(rank):
        def _thunk():
            env = EquityEnvironment(assets, look_back_reinforcement, input_stocks, episode_length, 0, assets_qunantity, network, supervised=False, file=file, reward_config=reward_config, config = env_config)
            #env.seed(seed + rank)
            if logger.get_dir():
                env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=40)
    args = parser.parse_args()    
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, num_cpu=2)


if __name__ == '__main__':
    main()
