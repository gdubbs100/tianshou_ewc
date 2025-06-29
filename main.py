import gymnasium as gym
import tianshou as ts
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.highlevel.trainer import EpochTrainCallbackDQNEpsLinearDecay

from algorithms.dqn import SoftDQNPolicy
from environments.custom_cartpole import ModifiableCartPole
from utils.continual_logger import ContinualTensorboardLogger
from utils.continual_trainer import ContinualOffpolicyTrainer

from datetime import datetime

if __name__=="__main__":
    ENV_NAME = "CartPole-v1"

    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = ts.utils.net.common.Net(
            state_shape=state_shape, 
            action_shape=action_shape,
            hidden_sizes = [128, 128],
            device="cpu"
        )
    optim = torch.optim.Adam(net.parameters(), lr = 1e-3)

    policy = SoftDQNPolicy(
        model = net,
        optim = optim,
        action_space = env.action_space,
        discount_factor = 0.99,
        estimation_step = 1,
        target_update_freq=1, # each update for soft updates
        tau = 0.05 # soft update parameter for polyak averaging
    )
    t = datetime.now().strftime("%d%m%Y%H%M%S")
    writer=SummaryWriter(f'log/continual/{ENV_NAME}/{t}_dqn/')

    def linear_decay(step, eps_start=1.0, eps_end=0.05, decay_steps=10_000):
        if step >= decay_steps:
            return eps_end
        else:
            return eps_start - (eps_start - eps_end) * (step / decay_steps)
        
    def set_eps_linear_decay(epoch, env_step):
        eps = linear_decay(env_step)
        policy.set_eps(eps)

    ## Making continual learner
    TASKS = [0, 1, 2]
    NUM_ENVS = 10
    PARAMS = {
        0: {},
        1: {"masspole": 0.1, "force_mag":15.0},
        2: {"masspole": 1.0, "force_mag": 2.0}
    }
    for task in TASKS:
        print(f"Training task: {task}...")
        train_envs = ts.env.DummyVectorEnv([lambda: ModifiableCartPole(**PARAMS[task]) for _ in range(NUM_ENVS)])
        train_collector = ts.data.Collector(
            policy=policy, ## the dqn
            env=train_envs,
            ## TODO: think about this - do we want to share replay buffers across tasks?
            ## will this setup allow that?
            buffer=ts.data.VectorReplayBuffer(total_size=20000, buffer_num=10), # total_size is num obs, buffer_num is number per env
            exploration_noise=True
        )
        ## create envs for each environment to be learnt
        eval_envs = [
            ts.env.DummyVectorEnv([lambda: ModifiableCartPole(**PARAMS[eval_task]) for _ in range(10)]) 
            for eval_task in TASKS
        ]
        eval_collectors = [
            ts.data.Collector(policy, eval_env, exploration_noise=False) 
            for eval_env in eval_envs
        ]

        # result = ts.trainer.OffpolicyTrainer(
        trainer = ContinualOffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=eval_collectors,
            max_epoch=5, step_per_epoch=10000, step_per_collect=10,
            update_per_step=0.1, episode_per_test=100, batch_size=64,
            train_fn=lambda epoch, env_step: set_eps_linear_decay(epoch, env_step), ## could adjust this to increment epsilon
            test_fn=lambda epoch, env_step: policy.set_eps(0.0),
            stop_fn=lambda mean_rewards: mean_rewards >= 500,
            logger=ContinualTensorboardLogger(writer)
        )
        breakpoint()
        
        results = trainer.run()