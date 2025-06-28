import gymnasium as gym
import tianshou as ts
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from datetime import datetime

if __name__=="__main__":
    ENV_NAME = "CartPole-v1"

    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(ENV_NAME) for _ in range(10)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(ENV_NAME) for _ in range(100)])

    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n


    ## wrapper to conform with tianshu
    class WrappedQNet(nn.Module):
        def __init__(self, net: nn.Module, action_dim: int):
            super().__init__()
            self.net = net
            self.q_head = nn.Linear(128, action_dim)

        def forward(self, obs, state=None, info={}):
            feat, state = self.net(obs, state=state, info=info)
            return self.q_head(feat), state

    net = ts.utils.net.common.Net(
            state_shape=state_shape, 
            action_shape=action_shape,
            hidden_sizes = [128, 128],
            device="cpu"
        )
    print(net)
    # net = WrappedQNet(net, np.prod(action_shape))
    optim = torch.optim.Adam(net.parameters(), lr = 1e-3)

        ## define the DQN policy
    policy = ts.policy.DQNPolicy(
        model = net, # our network
        optim=optim,
        action_space=env.action_space,
        discount_factor = 0.9,
        estimation_step=1,
        target_update_freq=320 ## overwrite this class to enable soft update?
    )
    t = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    writer=SummaryWriter(f'log/{ENV_NAME}/')

    train_collector = ts.data.Collector(
        policy=policy, ## the dqn
        env=train_envs,
        buffer=ts.data.VectorReplayBuffer(total_size=20000, buffer_num=10), # total_size is num obs, buffer_num is number per env
        exploration_noise=True
    )
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=25, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1), ## could adjust this to increment epsilon
        test_fn=lambda epoch, env_step: policy.set_eps(0.0),
        stop_fn=lambda mean_rewards: mean_rewards >= 500,
        logger=TensorboardLogger(writer)
    ).run()
