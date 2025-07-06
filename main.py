import gymnasium as gym
import tianshou as ts
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from algorithms.dqn import SoftDQNPolicy
from environments.custom_cartpole import ModifiableCartPole
from utils.continual_trainer import ContinualTrainer
from utils.logger import Logger

from datetime import datetime

if __name__=="__main__":
    ENV_NAME = "CartPole-v1"

    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    hidden_sizes = [64, 64]
    net = ts.utils.net.common.Net(
            state_shape=state_shape, 
            action_shape=action_shape,
            hidden_sizes = hidden_sizes,
            dueling_param = (
                {'hidden_sizes': hidden_sizes},
                {'hidden_sizes': hidden_sizes},
            ),
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
    logger = Logger(
        filepath=f'log/custom_continual/{ENV_NAME}/{t}_dqn/',
        csv_headers = [
            'training_task',
            'eval_task',
            'global_step',
            'mean_rew',
            'std_rew',
            'min_rew',
            'max_rew'
        ]
    )

    ## Making continual learner
    # TASKS = [0, 1, 2]
    NUM_ENVS = 10
    TASKS = {
        0: {'id':'ModifiableCartPole-v0'},
        1: {'id':'ModifiableCartPole-v0', "masscart": 0.1,"masspole": 0.1, "force_mag":25.0},
        2: {'id':'ModifiableCartPole-v0', "masspole": 2.0, "force_mag": 1.0}
    }
    MAX_EPOCH = 5
    STEPS_PER_COLLECT = 10
    STEPS_PER_EPOCH = 10000
    UPDATE_PER_STEP = 0.1
    EPISODE_PER_TEST=10
    BATCH_SIZE=64
    trainer = ContinualTrainer(
        policy=policy, 
        writer=logger, 
        tasks = TASKS, 
        steps_per_task=100_000,
        replay_buffer_size = 20_000)

    trainer.run(
        num_train_envs=10,
        num_eval_envs=10,
        steps_per_rollout=STEPS_PER_COLLECT,
        update_per_step=UPDATE_PER_STEP,
        eval_every=5000,
        save_model_every=5000,
        batch_size=BATCH_SIZE

    )





    # eps=1.0
    # global_steps = 0
    # for task in TASKS:
    #     print(f"Training task: {task}...")
    #     # train_envs = ts.env.DummyVectorEnv([lambda: ModifiableCartPole(**PARAMS[task]) for _ in range(NUM_ENVS)])
    #     train_envs = ts.env.DummyVectorEnv([lambda: gym.make(id='ModifiableCartPole-v0', **PARAMS[task]) for _ in range(NUM_ENVS)])
    #     train_collector = ts.data.Collector(
    #         policy=policy, ## the dqn
    #         env=train_envs,
    #         ## TODO: think about this - do we want to share replay buffers across tasks?
    #         ## will this setup allow that?
    #         buffer=ts.data.VectorReplayBuffer(total_size=20000, buffer_num=10), # total_size is num obs, buffer_num is number per env
    #         exploration_noise=True
    #     )
    #     train_collector.reset_env()
    #     ## create envs for each environment to be learnt
    #     eval_envs = [
    #         # ts.env.DummyVectorEnv([lambda: ModifiableCartPole(**PARAMS[eval_task]) for _ in range(10)])
    #         ts.env.DummyVectorEnv([lambda: gym.make(id='ModifiableCartPole-v0',**PARAMS[eval_task]) for _ in range(10)]) 
    #         for eval_task in TASKS
    #     ]
    #     eval_collectors = [
    #         ts.data.Collector(policy, eval_env, exploration_noise=False) 
    #         for eval_env in eval_envs
    #     ]
    #     # breakpoint()
    #     for epoch in range(MAX_EPOCH):
    #         epoch_steps=0
    #         while epoch_steps < STEPS_PER_EPOCH:
                
    #             eps = linear_decay(epoch_steps + global_steps)
    #             policy.set_eps(eps)
    #             results = train_collector.collect(
    #                 n_step=STEPS_PER_COLLECT,
    #                 reset_before_collect=False
    #             )
    #             epoch_steps += results.n_collected_steps

    #             if results.returns_stat is not None:
    #                 writer.add_scalar(f"train/{task}_mean", results.returns_stat.mean, global_steps+epoch_steps)
    #                 writer.add_scalar(f"train/{task}_std", results.returns_stat.std, global_steps+epoch_steps)
        
    #             ## do update
    #             policy.is_within_training_step=True
    #             num_updates = int(UPDATE_PER_STEP * results.n_collected_steps)
    #             loss=0
    #             for _ in range(num_updates):
    #                 update_results = policy.update(BATCH_SIZE, train_collector.buffer)
    #                 loss += update_results.loss
    #             policy.is_within_training_step=False

    #             writer.add_scalar(f"train/loss", loss / num_updates, global_steps + epoch_steps)
    #             writer.add_scalar(f"train/eps", eps, global_steps + epoch_steps)
            
    #         ## evaluate once per epoch
    #         policy.set_eps(0.0)
    #         for i, eval_collector in enumerate(eval_collectors):
    #             eval_results = eval_collector.collect(
    #                 n_episode=EPISODE_PER_TEST,
    #                 reset_before_collect=True
    #             )
    #             writer.add_scalar(f"test/{i}_mean", eval_results.returns_stat.mean, global_steps + epoch_steps)
    #             writer.add_scalar(f"test/{i}_std", eval_results.returns_stat.std, global_steps + epoch_steps)
    #             print(f"STEP: {global_steps + epoch_steps}: Achieved reward for task_{i}: {eval_results.returns_stat.mean} +/- {eval_results.returns_stat.std}")
            
    #         global_steps += epoch_steps
    #         # set_eps_linear_decay(epoch, global_steps)
            

            



        # result = ts.trainer.OffpolicyTrainer(
        # trainer = ContinualOffpolicyTrainer(
        #     policy=policy,
        #     train_collector=train_collector,
        #     test_collector=eval_collectors,
        #     max_epoch=5, step_per_epoch=10000, step_per_collect=10,
        #     update_per_step=0.1, episode_per_test=100, batch_size=64,
        #     train_fn=lambda epoch, env_step: set_eps_linear_decay(epoch, env_step), ## could adjust this to increment epsilon
        #     test_fn=lambda epoch, env_step: policy.set_eps(0.0),
        #     stop_fn=lambda mean_rewards: mean_rewards >= 500,
        #     logger=ContinualTensorboardLogger(writer)
        # )
        # breakpoint()
        
        # results = trainer.run()