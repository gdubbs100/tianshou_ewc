import gymnasium as gym

from typing import Any, Dict

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer

class ContinualTrainer:

    def __init__(self, policy, writer, tasks: Dict[Any, Any], steps_per_task: int):
        self.policy = policy
        self.writer = writer ## do I need this to be a separate class?
        #TODO: how to specify this?
        # list of param dicts?
        # register tasks so env_id is included in dict
        self.tasks = tasks
        self.steps_per_task = steps_per_task
        self.make_env_fn = lambda params: gym.make(**params)
    
    def evaluate(self):
        pass

    def run(self, 
            num_train_envs: int, 
            num_eval_envs: int, 
            steps_per_rollout: int,
            update_per_step: int,
            eval_every: int,
            batch_size: int
        ):
        ## steps per collect
        ## eval frequency
        ## save frequency (networks)

        ## create evaluation envs
        eval_envs = [
            DummyVectorEnv([
                lambda: self.make_env_fn(eval_task)
                for _ in range(num_eval_envs)
            ])
            for eval_task in self.tasks.values()
        ]
        eval_collectors = [
            Collector(self.policy, eval_env, exploration_noise=False) 
            for eval_env in eval_envs
        ]

        ## start training
        global_steps=0
        for task_id, task in self.tasks.items():
            
            ## TODO: should there be an 'on task start' fn?

            ## create the training envs for current task
            train_envs = DummyVectorEnv(
                [lambda: self.make_env_fn(task) for _ in range(num_train_envs)]
            )
            train_collector = Collector(
                policy=self.policy,
                env = train_envs,
                ## TODO: how to parameterize this?
                buffer = VectorReplayBuffer(total_size = 20000, buffer_num = num_train_envs),
                exploration_noise=True
            )
            train_collector.reset_env() # get started 
            task_steps = 0
            ## NOTE: is this an off-policy loop?
            while task_steps < self.steps_per_task:
                results = train_collector.collect(
                    n_step=steps_per_rollout,
                    reset_before_collect=False
                )
                task_steps += results.n_collected_steps
                global_steps += task_steps

                if results.returns_stat is not None:
                    self.writer.add_scalar(f"train/{task_id}_mean", results.returns_stat.mean, global_steps)
                    self.writer.add_scalar(f"train/{task_id}_std", results.returns_stat.std, global_steps)
                    self.writer.add_scalar(f"train/{task_id}_steps", task_steps, global_steps)
        
                ## do update
                ## TODO: perhaps create a policy.train_mode / policy.test_mode?
                self.policy.is_within_training_step=True
                num_updates = int(update_per_step * results.n_collected_steps)
                loss=0
                for _ in range(num_updates):
                    update_results = self.policy.update(batch_size, train_collector.buffer)
                    loss += update_results.loss
                ## TODO: perhaps create a policy.train_mode / policy.test_mode?
                self.policy.is_within_training_step=False

                self.writer.add_scalar(f"train/loss", loss / num_updates, global_steps)
                ## TODO: how to log policy specific params?
                # self.writer.add_scalar(f"train/eps", eps, global_steps + task_steps)
                if global_steps % eval_every == 0:
                    ## TODO: perhaps create a policy.train_mode / policy.test_mode?
                    self.policy.set_eps(0.0)
                    for i, eval_collector in enumerate(eval_collectors):
                        eval_results = eval_collector.collect(
                            n_episode=num_eval_envs,
                            reset_before_collect=True
                        )
                        self.writer.add_scalar(f"test/{i}_mean", eval_results.returns_stat.mean, global_steps)
                        self.writer.add_scalar(f"test/{i}_std", eval_results.returns_stat.std, global_steps)
                        print(
                            f"STEP: {global_steps}: Achieved reward for task_{i}: {eval_results.returns_stat.mean} +/- {eval_results.returns_stat.std}"
                        )
                ## TODO: perhaps create a policy.train_mode / policy.test_mode?
                self.policy.set_eps(0.1)
                    

                 


            #     ## evaluate once per epoch
            #     policy.set_eps(0.0)
            #     for i, eval_collector in enumerate(eval_collectors):
            #         eval_results = eval_collector.collect(
            #             n_episode=EPISODE_PER_TEST,
            #             reset_before_collect=True
            #         )
            #         writer.add_scalar(f"test/{i}_mean", eval_results.returns_stat.mean, global_steps + epoch_steps)
            #         writer.add_scalar(f"test/{i}_std", eval_results.returns_stat.std, global_steps + epoch_steps)
            #         print(f"STEP: {global_steps + epoch_steps}: Achieved reward for task_{i}: {eval_results.returns_stat.mean} +/- {eval_results.returns_stat.std}")
                
            #     global_steps += epoch_steps
            #     # set_eps_linear_decay(epoch, global_steps)
                

            
