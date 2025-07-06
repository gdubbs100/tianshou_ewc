import gymnasium as gym

from typing import Any, Dict, List

from tianshou.policy import BasePolicy
from tianshou.env import DummyVectorEnv
from tianshou.data import (
    Collector, 
    VectorReplayBuffer,
    PrioritizedVectorReplayBuffer
)

from .logger import Logger

class ContinualTrainer:

    def __init__(
            self, 
            policy: BasePolicy, 
            writer: Logger, 
            tasks: Dict[Any, Any], 
            steps_per_task: int, 
            replay_buffer_size: int
        ):
        self.policy = policy
        self.writer = writer ## do I need this to be a separate class?
        self.tasks = tasks
        self.steps_per_task = steps_per_task
        self.make_env_fn = lambda params: gym.make(**params)
        ## TODO: should I pass a replay buffer to the trainer?
        self.replay_buffer_size = replay_buffer_size
    
    def initialise_train_envs(self, current_task, num_train_envs: int) -> Collector:
        train_envs = DummyVectorEnv(
            [lambda: self.make_env_fn(current_task) for _ in range(num_train_envs)]
        )
        train_collector = Collector(
            policy=self.policy,
            env = train_envs,
            buffer = PrioritizedVectorReplayBuffer(
                total_size = self.replay_buffer_size, 
                buffer_num = num_train_envs,
                ## TODO: parameterize
                alpha=0.6,
                beta=0.4
            ),
            exploration_noise=True
        )
        return train_collector

    def initialise_eval_envs(self, num_eval_envs: int) -> List[Collector]:
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
        return eval_collectors

    def run(self, 
            num_train_envs: int, 
            num_eval_envs: int, 
            steps_per_rollout: int,
            update_per_step: int,
            eval_every: int,
            save_model_every: int,
            batch_size: int
        ):

        ## create evaluation envs
        eval_collectors = self.initialise_eval_envs(num_eval_envs)

        ## start training
        global_steps=0
        for task_id, task in self.tasks.items():
            ## policy func for each task start
            ## TODO: pass args to this?
            self.policy.on_task_start()

            ## create the training envs for current task
            train_collector = self.initialise_train_envs(task, num_train_envs)
            train_collector.reset_env() # get started 
            task_steps = 0

            ## NOTE: is this an off-policy loop?
            while task_steps < self.steps_per_task:
                ## policy func for each step
                self.policy.on_task_step(task_steps)

                ## set training mode
                self.policy.training_mode()
                

                results = train_collector.collect(
                    n_step=steps_per_rollout,
                    reset_before_collect=False
                )
                task_steps += results.n_collected_steps
                global_steps += results.n_collected_steps

                if results.returns_stat is not None:
                    self.writer.add_scalar(f"train/{task_id}_mean", results.returns_stat.mean, global_steps)
                    self.writer.add_scalar(f"train/{task_id}_std", results.returns_stat.std, global_steps)
                    self.writer.add_scalar(f"train/{task_id}_steps", task_steps, global_steps)
        
                ## do update
                num_updates = int(update_per_step * results.n_collected_steps)
                loss=0
                for _ in range(num_updates):
                    update_results = self.policy.update(batch_size, train_collector.buffer)
                    loss += update_results.loss

                
                self.writer.add_scalar(f"train/loss", loss / num_updates, global_steps)
                ## TODO: find a way to do this that is generic across policies
                self.writer.add_scalar(f"train/eps", self.policy.eps, global_steps + task_steps)

                ## policy specific testing mode
                self.policy.testing_mode()
                if global_steps % eval_every == 0:
                    self.evaluate(eval_collectors, global_steps, task_id, num_eval_envs)
                
                if global_steps % save_model_every == 0:
                    self.writer.save_network(self.policy)

        ## TODO:
        ## log final models/print final results? Run one last eval?
        
        ## close the logger
        self.writer.close()

    def evaluate(
            self, 
            eval_collectors: List[Collector], 
            global_steps: int, 
            current_task_id: int, 
            num_eval_envs: int
        ):
        for i, eval_collector in enumerate(eval_collectors):
            eval_results = eval_collector.collect(
                n_episode=num_eval_envs,
                reset_before_collect=True
            )

            self.writer.add_scalar(f"test/{i}_mean", eval_results.returns_stat.mean, global_steps)
            self.writer.add_scalar(f"test/{i}_std", eval_results.returns_stat.std, global_steps)
            self.writer.write_csv_row(
                [
                    current_task_id, 
                    i, 
                    global_steps,
                    eval_results.returns_stat.mean, 
                    eval_results.returns_stat.std, 
                    eval_results.returns_stat.min, 
                    eval_results.returns_stat.max
                ]
            )
            print(
                f"STEP: {global_steps}: Achieved reward for task_{i}: {eval_results.returns_stat.mean} +/- {eval_results.returns_stat.std}"
            )
