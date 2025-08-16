import torch
import numpy as np

from dataclasses import dataclass

from tianshou.policy import DQNPolicy
from tianshou.policy.modelfree.dqn import TDQNTrainingStats, DQNTrainingStats
from tianshou.data import to_torch_as, Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol

from typing import Any, Tuple
from collections import defaultdict
from utils.misc import linear_decay

@dataclass(kw_only=True)
class EWCDQNTrainingStats(DQNTrainingStats):
    loss: float
    ewc_penalty: float

## do I need to add TrainingStats?
class SoftDQNPolicy(DQNPolicy):
    """
    Subclass base Tianshou DQN with soft target network updates (polyak averaging)
    """

    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super().__init__(*args, **kwargs)
        
    
    def sync_weight(self):
        """
        Implement soft update of target network weights
        """
        old_weights = self.model_old.state_dict()
        latest_weights = self.model.state_dict()
        for key in latest_weights.keys():
            old_weights[key] = (
                self.tau * latest_weights[key]
                + (1 - self.tau) * old_weights[key]
            )
        self.model_old.load_state_dict(old_weights)
    
    def training_mode(self):
        self.is_within_training_step=True
        self.set_eps(self.eps)
    
    def testing_mode(self, eps=0.0):
        self.is_within_training_step=True
        self.set_eps(eps)

    def on_task_start(self, *args):
        self.eps = 1.0

    def on_task_step(self, task_step):
        self.calc_eps_linear_decay(task_step)

    def calc_eps_linear_decay(self, env_step):
        self.eps = linear_decay(env_step)

class EWCDQNPolicy(SoftDQNPolicy):

    def __init__(self, ewc_reg_penalty: float, *args, **kwargs):
        self.ewc_reg_penalty = ewc_reg_penalty
        self.ewc_params = dict()
        super().__init__(*args, **kwargs)
        
    def on_task_start(self, task_id, *args):
        super().on_task_start(*args)
        print(f"Learning task {task_id}")
    
    def on_task_end(self, current_task_id:int, buffer: ReplayBuffer, sample_size: int):
        print("Let's learn new regularisation weights")
        self.compute_ewc_params(
            task_id=current_task_id,
            buffer=buffer,
            sample_size=sample_size
        )

    ## TODO: should this be done in batches?
    def compute_fisher(self, buffer: ReplayBuffer, sample_size: int):
 
        ## create base fisher
        fisher = defaultdict(lambda: 0)

        ## get a sample
        batch, indices = buffer.sample(sample_size)
        batch = self.process_fn(batch, buffer, indices)

        # q_values = self(batch).logits
        # q_s_a = q_values.gather(1, torch.tensor(batch.act).unsqueeze(-1))
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q
        td_loss = (td_error.pow(2)).mean()

        ## NOTE: what is the correct loss for calculating the fisher for dqn?
        # td_loss = torch.pow(q_s_a - batch.returns, 2).mean()

        ## this is good because gradients wont go back to original network
        grads = torch.autograd.grad(td_loss, self.model.parameters(), retain_graph=False)
        for (name, _), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                fisher[name] += grad.detach()**2
        for name in fisher:
            fisher[name] /= len(batch.obs)

        return fisher

    def compute_ewc_params(self, task_id: int, buffer: ReplayBuffer, sample_size:int):
        ## do this for a task
        fisher = self.compute_fisher(buffer, sample_size)
        ## policy.model excludes target networks
        mean_params = {
            n: p.detach().clone() for n, p in self.model.named_parameters()
        }

        self.ewc_params[task_id] = {
            'mean_params': mean_params,
            'fisher': fisher
        }

    def get_ewc_reg_penalty(self, current_task_id: int) -> float:
        penalty = 0.0

        ## if ewc_params has not been initialised, return 0 penalty
        if not self.ewc_params:
            return penalty
        
        ## otherwise calc penalty
        for task_id, ewc_params in self.ewc_params.items():
            if task_id == current_task_id:
                continue
            mean_params = ewc_params['mean_params']
            fisher = ewc_params['fisher']

        for name, param in self.model.named_parameters():
            penalty += (fisher[name] * (param - mean_params[name])**2).sum()

        return self.ewc_reg_penalty * penalty

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TDQNTrainingStats:
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()

        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        ## get current task id
        current_task_id = batch['info']['task_id'].max()
        assert all(batch['info']['task_id']==current_task_id), f"You have varied task_ids in your batch: {batch['info']['task_id']}"
        ewc_reg_penalty = self.get_ewc_reg_penalty(current_task_id=current_task_id)

        if self.clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean") + ewc_reg_penalty
        else:
            loss = (td_error.pow(2) * weight).mean() + ewc_reg_penalty

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1

        return EWCDQNTrainingStats(
            loss=loss.item(), 
            ewc_penalty=ewc_reg_penalty
        )       

