def linear_decay(step, eps_start=1.0, eps_end=0.05, decay_steps=10_000):
    if step >= decay_steps:
        return eps_end
    else:
        return eps_start - (eps_start - eps_end) * (step / decay_steps)
    
class TaskIdHook:

    def __init__(self, task_id: int):
        self.task_id = task_id
    
    def __call__(self, action_batch, rollout_batch):
        ## save task_id in info
        rollout_batch['info']['task_id'] = [self.task_id]