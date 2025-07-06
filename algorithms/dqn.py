from tianshou.policy import DQNPolicy
from utils.misc import linear_decay

## do I need to add TrainingStats?
class SoftDQNPolicy(DQNPolicy):
    """
    Subclass base Tianshou DQN with soft target network updates (polyak averaging)
    """

    def __init__(self, *args, tau, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
    
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

    def on_task_start(self):
        self.eps = 1.0

    def on_task_step(self, task_step):
        self.calc_eps_linear_decay(task_step)

    def calc_eps_linear_decay(self, env_step):
        self.eps = linear_decay(env_step)
