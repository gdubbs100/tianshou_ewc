from tianshou.policy import DQNPolicy

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