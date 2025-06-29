from tianshou.utils.logger.base import DataScope
from tianshou.utils.logger.tensorboard import TensorboardLogger

class ContinualTensorboardLogger(TensorboardLogger):
    """
    At this stage, simply overwrite the log_test_data to enable separate environment logging
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ## TODO: can probably adjust this more
    ## check what log_data is, perhaps can just write more directly what I want to log
    # (e.g. avg reward for each task)
    def log_test_data(self, log_data: dict, step: int, task_id: int):
        if step - self.last_log_test_step >= self.test_interval:
            log_data = self.prepare_dict_for_logging(log_data)
            self.write(f"{DataScope.TEST}/{task_id}_env_step", step, log_data)
            self.last_log_test_step = step