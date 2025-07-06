import os
import csv
import torch
from typing import List, Any
from tianshou.policy import BasePolicy
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, filepath: str, csv_headers: List[str]):
        ## create the folder
        self.filepath = filepath
        self.tf_writer = SummaryWriter(filepath)

        ## initially we want training_task, eval_task, global_step, mean_rew, std_rew (min_rew, max_rew)
        self.csv_headers = csv_headers
        self.initialise_csv()

    def initialise_csv(self):
        csv_location=os.path.join(self.filepath, 'results.csv')
        self.csv_file = open(csv_location,mode="w",newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # write headers
        self.csv_writer.writerow(self.csv_headers)

    ## TODO: what does this include?
    def write_csv_row(self, data: List[Any]):
        assert len(data) == len(self.csv_headers), "Data does not match csv format!"
        self.csv_writer.writerow(data)
        self.csv_file.flush()

    def add_scalar(self, name: str, data: float, step: int):
        self.tf_writer.add_scalar(name, data, step)

    def save_network(self, policy: BasePolicy):
        torch.save(
            policy.state_dict(), 
            os.path.join(self.filepath, "policy.pth")
        )

    def close(self):
        self.csv_file.close()
        self.tf_writer.close()
