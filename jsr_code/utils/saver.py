import os
import shutil
import torch
import numpy as np
from datetime import datetime

class Saver(object):
    def __init__(self, cfg):
        self.experiment_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME)
        self.experiment_checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.experiment_code_dir = os.path.join(self.experiment_dir, "code")
        os.makedirs(self.experiment_checkpoints_dir, exist_ok=True)
        os.makedirs(self.experiment_code_dir, exist_ok=True)
        os.system("rsync -avm --include='*/' --include='*.py' --exclude='*' ./ " + self.experiment_code_dir)

    def save_checkpoint(self, state, is_best, filename="checkpoint-latest.pth"):
        if is_best:
            filename = "checkpoint-best.pth"
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a') as f:
                f.write(str(state["epoch"]) + ", " + str(best_pred) + "\n")
        torch.save(state, os.path.join(self.experiment_checkpoints_dir, filename))

    def save_experiment_config(self, cfg):

        with open(os.path.join(self.experiment_dir, 'parameters.yaml'), 'w') as f:
            f.write(cfg.dump())


class ResultSaver(object):
    def __init__(self, cfg, dataset):
        print(cfg.EXPERIMENT.OUT_DIR, str(cfg.EXPERIMENT.NAME)) 
        self.experiment_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME)
        self.time_stamp = datetime.now().strftime(r'%Y%m%d_%H%M%S.%f').replace('.','_')
        self.experiment_tests = os.path.join(self.experiment_dir, "test_results", dataset, self.time_stamp)
        os.makedirs(self.experiment_tests, exist_ok=True)
        print ("Saving results to: {}".format(self.experiment_tests))
    
    def save_batch(self, sample, output): 
        for b in range(0, output["anomaly_score"].size()[0]):
            pred = output["anomaly_score"][b, ...].cpu().numpy().squeeze()
            np.save(os.path.join(self.experiment_tests, sample["image_name"][b] + ".npy"), pred.astype(np.float16))

    def save_stats(self, stats):
        with open(os.path.join(self.experiment_dir, 'tests.txt'), 'a') as f:
            s = self.time_stamp + ", " + "{:0.5f}, "*len(stats)
            f.write(s[:-2].format(*stats) + "\n")

