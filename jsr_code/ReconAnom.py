import os
import sys

import matplotlib.pyplot as plt
import torch
import importlib
sys.path.append('jsr_code/')
from config import get_cfg_defaults


class MethodEvaluator():
    def __init__(self, **kwargs):
        """ Model initialization. """
        raise NotImplementedError

    def evaluate(self, image):
        """ Implement forward pass for a particular method. Return anomaly score per pixel. """
        raise NotImplementedError


class ReconAnom(MethodEvaluator):
    def __init__(self, **kwargs) -> None:
        self.exp_dir = kwargs["exp_dir"]
        self.code_dir = os.path.join(self.exp_dir, "code")

        cfg_local = get_cfg_defaults()

        if os.path.isfile(os.path.join(self.exp_dir, "parameters.yaml")):
            with open(os.path.join(self.exp_dir, "parameters.yaml"), 'r') as f:
                cc = cfg_local._load_cfg_from_yaml_str(f)
            cfg_local.merge_from_file(os.path.join(self.exp_dir, "parameters.yaml"))
            cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME
        else:
            assert False, "Experiment directory does not contain parameters.yaml: {}".format(self.exp_dir)
        if (os.path.isfile(os.path.join(self.exp_dir, "checkpoints", "checkpoint-best.pth"))
                and cfg_local.EXPERIMENT.RESUME_CHECKPOINT is None):
            cfg_local.EXPERIMENT.RESUME_CHECKPOINT = os.path.join(self.exp_dir, "checkpoints", "checkpoint-best.pth")
        elif (cfg_local.EXPERIMENT.RESUME_CHECKPOINT is None
              or not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT)):
            assert False, "Experiment dir does not contain best checkpoint, or no checkpoint specified or specified checkpoint does not exist: {}".format(
                cfg_local.EXPERIMENT.RESUME_CHECKPOINT)

        if not torch.cuda.is_available():
            print("GPU is disabled")
            cfg_local.SYSTEM.USE_GPU = False

        if cfg_local.MODEL.SYNC_BN is None:
            if cfg_local.SYSTEM.USE_GPU and len(cfg_local.SYSTEM.GPU_IDS) > 1:
                cfg_local.MODEL.SYNC_BN = True
            else:
                cfg_local.MODEL.SYNC_BN = False

        if cfg_local.INPUT.BATCH_SIZE_TRAIN is None:
            cfg_local.INPUT.BATCH_SIZE_TRAIN = 4 * len(cfg_local.SYSTEM.GPU_IDS)

        if cfg_local.INPUT.BATCH_SIZE_TEST is None:
            cfg_local.INPUT.BATCH_SIZE_TEST = cfg_local.INPUT.BATCH_SIZE_TRAIN

        cfg_local.freeze()
        self.device = torch.device("cuda:0" if cfg_local.SYSTEM.USE_GPU else "cpu")

        self.mean_tensor = torch.FloatTensor(cfg_local.INPUT.NORM_MEAN)[None, :, None, None].to(self.device)
        self.std_tensor = torch.FloatTensor(cfg_local.INPUT.NORM_STD)[None, :, None, None].to(self.device)

        # Define network
        sys.path.insert(0, self.code_dir)
        kwargs = {'cfg': cfg_local}
        spec = importlib.util.spec_from_file_location("models", os.path.join(self.exp_dir, "net", "models.py"))
        model_module = spec.loader.load_module()
        print(self.exp_dir, model_module)
        self.model = getattr(model_module, cfg_local.MODEL.NET)(**kwargs)
        sys.path = sys.path[1:]

        if cfg_local.EXPERIMENT.RESUME_CHECKPOINT is not None:
            if not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT):
                raise RuntimeError("=> no checkpoint found at '{}'".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT))
            checkpoint = torch.load(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")
            if cfg_local.SYSTEM.USE_GPU and torch.cuda.device_count() > 1:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT,
                                                                checkpoint['epoch']))
            del checkpoint
        else:
            raise RuntimeError("=> model checkpoint has to be provided for testing!")

        # Using cuda
        self.model.to(self.device)
        self.model.eval()

        to_del = []
        for k, v in sys.modules.items():
            if k[:3] == "net":
                to_del.append(k)
        for k in to_del:
            del sys.modules[k]

    def evaluate(self, image):
        img = (image.to(self.device) - self.mean_tensor) / self.std_tensor
        with torch.no_grad():
            output = self.model(img)
        return output["anomaly_score"][:, 0, ...]


def get_model():
    params = {"exp_dir": "./jsr_code"}
    evaluator = ReconAnom(**params)

    return evaluator


if __name__ == "__main__":
    params = {"exp_dir": "."}
    evaluator = ReconAnom(**params)
    img = torch.rand((2, 3, 1024, 1024)).cuda()
    # img as tensor in shape [B, 3, H, W] with values in range (0,1)
    out = evaluator.evaluate(img)
    plt.imshow(out[1].cpu())
    plt.show()
    print(out)
