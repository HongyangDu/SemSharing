import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss(object):
    def __init__(self, loss_cfg, weight=None, use_cuda=True, **kwargs):
        self.ignore_index = loss_cfg.IGNORE_LABEL
        self.weight = weight
        self.size_average = loss_cfg.SIZE_AVG
        self.batch_average = loss_cfg.BATCH_AVG 
        self.cuda = use_cuda

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

class FocalLoss(object):
    def __init__(self, loss_cfg, weight=None, use_cuda=True, **kwargs):
        self.ignore_index = loss_cfg.IGNORE_LABEL
        self.weight = weight
        self.size_average = loss_cfg.SIZE_AVG
        self.batch_average = loss_cfg.BATCH_AVG 
        self.cuda = use_cuda
    
    def __call__(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction="mean" if self.size_average else "sum")
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class ReconstructionAnomalyLoss(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL

    def __call__(self, output, target):

        inv_mask = (target == 0).float()[:, None, ...]
        mask_local = (target == 1).float()[:, None, ...]
        mask_count = (mask_local > 0).float().sum() + 1e-7
        inv_mask_count = (inv_mask > 0).float().sum() + 1e-7
        
        loss = output["recon_loss"]
        margin = 0.999
        loss_road = (F.relu(loss - (1 - margin)) * mask_local).sum() / mask_count
        loss_bg = (F.relu(margin - loss) * inv_mask).sum() / inv_mask_count  
        return loss_road + loss_bg


class ReconstructionAnomalyLossFuseSimple(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL
        self.nll_loss = nn.NLLLoss(ignore_index=self.ignore_label, reduction="mean")

    def __call__(self, output, target):
        recon_loss_2class = torch.cat([output["recon_loss"], 1-output["recon_loss"]], dim=1)
        log_softmax = F.log_softmax(output["binary_segmentation"], dim=1) + F.log_softmax(recon_loss_2class, dim=1)
        xent_loss = self.nll_loss(log_softmax, target.long())

        return xent_loss


class ReconstructionAnomalyLossFuseSimpleTrain(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL
        self.nll_loss = nn.NLLLoss(ignore_index=self.ignore_label, reduction="mean")

    def __call__(self, output, target):
        log_softmax = F.log_softmax(output["binary_segmentation"], dim=1)
        xent_loss = self.nll_loss(log_softmax, target.long())

        return xent_loss


class ReconstructionAnomalyLossFuseTrainAux(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL
        self.nll_loss = nn.NLLLoss(ignore_index=self.ignore_label, reduction="mean")

    def __call__(self, output, target):

        inv_mask = (target == 0).float()[:, None, ...]
        mask_local = (target == 1).float()[:, None, ...]
        mask_count = (mask_local > 0).float().sum() + 1e-7
        inv_mask_count = (inv_mask > 0).float().sum() + 1e-7
        
        loss = output["recon_loss"]
        margin = 0.999
        loss_road = (F.relu(loss - (1 - margin)) * mask_local).sum() / mask_count
        loss_bg = (F.relu(margin - loss) * inv_mask).sum() / inv_mask_count  

        log_softmax = F.log_softmax(output["binary_segmentation"], dim=1)
        xent_loss = self.nll_loss(log_softmax, target.long())

        return xent_loss + 0.5*(loss_road + loss_bg)

