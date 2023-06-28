import os
import torch
import numpy as np
import cv2
import matplotlib
import copy
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, cfg, directory):
        self.directory = directory
        self.current_cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
        self.current_cmap.set_bad(color='black')
        self.dump_dir_tr = os.path.join(self.directory, "vis", "train")
        self.dump_dir_val = os.path.join(self.directory, "vis", "val")
        os.makedirs(self.dump_dir_tr, exist_ok=True)
        os.makedirs(self.dump_dir_val, exist_ok=True)
        self.img_mean = np.array(cfg.INPUT.NORM_MEAN)[None, None, :]
        self.img_std = np.array(cfg.INPUT.NORM_STD)[None, None, :]

    def denormalize_img(self, img):
        return img*self.img_std + self.img_mean

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
    
    def decode_target(self, segm):
        labels = [0, 1]
        colors = [[255, 0, 0],[128,64,128]]
        new_segm = 255*np.ones(shape=list(segm.shape)+[3], dtype=np.uint8)
        for l in labels:
            new_segm[segm==l, :] = colors[l] 
        return new_segm

    def visualize_image(self, writer, dataset, image, target, output, global_step, epoch, epoch_id, validation=False):
        if isinstance(output, dict):
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Image', grid_image, global_step)
            grid_image = make_grid(output["anomaly_score"][:3].clone().cpu().data, 3, normalize=False, range=(0, 1))
            writer.add_image('Anomaly score', grid_image, global_step)
            
            segm_map = (255*decode_seg_map_sequence(torch.max(output["segmentation"], 1)[1].detach().cpu().numpy(), dataset=dataset).numpy()).astype(np.uint8)
            segm_masked = output.get("segmentation_masked", None)
            second_segm = 0 
            if segm_masked is not None:
                second_segm = 1 
                segm_map2 = (255*decode_seg_map_sequence(torch.max(segm_masked, 1)[1].detach().cpu().numpy(), dataset=dataset).numpy()).astype(np.uint8)

            target_map = self.decode_target(target.detach().cpu().numpy())
            w = image.size()[3]
            h = image.size()[2]
            b = image.size()[0]
            out_img = np.zeros(shape=[6*h + second_segm*h, b*w, 3], dtype=np.uint8)
            for i in range(0, b):
                out_img[:h, i*w:(i+1)*w, :] = (255*self.denormalize_img(image[i, ...].clone().detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8)
                err_img = (255*self.current_cmap(output["anomaly_score"][i, 0, ...].clone().detach().cpu().numpy())).astype(np.uint8)[:, :, :-1]
                out_img[h:2*h, i*w:(i+1)*w, :] = err_img
                out_img[2*h:3*h, i*w:(i+1)*w, :] = (255*output["recon_img"][i, ...].clone().detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
                err_img = (255*self.current_cmap(output["recon_loss"][i, 0, ...].clone().detach().cpu().numpy())).astype(np.uint8)[:, :, :-1]
                out_img[3*h:4*h, i*w:(i+1)*w, :] = err_img
                out_img[4*h:5*h, i*w:(i+1)*w, :] = segm_map[i, ...].transpose(1, 2, 0)
                out_img[5*h:6*h, i*w:(i+1)*w, :] = target_map[i, ...]
                if second_segm > 0:
                    out_img[6*h:7*h, i*w:(i+1)*w, :] = segm_map2[i, ...].transpose(1, 2, 0)

            out_img = cv2.resize(out_img, (int(out_img.shape[1]/2), int(out_img.shape[0]/2)))
            cv2.imwrite(os.path.join(self.dump_dir_val if validation else self.dump_dir_tr, "e{:04d}_i{:08d}.jpg".format(epoch, epoch_id)), out_img[:, :, ::-1])
        else:
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Image', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)
