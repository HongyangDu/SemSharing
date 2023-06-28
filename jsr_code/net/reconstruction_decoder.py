import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.models as models
from math import exp
from net.aspp import ASPP


class ReconstructionDecoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ReconstructionDecoder, self).__init__()
        self.layers_dim = [16, 32, 64, 128]
        self.latent_dim = cfg.MODEL.RECONSTRUCTION.LATENT_DIM

        self.mean_tensor = torch.FloatTensor(cfg.INPUT.NORM_MEAN)[None, :, None, None]
        self.std_tensor = torch.FloatTensor(cfg.INPUT.NORM_STD)[None, :, None, None]

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.bottleneck = ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=self.latent_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True)

        self.skip_dim = 0 
        if cfg.MODEL.RECONSTRUCTION.SKIP_CONN:
            self.skip_dim = cfg.MODEL.RECONSTRUCTION.SKIP_CONN_DIM 
            self.skip_conn = nn.Sequential(nn.Conv2d(256, self.skip_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                           nn.BatchNorm2d(self.skip_dim),
                                           nn.ReLU())

        self.dec_layer0 = conv_block(self.latent_dim, self.layers_dim[3])
        self.dec_layer1 = conv_block(self.layers_dim[3], self.layers_dim[2])
        self.dec_layer2 = conv_block(self.layers_dim[2]+self.skip_dim, self.layers_dim[1])
        self.dec_layer3 = conv_block(self.layers_dim[1], self.layers_dim[0])
        self.final_layer = nn.Conv2d(self.layers_dim[0], out_channels=3, kernel_size=1, stride=1, padding=0)

        self.recon_loss = SSIMLoss(window_size=11, absval=True) 


    def forward(self, img, encoder_feat, low_level_feat):
        bl = self.bottleneck(encoder_feat)
        # decoder
        d_l0 = self.dec_layer0(self.up2(bl))
        d_l1 = self.dec_layer1(self.up2(d_l0))
        if self.skip_dim > 0:
            skip = F.interpolate(self.skip_conn(low_level_feat), size=d_l1.size()[2:], mode='bilinear', align_corners=True)
            d_l1 = torch.cat([d_l1, skip], dim=1) 
        d_l2 = self.dec_layer2(self.up2(d_l1))
        d_l3 = self.dec_layer3(self.up_size(d_l2, img.shape[2:]))
        
        #print ("encoded feat: ", encoder_feat.size())
        #print ("LOW: ", low_level_feat.size())
        #print ("BL: ", bl.size())
        #print ("d0: ", d_l0.size())
        #print ("d1: ", d_l1.size())
        #print ("d2: ", d_l2.size())
        #print ("d3: ", d_l3.size())
        if img.is_cuda and self.mean_tensor.get_device() != img.get_device():
            self.mean_tensor = self.mean_tensor.cuda(img.get_device())
            self.std_tensor = self.std_tensor.cuda(img.get_device())

        recons = torch.clamp(self.final_layer(d_l3)*self.std_tensor+self.mean_tensor, 0, 1)
        recons_loss = self.recon_loss(recons, torch.clamp(img*self.std_tensor+self.mean_tensor, 0, 1))[:, None, ...]

        return [recons, recons_loss, bl]


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
           )


# modified code from https://github.com/Po-Hsun-Su/pytorch-ssim
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size, absval):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.absval = absval
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, recons, input):
        (_, channel, _, _) = input.size()
        if channel == self.channel and self.window.data.type() == input.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if input.is_cuda:
                window = window.cuda(input.get_device())
            window = window.type_as(input)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(input, self.window, padding = self.window_size//2, groups = self.channel)
        mu2 = F.conv2d(recons, self.window, padding = self.window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(input*input, self.window, padding = self.window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(recons*recons, self.window, padding = self.window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(input*recons, self.window, padding = self.window_size//2, groups = self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if self.absval:
            return 1-torch.clamp(torch.abs(ssim_map), 0, 1).mean(1)
        else:
            return 1-torch.clamp(ssim_map, 0, 1).mean(1)


