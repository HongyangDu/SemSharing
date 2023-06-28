import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from net.aspp import build_aspp
from net.decoder import build_decoder
from net.backbone import build_backbone
from net.reconstruction_decoder import ReconstructionDecoder
from net.sync_batchnorm.replicate import patch_replication_callback

class DeepLab(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DeepLab, self).__init__()

        output_stride = cfg.MODEL.OUT_STRIDE
        if cfg.MODEL.BACKBONE == 'drn':
            output_stride = 8

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(cfg.MODEL.BACKBONE, output_stride, BatchNorm)
        self.aspp = build_aspp(cfg.MODEL.BACKBONE, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, cfg.MODEL.BACKBONE, BatchNorm)

        self.freeze_bn = cfg.MODEL.FREEZE_BN 

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class DeepLabCommon(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(DeepLabCommon, self).__init__()
        self.freeze_bn = cfg.MODEL.FREEZE_BN 
        
        # inicialize trained segmentation model
        self.deeplab = DeepLab(cfg, cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS)
        if not os.path.isfile(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL):
            raise RuntimeError("=> pretrained segmentation model not found at '{}'" .format(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL))
        checkpoint = torch.load(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL, map_location="cpu")
        self.deeplab.load_state_dict(checkpoint['state_dict'])
        for parameter in self.deeplab.parameters():
            parameter.requires_grad = False
        del checkpoint
        
        # Reconstruction block 
        # 1) reconstruction decoder from segmentation model encoder features through small bottleneck
        self.recon_dec = ReconstructionDecoder(cfg)

    def forward(self, input):
        return None 

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_10x_lr_params(self):
        return None        
    
    def get_1x_lr_params(self):
        return None


class DeepLabRecon(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabRecon, self).__init__(cfg, **kwargs)


    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            x = self.deeplab.decoder(x, low_level_feat)
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        recon, recon_loss, bl = self.recon_dec(input, encoder_feat, low_level_feat)
        
        return {"input": input,
                "segmentation": segmentation,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "recon_bottleneck": bl,
                "anomaly_score": recon_loss, 
                }


class DeepLabReconFuseSimple(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabReconFuseSimple, self).__init__(cfg, **kwargs)
        
        # 2) merging of multiclass segmentation ouput and road reconstruction loss
        self.fuse_conv = nn.Sequential(
               nn.Conv2d(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS, 8, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(8),
               nn.ReLU(inplace=True),
               nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0),
               nn.BatchNorm2d(2),
               nn.ReLU(inplace=True))

    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            x = self.deeplab.decoder(x, low_level_feat)
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        recon, recon_loss, bl = self.recon_dec(input, encoder_feat, low_level_feat)
        x = self.fuse_conv(segmentation)
        
        perpixel = F.softmax(x * torch.cat([recon_loss, 1-recon_loss], dim=1), dim=1)[:, 0:1, ...]

        return {"input": input,
                "segmentation": segmentation,
                "binary_segmentation": x,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "recon_bottleneck": bl,
                "anomaly_score": perpixel, 
                }


class DeepLabReconFuseSimpleTrain(DeepLabCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabReconFuseSimpleTrain, self).__init__(cfg, **kwargs)
        
        # 2) merging of multiclass segmentation ouput and road reconstruction loss
        self.fuse_conv = nn.Sequential(
               nn.Conv2d(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS+1, 8, kernel_size=3, stride=1, padding=1),
               nn.BatchNorm2d(8),
               nn.ReLU(inplace=True),
               nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0),
               nn.BatchNorm2d(2),
               nn.ReLU(inplace=True))

    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            x = self.deeplab.decoder(x, low_level_feat)
            segmentation = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        recon, recon_loss, bl = self.recon_dec(input, encoder_feat, low_level_feat)
        x = self.fuse_conv(torch.cat([segmentation, recon_loss], dim=1))
        
        perpixel = F.softmax(x, dim=1)[:, 0:1, ...]

        return {"input": input,
                "segmentation": segmentation,
                "binary_segmentation": x,
                "recon_img": recon,
                "recon_loss": recon_loss,
                "recon_bottleneck": bl,
                "anomaly_score": perpixel, 
                }

