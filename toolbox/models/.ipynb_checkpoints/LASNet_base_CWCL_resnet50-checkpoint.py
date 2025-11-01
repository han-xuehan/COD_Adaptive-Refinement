import os
import torch.nn as nn
import torch
#from resnet import Backbone_ResNet152_in3
import torch.nn.functional as F
import numpy as np
from toolbox.dual_self_att import CAM_Module
from .backbone_vit import EfficientViTBackbone,EfficientViTLargeBackbone,efficientvit_backbone_b1
#from toolbox.instance_whitening import InstanceWhitening
from .res2net_v1b_base import Res2Net_model
from .seg import SegHead
from .seg_model_zoo import create_seg_model
import timm

from .nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    LiteMLA,
)


        
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class prediction_decoder(nn.Module):
    def __init__(self, channel1=64, channel2=128, channel3=256, channel4=256, channel5=512, n_classes=1):
    #def __init__(self, channel1=64*2, channel2=256*2, channel3=512*2, channel4=1024*2, channel5=2048*2, n_classes=5):
        super(prediction_decoder, self).__init__()
        # 15 20
        self.decoder5 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel5, channel5, kernel_size=3, padding=3, dilation=3),
                #LiteMLA(channel5, channel5),
                BasicConv2d(channel5, channel4, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 30 40
        self.decoder4 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel4, channel4, kernel_size=3, padding=3, dilation=3),
                #LiteMLA(channel4, channel4),
                BasicConv2d(channel4, channel3, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 60 80
        self.decoder3 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel3, channel3, kernel_size=3, padding=3, dilation=3),
                #LiteMLA(channel3, channel3),
                BasicConv2d(channel3, channel2, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 120 160
        self.decoder2 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel2, channel2, kernel_size=3, padding=3, dilation=3),
                #LiteMLA(channel2, channel2),
                BasicConv2d(channel2, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        
        self.semantic_pred2 = nn.Conv2d(channel1, 1, kernel_size=3, padding=1)
        # 240 320 -> 480 640
        self.decoder1 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
                #LiteMLA(channel1, channel1),
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 480 640
                BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
                nn.Conv2d(channel1, 1, kernel_size=3, padding=1)
                )

    def forward(self, x5, x4, x3, x2, x1):
        #print(x5.shape) torch.Size([4, 256, 15, 20])
        
        
        x5_decoder = self.decoder5(x5)
        #print(xxx)
        # for PST900 dataset
        # since the input size is 720x1280, the size of x5_decoder and x4_decoder is 23 and 45, so we cannot use 2x upsampling directrly.
        x5_decoder_resized = F.interpolate(x5_decoder, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x4_decoder = self.decoder4(x5_decoder_resized + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        #semantic_pred2 =  self.sigmoid(self.semantic_pred2(x2_decoder))
        semantic_pred2 =  self.semantic_pred2(x2_decoder)
        
        x1_resized = F.interpolate(x1, size=x2_decoder.shape[2:], mode='bilinear', align_corners=False)
        
        x1_decoder = x2_decoder + x1_resized
        
        semantic_pred =  self.decoder1(x1_decoder)
    
        return semantic_pred,semantic_pred2, x5_decoder, x4_decoder, x3_decoder, x2_decoder#, x1_decoder

def load_state_dict_from_file(file: str, only_state_dict=True):
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint
    
    
class LASNet(nn.Module):
    def __init__(self, n_classes=1,ind=101):
        super(LASNet, self).__init__()
        #self.backbone=Res2Net_model(ind)
        #self.backbone1 =Res2Net_model(ind)
        self.rgb_encoder: nn.Module = timm.create_model(
            model_name="resnet50d", features_only=True, out_indices=range(0, 5)
        )
        self.depth_encoder: nn.Module = timm.create_model(
            model_name="resnet50d", features_only=True, out_indices=range(0, 5)
        )
        #if pretrained:
        self.rgb_encoder.load_state_dict(torch.load('./pretrained/resnet50d.pth', map_location="cpu"), strict=False)
        self.depth_encoder.load_state_dict(torch.load('./pretrained/resnet50d.pth', map_location="cpu"), strict=False)
        
        self.decoder = prediction_decoder(64,256,512,1024,2048, n_classes)
        #self.training = training
        #print(1,training)

    def forward(self, rgb, depth,training = True):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)  
        
        rgb_dict = self.rgb_encoder(x)
        t_dict = self.depth_encoder(ir)
        
        x4 = rgb_dict[4] + t_dict[4]
        x3 = rgb_dict[3] + t_dict[3]
        x2 = rgb_dict[2] + t_dict[2]
        x1 = rgb_dict[1] + t_dict[1]
        x0 = rgb_dict[0] + t_dict[0]
        
        semantic, semantic2,  x5_decoder_RGBD, x4_decoder_RGBD, x3_decoder_RGBD, x2_decoder_RGBD = self.decoder(x4,x3,x2,x1,x0)
        semantic2 = torch.nn.functional.interpolate(semantic2, scale_factor=2, mode='bilinear')
        RGBD_decoder=[x5_decoder_RGBD, x4_decoder_RGBD, x3_decoder_RGBD, x2_decoder_RGBD]
        
        if training:
            semantic_Depth, semantic2_Depth, x5_decoder_Depth, x4_decoder_Depth, x3_decoder_Depth, x2_decoder_Depth  = self.decoder(t_dict[4],t_dict[3],t_dict[2],t_dict[1],t_dict[0])
            Depth_decoder = [x5_decoder_Depth, x4_decoder_Depth, x3_decoder_Depth, x2_decoder_Depth]
            return semantic, semantic2, rgb_dict[4], t_dict[4], RGBD_decoder, Depth_decoder
        
        return semantic, semantic2

if __name__ == '__main__':
    LASNet()
    # for PST900 dataset
    # LASNet(5)
