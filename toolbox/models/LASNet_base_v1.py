import os
import torch.nn as nn
import torch
#from resnet import Backbone_ResNet152_in3
import torch.nn.functional as F
import numpy as np
from toolbox.dual_self_att import CAM_Module
from .backbone_vit import EfficientViTBackbone,EfficientViTLargeBackbone,efficientvit_backbone_b1
from .res2net_v1b_base import Res2Net_model
from .seg import SegHead
from .seg_model_zoo import create_seg_model


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
        self.semantic_pred3 = nn.Conv2d(channel2, 1, kernel_size=3, padding=1)
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
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x5, x4, x3, x2, x1):
        #print(x5.shape) torch.Size([4, 256, 15, 20])
        x5_decoder = self.decoder5(x5)
        
        x5_decoder = F.interpolate(x5_decoder, size=x4.size()[2:], mode="bilinear", align_corners=True)
        
        x4_decoder = self.decoder4(x5_decoder + x4)
        
        
        x3_decoder = self.decoder3(x4_decoder + x3)
        #print(f"x3_decoder shape: {x3_decoder.shape}")
        semantic_pred3 =  self.semantic_pred3(x3_decoder) 
        #print(f"semantic_pred3 shape: {semantic_pred3.shape}")
        
        #x2_resized = F.interpolate(x2, size=x3_decoder.shape[2:], mode='bilinear', align_corners=False)
        #semantic_pred2 =  self.decoder3(x3_decoder + x2_resized)
        
        x2_decoder = self.decoder2(x3_decoder + x2)
        #print(f"x2_decoder shape: {x2_decoder.shape}")
        semantic_pred2 =  self.semantic_pred2(x2_decoder)
        #print(f"semantic_pred2 shape: {semantic_pred2.shape}")
        
        x1_resized = F.interpolate(x1, size=x2_decoder.shape[2:], mode='bilinear', align_corners=False)
        semantic_pred =  self.decoder1(x2_decoder + x1_resized)
        #print(f"semantic_pred shape: {semantic_pred2.shape}")
        return semantic_pred,semantic_pred2,semantic_pred3

def load_state_dict_from_file(file: str, only_state_dict=True):
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint





# def norm_layer(channel, norm_name='gn'):
#     if norm_name == 'bn':
#         return nn.BatchNorm2d(channel)
#     elif norm_name == 'gn':
#         return nn.GroupNorm(min(32, channel // 4), channel)


class ChannelCompression(nn.Module):
    def __init__(self, in_c, out_c=64):
        super(ChannelCompression, self).__init__()
        intermediate_c = in_c // 4 if in_c >= 256 else 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, intermediate_c, 1, bias=False),
            norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, intermediate_c, 3, 1, 1, bias=False),
            norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, out_c, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 4, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


import torch
import torch.nn as nn

class CrossAttentionFusionPool(nn.Module):
    def __init__(self, channel, dilation, kernel=5, norm_layer=nn.BatchNorm2d):
        super(CrossAttentionFusionPool, self).__init__()
        self.spatial_att_1 = SpatialAttention()
        self.spatial_att_2 = SpatialAttention()
        self.channel_att_1 = ChannelAttention(channel=channel)
        self.channel_att_2 = ChannelAttention(channel=channel)
        self.pool_size = 2 * (kernel - 1) * dilation + 1
        self.pool1 = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        self.pool2 = nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        self.gap = nn.AdaptiveMaxPool3d((2, 1, 1))
        # 3D卷积部分
        self.d_conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=1, dilation=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, kernel, kernel), stride=1, dilation=(1, dilation, dilation), 
                      padding=(0, dilation, dilation), bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=1, dilation=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, kernel, kernel), stride=1, dilation=(1, dilation, dilation), 
                      padding=(0, dilation, dilation), bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=2)
        # 融合后的特征精炼
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, depth):
        # 跨模态注意力
        rgb_1 = rgb * self.spatial_att_1(depth)
        depth_1 = depth * self.spatial_att_2(rgb)
        rgb_1 = rgb_1 * self.channel_att_1(rgb_1)
        depth_1 = depth_1 * self.channel_att_2(depth_1)

        # 池化和堆叠
        rgb_2 = self.pool1(rgb_1)
        depth_2 = self.pool2(depth_1)

        rgb_2 = rgb_2.unsqueeze(2)  # 增加维度
        depth_2 = depth_2.unsqueeze(2)
        f = torch.cat([rgb_2, depth_2], dim=2)  # 堆叠两个模态特征

        # 3D卷积处理
        f = self.d_conv1(self.d_conv2(f))
        f = self.gap(f)  # 全局自适应池化
        f = self.softmax(f)  # 权重归一化

        # 融合特征加权求和
        fused = f[:, :, 0, :, :] * rgb_1 + f[:, :, 1, :, :] * depth_1
        
        #f_final = rgb_1 + depth_1 + fused
        
        fused = self.fuse_refine(fused)
        
        return fused
        

    
class LASNet(nn.Module):
    def __init__(self, n_classes=1,ind=50):
        super(LASNet, self).__init__()
        
        # backbone == efficientvit
        #self.backbone = efficientvit_backbone_b1()
        #self.backbone1 =efficientvit_backbone_b1()
        
        # backbone == Res2Net50
        self.backbone=Res2Net_model(ind)
        self.backbone1 =Res2Net_model(ind)
        
        
        self.hfm4 = CrossAttentionFusionPool(channel=2048, dilation = 1)
        self.hfm3 = CrossAttentionFusionPool(channel=1024, dilation = 2)
        self.hfm2 = CrossAttentionFusionPool(channel=512, dilation = 2)
        self.hfm1 = CrossAttentionFusionPool(channel=256, dilation = 4)
        self.hfm0 = CrossAttentionFusionPool(channel=64, dilation = 4)
        
        
        
        
        #weight = load_state_dict_from_file('./checkpoints/b1-r288.pt')
        #model_dict = self.backbone1.state_dict()
        #weight = {k.replace('backbone.',''): v for k, v in weight.items() if k.replace('backbone.','') in model_dict}
        #print(weight.keys()==model_dict.keys())
        #model_dict.update(weight)
        #self.backbone1.load_state_dict(weight)
        
        

        # reduce the channel number, input: 480 640
        #self.decoder = prediction_decoder(16,32,64,128,256, n_classes)
        #self.decoder = prediction_decoder(64,256,512,1024,2048, n_classes)
        self.decoder = prediction_decoder(64,256,512,1024,2048, n_classes)

    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)  
        
        rgb_dict = self.backbone(x)
        t_dict = self.backbone1(ir)

        # 原本的特征融合
        # x4 = rgb_dict[4] + t_dict[4]
        # x3 = rgb_dict[3] + t_dict[3]
        # x2 = rgb_dict[2] + t_dict[2]
        # x1 = rgb_dict[1] + t_dict[1]
        # x0 = rgb_dict[0] + t_dict[0]
        
        #rgb_dict 的维度 [B, C, H, W]
        #rgb_dict[4] torch.Size([4, 2048, 15, 20])
        #rgb_dict[3] torch.Size([4, 1024, 30, 40])
        #rgb_dict[2] torch.Size([4, 512, 60, 80])
        #rgb_dict[1] torch.Size([4, 256, 120, 160])
        #rgb_dict[0] torch.Size([4, 64, 120, 160])
        
        #t_dict 的维度 [B, C, H, W]
        #t_dict[4] torch.Size([4, 2048, 15, 20])
        #t_dict[3] torch.Size([4, 1024, 30, 40])
        #t_dict[2] torch.Size([4, 512, 60, 80])
        #t_dict[1] torch.Size([4, 256, 120, 160])
        #t_dict[0] torch.Size([4, 64, 120, 160])

        
        # 下面是送入到decoder前的维度 [B, C, H, W]
        # x4:  torch.Size([4, 2048, 15, 20])
        # x3:  torch.Size([4, 1024, 30, 40])
        # x2:  torch.Size([4, 512, 60, 80])
        # x1:  torch.Size([4, 256, 120, 160])
        # x0:  torch.Size([4, 64, 120, 160])
        
        #semantic, semantic2 = self.decoder(x4,x3,x2,x1,x0)
        
        
        x4 = self.hfm4(rgb_dict[4], t_dict[4])
        x3 = self.hfm3(rgb_dict[3], t_dict[3])
        x2 = self.hfm2(rgb_dict[2], t_dict[2])
        x1 = self.hfm1(rgb_dict[1], t_dict[1])
        x0 = self.hfm0(rgb_dict[0], t_dict[0])
        
        semantic, semantic2, semantic3 = self.decoder(x4,x3,x2,x1,x0)
        semantic2 = torch.nn.functional.interpolate(semantic2, scale_factor=2, mode='bilinear')
        semantic3 = torch.nn.functional.interpolate(semantic3, scale_factor=4, mode='bilinear')

        
        return semantic, semantic2, semantic3, rgb_dict[4], t_dict[4]
        

if __name__ == '__main__':
    LASNet(9)
    # for PST900 dataset
    # LASNet(5)
