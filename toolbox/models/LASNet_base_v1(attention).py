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

#from fusion.MAGNet import GFM, MAFM


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
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x5, x4, x3, x2, x1):
        #print(x5.shape) torch.Size([4, 256, 15, 20])
        x5_decoder = self.decoder5(x5)
        #print(xxx)
        # for PST900 dataset
        # since the input size is 720x1280, the size of x5_decoder and x4_decoder is 23 and 45, so we cannot use 2x upsampling directrly.
        x5_decoder = F.interpolate(x5_decoder, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x4_decoder = self.decoder4(x5_decoder + x4)
        x3_decoder = self.decoder3(x4_decoder + x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        #semantic_pred2 =  self.sigmoid(self.semantic_pred2(x2_decoder))
        semantic_pred2 =  self.semantic_pred2(x2_decoder)
        
        #x2_decoder = F.interpolate(x2_decoder, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x1_resized = F.interpolate(x1, size=x2_decoder.shape[2:], mode='bilinear', align_corners=False)
        #print(x2_decoder.shape,x1_resized.shape)
        #print(ss)
        #semantic_pred =  self.sigmoid(self.decoder1(x2_decoder + x1_resized))
        semantic_pred =  self.decoder1(x2_decoder + x1_resized)
    
        return semantic_pred,semantic_pred2

def load_state_dict_from_file(file: str, only_state_dict=True):
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint








import torch
import torch.nn as nn

class HFM(nn.Module):
    def __init__(self, num_channels):
        """
        :param num_channels: 输入特征的通道数
        """
        super(HFM, self).__init__()

        # 模态交互模块
        self.interaction_module = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 差异特征提取模块
        self.diff_module = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 多尺度卷积模块
        self.multi_scale_module = nn.ModuleList([
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),  # 3x3卷积
            nn.Conv2d(num_channels, num_channels, kernel_size=5, padding=2),  # 5x5卷积
            nn.Conv2d(num_channels, num_channels, kernel_size=7, padding=3)   # 7x7卷积
        ])

        # 最终融合模块
        self.fusion_module = nn.Sequential(
            nn.Conv2d(num_channels * 3, num_channels, kernel_size=1),  # 通道整合
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 输出调整模块
        self.output_adjustment = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

    def forward(self, rgb_features, depth_features):
        """
        :param rgb_features: RGB模态输入特征 (batch_size, num_channels, H, W)
        :param depth_features: 深度模态输入特征 (batch_size, num_channels, H, W)
        :return: 融合后的输出特征
        """
        # 模态交互
        concatenated = torch.cat((rgb_features, depth_features), dim=1)
        interacted = self.interaction_module(concatenated)

        # 差异特征提取
        diff_features = torch.abs(rgb_features - depth_features)
        diff_enhanced = self.diff_module(diff_features)

        # 多尺度特征提取
        multi_scale_features = [
            conv_layer(interacted + diff_enhanced) for conv_layer in self.multi_scale_module
        ]
        multi_scale_combined = torch.cat(multi_scale_features, dim=1)

        # 融合和输出调整
        fused_features = self.fusion_module(multi_scale_combined)
        output = self.output_adjustment(fused_features)

        return output

        

    
class LASNet(nn.Module):
    def __init__(self, n_classes=1,ind=50):
        super(LASNet, self).__init__()
        
        # backbone == efficientvit
        #self.backbone = efficientvit_backbone_b1()
        #self.backbone1 =efficientvit_backbone_b1()
        
        # backbone == Res2Net50
        self.backbone=Res2Net_model(ind)
        self.backbone1 =Res2Net_model(ind)
        
        
        self.hfm4 = HFM(num_channels=2048)
        self.hfm3 = HFM(num_channels=1024)
        self.hfm2 = HFM(num_channels=512)
        self.hfm1 = HFM(num_channels=256)
        self.hfm0 = HFM(num_channels=64)
        
        
        
        
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
        
        
        x4 = self.hfm4(t_dict[4], rgb_dict[4])
        x3 = self.hfm3(t_dict[3], rgb_dict[3])
        x2 = self.hfm2(t_dict[2], rgb_dict[2])
        x1 = self.hfm1(t_dict[1], rgb_dict[1])
        x0 = self.hfm0(t_dict[0], rgb_dict[0])
        
        semantic, semantic2 = self.decoder(x4,x3,x2,x1,x0)
        semantic2 = torch.nn.functional.interpolate(semantic2, scale_factor=2, mode='bilinear')

        
        return semantic, semantic2, rgb_dict[4], t_dict[4]
        

if __name__ == '__main__':
    LASNet(9)
    # for PST900 dataset
    # LASNet(5)
