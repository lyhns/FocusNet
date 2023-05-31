'''
Descripttion: 特征提取类:Feature_exa
              IF input(1,3,256,256)
              输出:
              FEB_1_vis: torch.Size([1, 32, 256, 256]) 
              FEB_2_GRDB_vis: torch.Size([1, 64, 256, 256]) 
              FEB_3_GRDB_vis: torch.Size([1, 128, 256, 256]) 
              FEB_4_vis: torch.Size([1, 256, 128, 128])
              FEB_5_vis: torch.Size([1, 512, 64, 64])
              
              图像融合类:FocusNet_Fusion
              
version: 
Author: lyh
Date: 2023-03-22 09:23:33
LastEditors: smile
LastEditTime: 2023-03-22 20:15:57
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
#from args.option import args
from torch.nn.utils import spectral_norm
import os
import cv2
import numpy as np
from torch.autograd import Variable
import model.ASA_attention as ASAattention
import model.GRDB as GRDB

# 定义torch卷积层
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True, activation='leakyrelu', dropout=False):
        super(Conv2d, self).__init__()
        padding = (kernel_size - 1) // 2  # 向下取整
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)  # 向下取整
        nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x

# 定义torch反卷积层
class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,  bn=False, activation='leakyrelu', dropout=False):
        super(Deconv2d, self).__init__()
        
        padding = (kernel_size ) // 2
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x
    
#特征提取
class Feature_exa(nn.Module):
    def __init__(self):
        super(Feature_exa, self).__init__()
        #CB conv block =conv + BN + leakyrelu
        self.FEB_1 = Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.ASA_1 = ASAattention.ASAattention(channel = 32, channel_out = 32, hw = 400)
        
        self.GRDB_2 = GRDB.RGBD(in_channels = 32, out_channels = 64)
        self.FEB_2 = Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.ASA_2 = ASAattention.ASAattention(channel = 64, channel_out = 64, hw = 400)
        
        self.GRDB_3 = GRDB.RGBD(in_channels = 64, out_channels = 128)
        self.FEB_3 = Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.ASA_3 = ASAattention.ASAattention(channel = 128, channel_out = 128, hw = 400)

        self.FEB_4 = Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.ASA_4 = ASAattention.ASAattention(channel = 256, channel_out = 256, hw = 400)
        
        self.FEB_5 = Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.ASA_5 = ASAattention.ASAattention(channel = 512, channel_out = 512, hw = 400)
        
    def forward(self,  img_input):
        #       V3加了GBDB
        # #层次1  FEB_1_vis
        # FEB_1 = self.FEB_1(img_input)
        # FEB_1 = self.ASA_1(FEB_1)
        
        # #层次2  FEB_2_GRDB
        # #FEB_2_GRDB = self.GRDB_2(FEB_1)
        # FEB_2 = self.FEB_2(FEB_1)
        # FEB_2 = self.ASA_2(FEB_2)
        
        # #层次3  FEB_3_GRDB_vis
        # #FEB_3_GRDB = self.GRDB_3(FEB_2_GRDB)
        # FEB_3 = self.FEB_3(FEB_2)
        # FEB_3 = self.ASA_3(FEB_3)
        
        # #层次4  FEB_4_vis
        # #FEB_4 = self.FEB_4(FEB_3_GRDB)
        # FEB_4 = self.FEB_4(FEB_3)
        # FEB_4 = self.ASA_4(FEB_4)
        
        # #层次5  FEB_5_vis
        # FEB_5 = self.FEB_5(FEB_4)
        # FEB_5 = self.ASA_5(FEB_5)
        
        #层次1  FEB_1_vis
        FEB_1 = self.FEB_1(img_input)
        FEB_1 = self.ASA_1(FEB_1)
        
        #层次2  FEB_2_GRDB
        FEB_2_GRDB = self.GRDB_2(FEB_1)
        FEB_2 = self.FEB_2(FEB_1)
        FEB_2 = self.ASA_2(FEB_2)
        
        #层次3  FEB_3_GRDB_vis
        FEB_3_GRDB = self.GRDB_3(FEB_2_GRDB)
        FEB_3 = self.FEB_3(FEB_2)
        FEB_3 = self.ASA_3(FEB_3)
        
        #层次4  FEB_4_vis
        #FEB_4 = self.FEB_4(FEB_3_GRDB)
        FEB_4 = self.FEB_4(FEB_3)
        FEB_4 = self.ASA_4(FEB_4)
        
        #层次5  FEB_5_vis
        FEB_5 = self.FEB_5(FEB_4)
        FEB_5 = self.ASA_5(FEB_5)
        # print(FEB_1_vis.shape,
        #       FEB_2_GRDB_vis.shape, 
        #       FEB_3_GRDB_vis.shape, 
        #       FEB_4_vis.shape, 
        #       FEB_5_vis.shape)
        return FEB_1, FEB_2_GRDB, FEB_3_GRDB, FEB_4, FEB_5
        #return FEB_1, FEB_2, FEB_3, FEB_4, FEB_5

class FocusNet_Fusion(nn.Module):
    def __init__(self):
        super(FocusNet_Fusion, self).__init__()
        #特征提取
        self.feature_extraction  = Feature_exa()
        #
        self.deconv_1 = Deconv2d(in_channels = 64, out_channels = 512, kernel_size = 3, 
                                           #padding = 1, 
                                           stride=1, bn=True, activation='leakyrelu', dropout=False)

        self.deconv_2 = Deconv2d(in_channels = 128, out_channels = 256, kernel_size = 3, 
                                           #padding = 1, 
                                           stride=1, bn=True, activation='leakyrelu', dropout=False)

        self.deconv_3 = Deconv2d(in_channels = 256, out_channels = 128, kernel_size = 3, 
                                           #padding = 1,
                                           stride=1, bn=True, activation='leakyrelu', dropout=False)
   
        self.deconv_4 = Deconv2d(in_channels = 512, out_channels = 64, kernel_size = 3, 
                                           #padding = 0, 
                                           stride=1, bn=True, activation='leakyrelu', dropout=False)

        self.deconv_5 = Deconv2d(in_channels = 1024, out_channels = 32, kernel_size = 3, 
                                           #padding = 0, 
                                           stride=1, bn=True, activation='leakyrelu', dropout=False)
        #self.deconv_5 = Deconv2d.Deconv2d(in_channels = 512, out_channels = 64, kernel_size = 4, 
        #                                    stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.bn_layer3_1 = nn.BatchNorm2d(768, eps=0.001, momentum=0, affine=True)
        self.bn_layer3_2 = nn.BatchNorm2d(384, eps=0.001, momentum=0, affine=True)
        self.bn_layer3_3 = nn.BatchNorm2d(192, eps=0.001, momentum=0, affine=True)
        self.bn_layer3_4 = nn.BatchNorm2d(96, eps=0.001, momentum=0, affine=True)
        
        self.ASA_layer3_2 = ASAattention.ASAattention(channel = 384, channel_out = 128, hw = 400)
        self.ASA_layer3_4 = ASAattention.ASAattention(channel = 96, channel_out = 128, hw = 400)
        # 放大
        #self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        #self.bn_layer4 = nn.BatchNorm2d(1216, eps=0.001, momentum=0, affine=True)
        #self.conv_layer5 = nn.Conv2d(1216, 608, 1)
        
        self.bn_layer4 = nn.BatchNorm2d(1440, eps=0.001, momentum=0, affine=True)
        self.conv_layer5 = nn.Conv2d(1440, 608, 1)
        #nn.init.xavier_uniform_(self.conv_layer5.weight)
        self.bn_layer5 = nn.BatchNorm2d(608, eps=0.001, momentum=0, affine=True)

        self.conv_layer6 = nn.Conv2d(608, 304, 1)
        #nn.init.xavier_uniform_(self.conv_layer6.weight)
        self.bn_layer6 = nn.BatchNorm2d(304, eps=0.001, momentum=0, affine=True)
        
        self.conv_layer7 = nn.Conv2d(304, 152, 1)
        #nn.init.xavier_uniform_(self.conv_layer7.weight)
        self.bn_layer7 = nn.BatchNorm2d(152, eps=0.001, momentum=0, affine=True) 
        #self.pool2d_layer7 = nn.MaxPool2d(2)
        
        self.conv_layer8 = nn.Conv2d(152, 76, 1)
        #nn.init.xavier_uniform_(self.conv_layer8.weight)
        self.bn_layer8 = nn.BatchNorm2d(76, eps=0.001, momentum=0, affine=True) 
        #self.pool2d_layer8 = nn.MaxPool2d(2)
        
        self.conv_layer9 = nn.Conv2d(76, 1, 1)
        #nn.init.xavier_uniform_(self.conv_layer9.weight)
        self.bn_layer9 = nn.BatchNorm2d(1, eps=0.001, momentum=0, affine=True)

        #self.activate = nn.LeakyReLU(negative_slope=0.2)

        
    def forward(self, img_IR,img_VIS):
        
        ####################特征提取图像#######################
        IR_layer_1,IR_layer_2,IR_layer_3,IR_layer_4,IR_layer_5 = self.feature_extraction(img_IR)
        VIS_layer_1,VIS_layer_2,VIS_layer_3,VIS_layer_4,VIS_layer_5 = self.feature_extraction(img_VIS)
        ################拼接后输入 第一层输出###################
        #----layer1_1
        layer1_1 = torch.cat((IR_layer_1,VIS_layer_1), 1)
        #print("layer1_1",torch.max(layer1_1,dim = 2))
        #----layer1_2
        layer1_2 = torch.cat((IR_layer_2,VIS_layer_2), 1)
        #print("layer1_2",torch.max(layer1_2,dim = 2))
        #----layer1_3
        layer1_3 = torch.cat((IR_layer_3,VIS_layer_3), 1)
        #print("layer1_3",torch.max(layer1_3,dim = 2))
        #----layer1_4
        layer1_4 = torch.cat((IR_layer_4,VIS_layer_4), 1)
        #print("layer1_4",torch.max(layer1_4,dim = 2))
        #----layer1_5
        layer1_5 = torch.cat((IR_layer_5,VIS_layer_5), 1)
        #print("layer1_5",torch.max(layer1_5,dim = 2))
        #print("layer1_1",layer1_1.shape,
        #      "layer1_2",layer1_2.shape,
        #      "layer1_3",layer1_3.shape,
        #      "layer1_4",layer1_4.shape,
        #      "layer1_5",layer1_5.shape)
        ###################第二层输出##########################
        
        #----layer2_1
        layer2_1 = self.deconv_1(layer1_1) #in：64 out：64
        #print("layer2_1",torch.max(layer2_1,dim = 2))
        #----layer2_2
        layer2_2 = self.deconv_2(layer1_2) #in：128 out：64
        #print("layer2_2",torch.max(layer2_2,dim = 2))
        #----layer2_3
        layer2_3 = self.deconv_3(layer1_3) #in：256 out：64
        #print("layer2_3",torch.max(layer2_3,dim = 2))
        #----layer2_4
        layer2_4 = self.deconv_4(layer1_4) #in：512 out：64
        #print("layer2_4",torch.max(layer2_4,dim = 2))
        #layer2_4 = self.up2(layer2_4)
        #----layer2_5
        layer2_5 = self.deconv_5(layer1_5) #in：1024 out：64
        #print("layer2_5",torch.max(layer2_5,dim = 2))
        #layer2_5 = self.up4(layer2_5)
        
        # print("layer2_1",layer2_1.shape,\
        #       "layer2_2",layer2_2.shape,\
        #       "layer2_3",layer2_3.shape,\
        #       "layer2_4",layer2_4.shape,\
        #       "layer2_5",layer2_5.shape)
        ######################第三层输出#########################
      
        #----layer3_1
        layer3_1 = torch.cat((layer2_1,layer2_2), 1) # out：128
        layer3_1 = self.bn_layer3_1(layer3_1)
        #print("layer3_1",torch.max(layer2_5,dim = 2))
        #----layer3_2
        layer3_2 = torch.cat((layer2_2,layer2_3), 1) # out：128 
        layer3_2 = self.bn_layer3_2(layer3_2)
        #layer3_2 = self.ASA_layer3_2(layer3_2)
        #print("layer3_2",torch.max(layer3_2,dim = 2))
        #----layer3_3
        layer3_3 = torch.cat((layer2_3,layer2_4), 1) # out：128 
        layer3_3 = self.bn_layer3_3(layer3_3)
        #print("layer3_3",torch.max(layer3_3,dim = 2))
        #----layer3_4
        layer3_4 = torch.cat((layer2_4,layer2_5), 1) # out：128 
        layer3_4 = self.bn_layer3_4(layer3_4)
        #layer3_4 = self.ASA_layer3_4(layer3_4)
        #print("layer3_4",torch.max(layer3_4,dim = 2))
        # print("layer3_1",layer3_1.shape,\
        #       "layer3_2",layer3_2.shape,\
        #       "layer3_3",layer3_3.shape,\
        #       "layer3_4",layer3_4.shape)
        #####################第四层输出##########################
        #第四层输出
         #----layer4
        layer4 = torch.cat((layer3_1,
                            layer3_2,
                            layer3_3,
                            layer3_4), 1)
        layer4 = self.bn_layer4(layer4)
        #print("layer4",torch.max(layer4,dim = 2))
        #----layer5
        layer5 = self.conv_layer5(layer4)
        layer5 = self.bn_layer5(layer5)
        #print("layer5",torch.max(layer5,dim = 2))
        #----layer6
        layer6 = self.conv_layer6(layer5)
        layer6 = self.bn_layer6(layer6)
        #print("layer6",torch.max(layer6,dim = 2))
        #----layer7
        layer7 = self.conv_layer7(layer6)
        layer7 = self.bn_layer7(layer7)
        #layer7 =self.pool2d_layer7(layer7)
        #print("layer7",torch.max(layer7,dim = 2))
        #----layer8
        layer8 = self.conv_layer8(layer7)
        layer8 = self.bn_layer8(layer8)
        #layer8 =self.pool2d_layer7(layer8)
        #print("layer8",torch.max(layer8,dim = 2))
        #----layer9
        layer9 = self.conv_layer9(layer8)
        #layer9 = self.bn_layer9(layer9)
        #print("layer9",torch.max(layer9,dim = 2))
        return layer9
        
'''
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
FocusNet_Fusion                               [1, 1, 400, 400]          5,164,802
├─Feature_exa: 1-1                            [1, 32, 400, 400]         --
│    └─Conv2d: 2-1                            [1, 32, 200, 200]         --
│    │    └─Conv2d: 3-1                       [1, 32, 200, 200]         544
│    │    └─BatchNorm2d: 3-2                  [1, 32, 200, 200]         64
│    │    └─LeakyReLU: 3-3                    [1, 32, 200, 200]         --
│    └─ASAattention: 2-2                      [1, 32, 400, 400]         64
│    │    └─ModuleList: 3-112                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-5            [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-6                      [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-7                    [1, 16, 200, 200]         32
│    │    └─Sigmoid: 3-8                      [1, 16, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-9            [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-10                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-11                   [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-12                     [1, 16, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-13           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-14                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-15                   [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-16                     [1, 16, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-17           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-18                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-19                   [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-20                     [1, 16, 200, 200]         --
│    └─RGBD: 2-3                              [1, 64, 400, 400]         --
│    │    └─DenseBlock: 3-21                  [1, 96, 400, 400]         27,712
│    │    └─Conv1: 3-22                       [1, 64, 400, 400]         6,208
│    │    └─Sobelxy: 3-23                     [1, 32, 400, 400]         576
│    │    └─Conv1: 3-24                       [1, 64, 400, 400]         2,112
│    └─Conv2d: 2-4                            [1, 64, 200, 200]         --
│    │    └─Conv2d: 3-25                      [1, 64, 200, 200]         32,832
│    │    └─BatchNorm2d: 3-26                 [1, 64, 200, 200]         128
│    │    └─LeakyReLU: 3-27                   [1, 64, 200, 200]         --
│    └─ASAattention: 2-5                      [1, 64, 400, 400]         128
│    │    └─ModuleList: 3-136                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-29           [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-30                     [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-31                   [1, 32, 200, 200]         64
│    │    └─Sigmoid: 3-32                     [1, 32, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-33           [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-34                     [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-35                   [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-36                     [1, 32, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-37           [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-38                     [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-39                   [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-40                     [1, 32, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-41           [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-42                     [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-43                   [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-44                     [1, 32, 200, 200]         --
│    └─RGBD: 2-6                              [1, 128, 400, 400]        --
│    │    └─DenseBlock: 3-45                  [1, 192, 400, 400]        110,720
│    │    └─Conv1: 3-46                       [1, 128, 400, 400]        24,704
│    │    └─Sobelxy: 3-47                     [1, 64, 400, 400]         1,152
│    │    └─Conv1: 3-48                       [1, 128, 400, 400]        8,320
│    └─Conv2d: 2-7                            [1, 128, 200, 200]        --
│    │    └─Conv2d: 3-49                      [1, 128, 200, 200]        131,200
│    │    └─BatchNorm2d: 3-50                 [1, 128, 200, 200]        256
│    │    └─LeakyReLU: 3-51                   [1, 128, 200, 200]        --
│    └─ASAattention: 2-8                      [1, 128, 400, 400]        256
│    │    └─ModuleList: 3-160                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-53           [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-54                     [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-55                   [1, 64, 200, 200]         128
│    │    └─Sigmoid: 3-56                     [1, 64, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-57           [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-58                     [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-59                   [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-60                     [1, 64, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-61           [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-62                     [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-63                   [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-64                     [1, 64, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-65           [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-66                     [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-67                   [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-68                     [1, 64, 200, 200]         --
│    └─Conv2d: 2-9                            [1, 256, 200, 200]        --
│    │    └─Conv2d: 3-69                      [1, 256, 200, 200]        524,544
│    │    └─BatchNorm2d: 3-70                 [1, 256, 200, 200]        512
│    │    └─LeakyReLU: 3-71                   [1, 256, 200, 200]        --
│    └─ASAattention: 2-10                     [1, 256, 400, 400]        512
│    │    └─ModuleList: 3-180                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-73           [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-74                     [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-75                   [1, 128, 200, 200]        256
│    │    └─Sigmoid: 3-76                     [1, 128, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-77           [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-78                     [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-79                   [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-80                     [1, 128, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-81           [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-82                     [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-83                   [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-84                     [1, 128, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-85           [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-86                     [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-87                   [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-88                     [1, 128, 200, 200]        --
│    └─Conv2d: 2-11                           [1, 512, 200, 200]        --
│    │    └─Conv2d: 3-89                      [1, 512, 200, 200]        2,097,664
│    │    └─BatchNorm2d: 3-90                 [1, 512, 200, 200]        1,024
│    │    └─LeakyReLU: 3-91                   [1, 512, 200, 200]        --
│    └─ASAattention: 2-12                     [1, 512, 400, 400]        1,024
│    │    └─ModuleList: 3-200                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-93           [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-94                     [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-95                   [1, 256, 200, 200]        512
│    │    └─Sigmoid: 3-96                     [1, 256, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-97           [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-98                     [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-99                   [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-100                    [1, 256, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-101          [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-102                    [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-103                  [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-104                    [1, 256, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-105          [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-106                    [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-107                  [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-108                    [1, 256, 200, 200]        --
├─Feature_exa: 1-2                            [1, 32, 400, 400]         (recursive)
│    └─Conv2d: 2-13                           [1, 32, 200, 200]         (recursive)
│    │    └─Conv2d: 3-109                     [1, 32, 200, 200]         (recursive)
│    │    └─BatchNorm2d: 3-110                [1, 32, 200, 200]         (recursive)
│    │    └─LeakyReLU: 3-111                  [1, 32, 200, 200]         --
│    └─ASAattention: 2-14                     [1, 32, 400, 400]         (recursive)
│    │    └─ModuleList: 3-112                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-113          [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-114                    [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-115                  [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-116                    [1, 16, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-117          [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-118                    [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-119                  [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-120                    [1, 16, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-121          [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-122                    [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-123                  [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-124                    [1, 16, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-125          [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-126                    [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-127                  [1, 16, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-128                    [1, 16, 200, 200]         --
│    └─RGBD: 2-15                             [1, 64, 400, 400]         (recursive)
│    │    └─DenseBlock: 3-129                 [1, 96, 400, 400]         (recursive)
│    │    └─Conv1: 3-130                      [1, 64, 400, 400]         (recursive)
│    │    └─Sobelxy: 3-131                    [1, 32, 400, 400]         (recursive)
│    │    └─Conv1: 3-132                      [1, 64, 400, 400]         (recursive)
│    └─Conv2d: 2-16                           [1, 64, 200, 200]         (recursive)
│    │    └─Conv2d: 3-133                     [1, 64, 200, 200]         (recursive)
│    │    └─BatchNorm2d: 3-134                [1, 64, 200, 200]         (recursive)
│    │    └─LeakyReLU: 3-135                  [1, 64, 200, 200]         --
│    └─ASAattention: 2-17                     [1, 64, 400, 400]         (recursive)
│    │    └─ModuleList: 3-136                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-137          [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-138                    [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-139                  [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-140                    [1, 32, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-141          [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-142                    [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-143                  [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-144                    [1, 32, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-145          [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-146                    [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-147                  [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-148                    [1, 32, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-149          [1, 32, 1, 1]             --
│    │    └─Sigmoid: 3-150                    [1, 32, 1, 1]             --
│    │    └─GroupNorm: 3-151                  [1, 32, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-152                    [1, 32, 200, 200]         --
│    └─RGBD: 2-18                             [1, 128, 400, 400]        (recursive)
│    │    └─DenseBlock: 3-153                 [1, 192, 400, 400]        (recursive)
│    │    └─Conv1: 3-154                      [1, 128, 400, 400]        (recursive)
│    │    └─Sobelxy: 3-155                    [1, 64, 400, 400]         (recursive)
│    │    └─Conv1: 3-156                      [1, 128, 400, 400]        (recursive)
│    └─Conv2d: 2-19                           [1, 128, 200, 200]        (recursive)
│    │    └─Conv2d: 3-157                     [1, 128, 200, 200]        (recursive)
│    │    └─BatchNorm2d: 3-158                [1, 128, 200, 200]        (recursive)
│    │    └─LeakyReLU: 3-159                  [1, 128, 200, 200]        --
│    └─ASAattention: 2-20                     [1, 128, 400, 400]        (recursive)
│    │    └─ModuleList: 3-160                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-161          [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-162                    [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-163                  [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-164                    [1, 64, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-165          [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-166                    [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-167                  [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-168                    [1, 64, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-169          [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-170                    [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-171                  [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-172                    [1, 64, 200, 200]         --
│    │    └─AdaptiveAvgPool2d: 3-173          [1, 64, 1, 1]             --
│    │    └─Sigmoid: 3-174                    [1, 64, 1, 1]             --
│    │    └─GroupNorm: 3-175                  [1, 64, 200, 200]         (recursive)
│    │    └─Sigmoid: 3-176                    [1, 64, 200, 200]         --
│    └─Conv2d: 2-21                           [1, 256, 200, 200]        (recursive)
│    │    └─Conv2d: 3-177                     [1, 256, 200, 200]        (recursive)
│    │    └─BatchNorm2d: 3-178                [1, 256, 200, 200]        (recursive)
│    │    └─LeakyReLU: 3-179                  [1, 256, 200, 200]        --
│    └─ASAattention: 2-22                     [1, 256, 400, 400]        (recursive)
│    │    └─ModuleList: 3-180                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-181          [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-182                    [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-183                  [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-184                    [1, 128, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-185          [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-186                    [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-187                  [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-188                    [1, 128, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-189          [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-190                    [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-191                  [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-192                    [1, 128, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-193          [1, 128, 1, 1]            --
│    │    └─Sigmoid: 3-194                    [1, 128, 1, 1]            --
│    │    └─GroupNorm: 3-195                  [1, 128, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-196                    [1, 128, 200, 200]        --
│    └─Conv2d: 2-23                           [1, 512, 200, 200]        (recursive)
│    │    └─Conv2d: 3-197                     [1, 512, 200, 200]        (recursive)
│    │    └─BatchNorm2d: 3-198                [1, 512, 200, 200]        (recursive)
│    │    └─LeakyReLU: 3-199                  [1, 512, 200, 200]        --
│    └─ASAattention: 2-24                     [1, 512, 400, 400]        (recursive)
│    │    └─ModuleList: 3-200                 --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-201          [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-202                    [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-203                  [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-204                    [1, 256, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-205          [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-206                    [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-207                  [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-208                    [1, 256, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-209          [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-210                    [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-211                  [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-212                    [1, 256, 200, 200]        --
│    │    └─AdaptiveAvgPool2d: 3-213          [1, 256, 1, 1]            --
│    │    └─Sigmoid: 3-214                    [1, 256, 1, 1]            --
│    │    └─GroupNorm: 3-215                  [1, 256, 200, 200]        (recursive)
│    │    └─Sigmoid: 3-216                    [1, 256, 200, 200]        --
├─Deconv2d: 1-3                               [1, 512, 400, 400]        --
│    └─ConvTranspose2d: 2-25                  [1, 512, 400, 400]        295,424
│    └─BatchNorm2d: 2-26                      [1, 512, 400, 400]        1,024
│    └─LeakyReLU: 2-27                        [1, 512, 400, 400]        --
├─Deconv2d: 1-4                               [1, 256, 400, 400]        --
│    └─ConvTranspose2d: 2-28                  [1, 256, 400, 400]        295,168
│    └─BatchNorm2d: 2-29                      [1, 256, 400, 400]        512
│    └─LeakyReLU: 2-30                        [1, 256, 400, 400]        --
├─Deconv2d: 1-5                               [1, 128, 400, 400]        --
│    └─ConvTranspose2d: 2-31                  [1, 128, 400, 400]        295,040
│    └─BatchNorm2d: 2-32                      [1, 128, 400, 400]        256
│    └─LeakyReLU: 2-33                        [1, 128, 400, 400]        --
├─Deconv2d: 1-6                               [1, 64, 400, 400]         --
│    └─ConvTranspose2d: 2-34                  [1, 64, 400, 400]         294,976
│    └─BatchNorm2d: 2-35                      [1, 64, 400, 400]         128
│    └─LeakyReLU: 2-36                        [1, 64, 400, 400]         --
├─Deconv2d: 1-7                               [1, 32, 400, 400]         --
│    └─ConvTranspose2d: 2-37                  [1, 32, 400, 400]         294,944
│    └─BatchNorm2d: 2-38                      [1, 32, 400, 400]         64
│    └─LeakyReLU: 2-39                        [1, 32, 400, 400]         --
├─BatchNorm2d: 1-8                            [1, 768, 400, 400]        1,536
├─BatchNorm2d: 1-9                            [1, 384, 400, 400]        768
├─BatchNorm2d: 1-10                           [1, 192, 400, 400]        384
├─BatchNorm2d: 1-11                           [1, 96, 400, 400]         192
├─BatchNorm2d: 1-12                           [1, 1440, 400, 400]       2,880
├─Conv2d: 1-13                                [1, 608, 400, 400]        876,128
├─BatchNorm2d: 1-14                           [1, 608, 400, 400]        1,216
├─Conv2d: 1-15                                [1, 304, 400, 400]        185,136
├─BatchNorm2d: 1-16                           [1, 304, 400, 400]        608
├─Conv2d: 1-17                                [1, 152, 400, 400]        46,360
├─BatchNorm2d: 1-18                           [1, 152, 400, 400]        304
├─Conv2d: 1-19                                [1, 76, 400, 400]         11,628
├─BatchNorm2d: 1-20                           [1, 76, 400, 400]         152
├─Conv2d: 1-21                                [1, 1, 400, 400]          77
===============================================================================================
Total params: 40,086,315
Trainable params: 40,086,315
Non-trainable params: 0
Total mult-adds (T): 3.04
===============================================================================================
Input size (MB): 1.28
Forward/backward pass size (MB): 18730.24
Params size (MB): 139.68
Estimated Total Size (MB): 18871.20
===============================================================================================
'''

'''
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
FocusNet_Fusion                               [1, 1216, 256, 256]       --
├─Feature_exa: 1-1                            [1, 32, 256, 256]         --
│    └─Conv2d: 2-1                            [1, 32, 128, 128]         --
│    │    └─Conv2d: 3-1                       [1, 32, 128, 128]         1,568
│    │    └─BatchNorm2d: 3-2                  [1, 32, 128, 128]         64
│    │    └─ReLU: 3-3                         [1, 32, 128, 128]         --
│    └─ASAattention: 2-2                      [1, 32, 256, 256]         64
│    │    └─ModuleList: 3-38                  --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-5            [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-6                      [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-7                    [1, 16, 128, 128]         32
│    │    └─Sigmoid: 3-8                      [1, 16, 128, 128]         --
│    │    └─AdaptiveAvgPool2d: 3-9            [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-10                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-11                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-12                     [1, 16, 128, 128]         --
│    │    └─AdaptiveAvgPool2d: 3-13           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-14                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-15                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-16                     [1, 16, 128, 128]         --
│    │    └─AdaptiveAvgPool2d: 3-17           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-18                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-19                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-20                     [1, 16, 128, 128]         --
│    └─RGBD: 2-3                              [1, 64, 256, 256]         --
│    │    └─DenseBlock: 3-21                  [1, 96, 256, 256]         27,712
│    │    └─Conv1: 3-22                       [1, 64, 256, 256]         6,208
│    │    └─Sobelxy: 3-23                     [1, 32, 256, 256]         576
│    │    └─Conv1: 3-24                       [1, 64, 256, 256]         2,112
│    └─RGBD: 2-4                              [1, 128, 256, 256]        --
│    │    └─DenseBlock: 3-25                  [1, 192, 256, 256]        110,720
│    │    └─Conv1: 3-26                       [1, 128, 256, 256]        24,704
│    │    └─Sobelxy: 3-27                     [1, 64, 256, 256]         1,152
│    │    └─Conv1: 3-28                       [1, 128, 256, 256]        8,320
│    └─Conv2d: 2-5                            [1, 256, 128, 128]        --
│    │    └─Conv2d: 3-29                      [1, 256, 128, 128]        524,544
│    │    └─BatchNorm2d: 3-30                 [1, 256, 128, 128]        512
│    │    └─ReLU: 3-31                        [1, 256, 128, 128]        --
│    └─Conv2d: 2-6                            [1, 512, 64, 64]          --
│    │    └─Conv2d: 3-32                      [1, 512, 64, 64]          2,097,664
│    │    └─BatchNorm2d: 3-33                 [1, 512, 64, 64]          1,024
│    │    └─ReLU: 3-34                        [1, 512, 64, 64]          --
├─Feature_exa: 1-2                            [1, 32, 256, 256]         (recursive)
│    └─Conv2d: 2-7                            [1, 32, 128, 128]         (recursive)
│    │    └─Conv2d: 3-35                      [1, 32, 128, 128]         (recursive)
│    │    └─BatchNorm2d: 3-36                 [1, 32, 128, 128]         (recursive)
│    │    └─ReLU: 3-37                        [1, 32, 128, 128]         --
│    └─ASAattention: 2-8                      [1, 32, 256, 256]         (recursive)
│    │    └─ModuleList: 3-38                  --                        (recursive)
│    │    └─AdaptiveAvgPool2d: 3-39           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-40                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-41                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-42                     [1, 16, 128, 128]         --
│    │    └─AdaptiveAvgPool2d: 3-43           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-44                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-45                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-46                     [1, 16, 128, 128]         --
│    │    └─AdaptiveAvgPool2d: 3-47           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-48                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-49                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-50                     [1, 16, 128, 128]         --
│    │    └─AdaptiveAvgPool2d: 3-51           [1, 16, 1, 1]             --
│    │    └─Sigmoid: 3-52                     [1, 16, 1, 1]             --
│    │    └─GroupNorm: 3-53                   [1, 16, 128, 128]         (recursive)
│    │    └─Sigmoid: 3-54                     [1, 16, 128, 128]         --
│    └─RGBD: 2-9                              [1, 64, 256, 256]         (recursive)
│    │    └─DenseBlock: 3-55                  [1, 96, 256, 256]         (recursive)
│    │    └─Conv1: 3-56                       [1, 64, 256, 256]         (recursive)
│    │    └─Sobelxy: 3-57                     [1, 32, 256, 256]         (recursive)
│    │    └─Conv1: 3-58                       [1, 64, 256, 256]         (recursive)
│    └─RGBD: 2-10                             [1, 128, 256, 256]        (recursive)
│    │    └─DenseBlock: 3-59                  [1, 192, 256, 256]        (recursive)
│    │    └─Conv1: 3-60                       [1, 128, 256, 256]        (recursive)
│    │    └─Sobelxy: 3-61                     [1, 64, 256, 256]         (recursive)
│    │    └─Conv1: 3-62                       [1, 128, 256, 256]        (recursive)
│    └─Conv2d: 2-11                           [1, 256, 128, 128]        (recursive)
│    │    └─Conv2d: 3-63                      [1, 256, 128, 128]        (recursive)
│    │    └─BatchNorm2d: 3-64                 [1, 256, 128, 128]        (recursive)
│    │    └─ReLU: 3-65                        [1, 256, 128, 128]        --
│    └─Conv2d: 2-12                           [1, 512, 64, 64]          (recursive)
│    │    └─Conv2d: 3-66                      [1, 512, 64, 64]          (recursive)
│    │    └─BatchNorm2d: 3-67                 [1, 512, 64, 64]          (recursive)
│    │    └─ReLU: 3-68                        [1, 512, 64, 64]          --
├─Deconv2d: 1-3                               [1, 512, 256, 256]        --
│    └─ConvTranspose2d: 2-13                  [1, 512, 256, 256]        295,424
│    └─BatchNorm2d: 2-14                      [1, 512, 256, 256]        1,024
│    └─ReLU: 2-15                             [1, 512, 256, 256]        --
├─Deconv2d: 1-4                               [1, 256, 256, 256]        --
│    └─ConvTranspose2d: 2-16                  [1, 256, 256, 256]        295,168
│    └─BatchNorm2d: 2-17                      [1, 256, 256, 256]        512
│    └─ReLU: 2-18                             [1, 256, 256, 256]        --
├─Deconv2d: 1-5                               [1, 128, 256, 256]        --
│    └─ConvTranspose2d: 2-19                  [1, 128, 256, 256]        295,040
│    └─BatchNorm2d: 2-20                      [1, 128, 256, 256]        256
│    └─ReLU: 2-21                             [1, 128, 256, 256]        --
├─Deconv2d: 1-6                               [1, 64, 128, 128]         --
│    └─ConvTranspose2d: 2-22                  [1, 64, 128, 128]         294,976
│    └─BatchNorm2d: 2-23                      [1, 64, 128, 128]         128
│    └─ReLU: 2-24                             [1, 64, 128, 128]         --
├─Upsample: 1-7                               [1, 64, 256, 256]         --
├─Deconv2d: 1-8                               [1, 32, 64, 64]           --
│    └─ConvTranspose2d: 2-25                  [1, 32, 64, 64]           294,944
│    └─BatchNorm2d: 2-26                      [1, 32, 64, 64]           64
│    └─ReLU: 2-27                             [1, 32, 64, 64]           --
├─Upsample: 1-9                               [1, 32, 256, 256]         --
├─ASAattention: 1-10                          [1, 128, 256, 256]        256
│    └─ModuleList: 2-28                       --                        --
│    │    └─Sequential: 3-69                  [1, 128, 256, 256]        49,536
│    │    └─Sequential: 3-70                  [1, 128, 256, 256]        442,752
│    │    └─Sequential: 3-71                  [1, 128, 256, 256]        1,229,184
│    │    └─Sequential: 3-72                  [1, 128, 256, 256]        2,408,832
│    └─AdaptiveAvgPool2d: 2-29                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-30                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-31                        [1, 64, 256, 256]         128
│    └─Sigmoid: 2-32                          [1, 64, 256, 256]         --
│    └─AdaptiveAvgPool2d: 2-33                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-34                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-35                        [1, 64, 256, 256]         (recursive)
│    └─Sigmoid: 2-36                          [1, 64, 256, 256]         --
│    └─AdaptiveAvgPool2d: 2-37                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-38                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-39                        [1, 64, 256, 256]         (recursive)
│    └─Sigmoid: 2-40                          [1, 64, 256, 256]         --
│    └─AdaptiveAvgPool2d: 2-41                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-42                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-43                        [1, 64, 256, 256]         (recursive)
│    └─Sigmoid: 2-44                          [1, 64, 256, 256]         --
├─ASAattention: 1-11                          [1, 128, 256, 256]        256
│    └─ModuleList: 2-45                       --                        --
│    │    └─Sequential: 3-73                  [1, 128, 256, 256]        12,672
│    │    └─Sequential: 3-74                  [1, 128, 256, 256]        110,976
│    │    └─Sequential: 3-75                  [1, 128, 256, 256]        307,584
│    │    └─Sequential: 3-76                  [1, 128, 256, 256]        602,496
│    └─AdaptiveAvgPool2d: 2-46                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-47                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-48                        [1, 64, 256, 256]         128
│    └─Sigmoid: 2-49                          [1, 64, 256, 256]         --
│    └─AdaptiveAvgPool2d: 2-50                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-51                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-52                        [1, 64, 256, 256]         (recursive)
│    └─Sigmoid: 2-53                          [1, 64, 256, 256]         --
│    └─AdaptiveAvgPool2d: 2-54                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-55                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-56                        [1, 64, 256, 256]         (recursive)
│    └─Sigmoid: 2-57                          [1, 64, 256, 256]         --
│    └─AdaptiveAvgPool2d: 2-58                [1, 64, 1, 1]             --
│    └─Sigmoid: 2-59                          [1, 64, 1, 1]             --
│    └─GroupNorm: 2-60                        [1, 64, 256, 256]         (recursive)
│    └─Sigmoid: 2-61                          [1, 64, 256, 256]         --
===============================================================================================
'''   