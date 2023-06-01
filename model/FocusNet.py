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

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True, activation='leakyrelu', dropout=False):
        super(Conv2d, self).__init__()
        padding = (kernel_size - 1) // 2  
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding) 
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

class Feature_exa(nn.Module):
    def __init__(self):
        super(Feature_exa, self).__init__()
        #CB conv block =conv + BN + leakyrelu
        self.FEB_1 = Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)     
        self.GRDB_2 = GRDB.RGBD(in_channels = 32, out_channels = 64)
        self.FEB_2 = Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.GRDB_3 = GRDB.RGBD(in_channels = 64, out_channels = 128)
        self.FEB_3 = Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.FEB_4 = Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        self.FEB_5 = Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride=2, bn=True, activation='leakyrelu', dropout=False)
        
    def forward(self,  img_input):
        FEB_1 = self.FEB_1(img_input)
        FEB_2_GRDB = self.GRDB_2(FEB_1)
        FEB_2 = self.FEB_2(FEB_1)
        FEB_3 = self.FEB_3(FEB_2)
        FEB_4_GRDB = self.FEB_4(FEB_3_GRDB)
        FEB_4 = self.FEB_4(FEB_3)
        FEB_5 = self.FEB_5(FEB_4)
        return FEB_1, FEB_2_GRDB, FEB_3, FEB_4_GRDB, FEB_5

    def Predicting_engagement(self,
                            IR_FEB_1, IR_FEB_2_GRDB, IR_FEB_3_GRDB, IR_FEB_4, IR_FEB_5,
                            VIS_FEB_1, VIS_FEB_2_GRDB, VIS_FEB_3_GRDB, VIS_FEB_4, VIS_FEB_5,
                            img_fusion):
        PE_cat_IR_1 = torch.cat((IR_FEB_1,IR_FEB_2_GRDB), 1)
        PE_cat_IR_2 = torch.cat((IR_FEB_2_GRDB,IR_FEB_2_GRDB), 1)
        PE_cat_IR_3 = torch.cat((IR_FEB_3_GRDB,IR_FEB_4), 1)
        PE_cat_IR_4 = torch.cat((IR_FEB_4,IR_FEB_5), 1)
                
        PE_cat_VIS_1 = torch.cat((VIS_FEB_1,VIS_FEB_2_GRDB), 1)
        PE_cat_VIS_2 = torch.cat((VIS_FEB_2_GRDB,VIS_FEB_3_GRDB), 1)
        PE_cat_VIS_3 = torch.cat((VIS_FEB_3_GRDB,VIS_FEB_4), 1)
        PE_cat_VIS_4 = torch.cat((VIS_FEB_4,VIS_FEB_5), 1)
        PE_cat_IR_1_2 = torch.mean(PE_cat_IR_1, dim=1)
        PE_cat_IR_1_2 = PE_cat_IR_1_2.unsqueeze(1)
        PE_cat_VIS_1_2 = torch.mean(PE_cat_IR_1, dim=1)
        PE_cat_VIS_1_2 = PE_cat_IR_1_2.unsqueeze(1)
        PE_ssim_1 = (ssim(PE_cat_IR_1_2, img_fusion) + ssim(PE_cat_IR_1_2,img_fusion)) / 2
        PE_ssim_1 = torch.sigmoid(PE_ssim_1)
        PE_cat_IR_2_2 = torch.mean(PE_cat_IR_2, dim=1)
        PE_cat_IR_2_2 = PE_cat_IR_2_2.unsqueeze(1)
        PE_cat_VIS_2_2 = torch.mean(PE_cat_VIS_2, dim=1)
        PE_cat_VIS_2_2 = PE_cat_VIS_2_2.unsqueeze(1)
        PE_ssim_2 = (ssim(PE_cat_IR_2_2, img_fusion) + ssim(PE_cat_IR_2_2,img_fusion)) / 2
        PE_ssim_2 = torch.sigmoid(PE_ssim_2)  
        PE_cat_IR_3_2 = torch.mean(PE_cat_IR_3, dim=1)
        PE_cat_IR_3_2 = PE_cat_IR_3_2.unsqueeze(1)
        PE_cat_VIS_3_2 = torch.mean(PE_cat_VIS_3, dim=1)
        PE_cat_VIS_3_2 = PE_cat_VIS_3_2.unsqueeze(1)
        PE_ssim_3 = (ssim(PE_cat_IR_3_2, img_fusion) + ssim(PE_cat_IR_3_2,img_fusion)) / 2
        PE_ssim_3 = torch.sigmoid(PE_ssim_3) 
        PE_cat_IR_4_2 = torch.mean(PE_cat_IR_4, dim=1)
        PE_cat_IR_4_2 = PE_cat_IR_4_2.unsqueeze(1)
        PE_cat_VIS_4_2 = torch.mean(PE_cat_VIS_4, dim=1)
        PE_cat_VIS_4_2 = PE_cat_VIS_4_2.unsqueeze(1)
        PE_ssim_4 = (ssim(PE_cat_IR_4_2, img_fusion) + ssim(PE_cat_IR_4_2,img_fusion)) / 2
        PE_ssim_4 = torch.sigmoid(PE_ssim_4)    
        PE_sum = PE_ssim_1 + PE_ssim_2 + PE_ssim_3 + PE_ssim_4
        PE_mean = PE_sum / 4
        PE_leak_1 = torch.tensor(1440)
        PE_leak_2 = torch.tensor(1440)
        PE_leak_3 = torch.tensor(1440)
        PE_leak_4 = torch.tensor(1440)
        PE_leak_1 = torch.abs(PE_ssim_1 - PE_mean)
        PE_leak_2 = torch.abs(PE_ssim_2 - PE_mean)
        PE_leak_3 = torch.abs(PE_ssim_3 - PE_mean)
        PE_leak_4 = torch.abs(PE_ssim_4 - PE_mean)
        PE_leak_sum = PE_leak_1 + PE_leak_2 + PE_leak_3 + PE_leak_4
        #print("PE_leak_1 + PE_leak_2 + PE_leak_3 + PE_leak_4",PE_leak_1,PE_leak_2, PE_leak_3, PE_leak_4)
        ALL_channels = torch.tensor(1440)
        CHA_layer_3_1 = torch.floor(ALL_channels * ( PE_leak_1 / PE_leak_sum)).int()
        CHA_layer_3_2 = torch.floor(ALL_channels * ( PE_leak_2 / PE_leak_sum)).int()
        CHA_layer_3_3 = torch.floor(ALL_channels * ( PE_leak_3 / PE_leak_sum)).int()
        CHA_layer_3_4 = ALL_channels - CHA_layer_3_1 - CHA_layer_3_2 - CHA_layer_3_3
                
        return CHA_layer_3_1, CHA_layer_3_2, CHA_layer_3_3, CHA_layer_3_4
 
class FocusNet_Fusion(nn.Module):
    def __init__(self):
        super(FocusNet_Fusion, self).__init__()
        #self.feature_extraction  = Feature_exa()
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
        self.ASA_layer3_1 = ASAattention.ASAattention(channel = 738, channel_out = 128, hw = 400)
        self.ASA_layer3_2 = ASAattention.ASAattention(channel = 384, channel_out = 128, hw = 400)
        self.ASA_layer3_3 = ASAattention.ASAattention(channel = 192, channel_out = 128, hw = 400)
        self.ASA_layer3_4 = ASAattention.ASAattention(channel = 96, channel_out = 128, hw = 400)
        self.bn_layer4 = nn.BatchNorm2d(1440, eps=0.001, momentum=0, affine=True)
        self.conv_layer5 = nn.Conv2d(1440, 608, 1)
        self.bn_layer5 = nn.BatchNorm2d(608, eps=0.001, momentum=0, affine=True)
        self.conv_layer6 = nn.Conv2d(608, 304, 1)
        self.bn_layer6 = nn.BatchNorm2d(304, eps=0.001, momentum=0, affine=True)
        self.conv_layer7 = nn.Conv2d(304, 152, 1)
        self.bn_layer7 = nn.BatchNorm2d(152, eps=0.001, momentum=0, affine=True) 
        self.conv_layer8 = nn.Conv2d(152, 76, 1)
        self.bn_layer8 = nn.BatchNorm2d(76, eps=0.001, momentum=0, affine=True) 
        self.conv_layer9 = nn.Conv2d(76, 1, 1)
        self.bn_layer9 = nn.BatchNorm2d(1, eps=0.001, momentum=0, affine=True)
    def forward(self, 
                IR_layer_1,IR_layer_2,IR_layer_3,IR_layer_4,IR_layer_5,
                VIS_layer_1,VIS_layer_2,VIS_layer_3,VIS_layer_4,VIS_layer_5,
                CHA_layer_3_1, CHA_layer_3_2, CHA_layer_3_3, CHA_layer_3_4
                ):
        layer1_1 = torch.cat((IR_layer_1,VIS_layer_1), 1)
        layer1_2 = torch.cat((IR_layer_2,VIS_layer_2), 1)
        layer1_3 = torch.cat((IR_layer_3,VIS_layer_3), 1)
        layer1_4 = torch.cat((IR_layer_4,VIS_layer_4), 1)
        layer1_5 = torch.cat((IR_layer_5,VIS_layer_5), 1)
        layer2_1 = self.deconv_1(layer1_1) 
        layer2_2 = self.deconv_2(layer1_2) 
        layer2_3 = self.deconv_3(layer1_3) 
        layer2_4 = self.deconv_4(layer1_4) 
        layer2_5 = self.deconv_5(layer1_5) 
        layer3_1 = torch.cat((layer2_1,layer2_2), 1) 
        layer3_1 = self.bn_layer3_1(layer3_1)
        New_conv3_1 = nn.Conv2d(in_channels=768, out_channels= CHA_layer_3_1 , kernel_size=1, stride=1, padding=0).cuda()
        layer3_1 = New_conv3_1(layer3_1)
        layer3_2 = torch.cat((layer2_2,layer2_3), 1) 
        layer3_2 = self.bn_layer3_2(layer3_2)
        New_conv3_2 = nn.Conv2d(in_channels=384, out_channels= CHA_layer_3_2, kernel_size=1, stride=1, padding=0).cuda()
        layer3_2 = New_conv3_2(layer3_2)
        layer3_3 = torch.cat((layer2_3,layer2_4), 1) 
        layer3_3 = self.bn_layer3_3(layer3_3)
        New_conv3_3 = nn.Conv2d(in_channels=192, out_channels = CHA_layer_3_3, kernel_size=1, stride=1, padding=0).cuda()
        layer3_3 = New_conv3_3(layer3_3)
        layer3_4 = torch.cat((layer2_4,layer2_5), 1) 
        layer3_4 = self.bn_layer3_4(layer3_4)
        New_conv3_4 = nn.Conv2d(in_channels=96, out_channels = CHA_layer_3_4, kernel_size=1, stride=1, padding=0).cuda()
        layer3_4 = New_conv3_4(layer3_4)
        layer4 = torch.cat((layer3_1,
                            layer3_2,
                            layer3_3,
                            layer3_4), 1)
        layer4 = self.bn_layer4(layer4)
        layer5 = self.conv_layer5(layer4)
        layer5 = self.bn_layer5(layer5)
        layer6 = self.conv_layer6(layer5)
        layer6 = self.bn_layer6(layer6)
        layer7 = self.conv_layer7(layer6)
        layer7 = self.bn_layer7(layer7)
        layer8 = self.conv_layer8(layer7)
        layer8 = self.bn_layer8(layer8)
        layer9 = self.conv_layer9(layer8)
        return layer9