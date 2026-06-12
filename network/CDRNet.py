#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import kl

from .backbones.pvtv2 import *
from .cd_tools import ConvBNReLU
from .SCC import StructuralCueCompensation
from .FPD import PseudoChangeDisentanglement
from .Edge import Edge_Prediction_Head


class Decoder(nn.Module):
    def __init__(self,
                 latent_dim, num_classes
                 ):
        super(Decoder, self).__init__()
        channel = 128

        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.FPD4 = PseudoChangeDisentanglement(channel)
        self.FPD3 = PseudoChangeDisentanglement(channel)
        self.FPD2 = PseudoChangeDisentanglement(channel)
        self.FPD1 = PseudoChangeDisentanglement(channel, is_last=True, num_classes=num_classes)

    def forward(self, edge_guidance, layer4, layer3, layer2, layer1, auem_4, auem_3, auem_2, auem_1):
        out4 = self.FPD4(layer4, self.down8(edge_guidance), auem_4)
        out3 = self.FPD3(layer3, self.down4(edge_guidance), auem_3, out4)
        out2 = self.FPD2(layer2, self.down2(edge_guidance), auem_2, out3)
        refined_out = self.FPD1(layer1, edge_guidance, auem_1, out2)
        return refined_out


'''模型整体框架'''
class CDRNet(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(CDRNet, self).__init__()
        channel = 128

        self.backbone = pvt_v2_b2()
        path = './pretrained_model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv1 = ConvBNReLU(2 * 64, 64, 1, 1, 0)
        self.conv2 = ConvBNReLU(2 * 128, 128, 1, 1, 0)
        self.conv3 = ConvBNReLU(2 * 320, 320, 1, 1, 0)
        self.conv4 = ConvBNReLU(2 * 512, 512, 1, 1, 0)

        self.conv_4 = ConvBNReLU(512, channel, 3, 1, 1)
        self.conv_3 = ConvBNReLU(320, channel, 3, 1, 1)
        self.conv_2 = ConvBNReLU(128, channel, 3, 1, 1)
        self.conv_1 = ConvBNReLU(64, channel, 3, 1, 1)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.scc_1 = StructuralCueCompensation(2 * 64, channel)
        self.scc_2 = StructuralCueCompensation(2 * 128, channel)
        self.scc_3 = StructuralCueCompensation(2 * 320, channel)
        self.scc_4 = StructuralCueCompensation(2 * 512, channel)

        self.edge_head = Edge_Prediction_Head(channel)

        self.decoder = Decoder(latent_dim, num_classes)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)

        return kl_div

    def Feature_Extraction(self, A, B):
        layer_1_A, layer_2_A, layer_3_A, layer_4_A = self.backbone(A)
        layer_1_B, layer_2_B, layer_3_B, layer_4_B = self.backbone(B)

        layer_1 = self.conv_1(self.conv1(torch.cat((layer_1_A, layer_1_B), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((layer_2_A, layer_2_B), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((layer_3_A, layer_3_B), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((layer_4_A, layer_4_B), dim=1)))

        return (
            layer_1,    layer_2,    layer_3,    layer_4,
            layer_1_A,  layer_2_A,  layer_3_A,  layer_4_A,
            layer_1_B,  layer_2_B,  layer_3_B,  layer_4_B,
        )

    def forward(self, A, B, y=None):
        (
            layer1, layer2, layer3, layer4,
            layer_1_A, layer_2_A, layer_3_A, layer_4_A,
            layer_1_B, layer_2_B, layer_3_B, layer_4_B,
        ) = self.Feature_Extraction(A, B)

        auem_1 = self.scc_1(torch.cat((layer_1_A, layer_1_B), 1))
        auem_2 = self.scc_2(torch.cat((layer_2_A, layer_2_B), 1))
        auem_3 = self.scc_3(torch.cat((layer_3_A, layer_3_B), 1))
        auem_4 = self.scc_4(torch.cat((layer_4_A, layer_4_B), 1))

        edge_guidance = self.edge_head(auem_4, auem_3, auem_2, auem_1)
        edge_out = F.interpolate(edge_guidance, size=A.shape[-2:], mode='bilinear', align_corners=True)
        edge_guidance = torch.sigmoid(edge_guidance)

        Refined_out_prior = self.decoder(
            edge_guidance,
            layer4, layer3, layer2, layer1,
            auem_4, auem_3, auem_2, auem_1,
        )

        return Refined_out_prior, edge_out


if __name__ == '__main__':
    A = torch.rand(4, 3, 256, 256).cuda()
    B = torch.rand(4, 3, 256, 256).cuda()

    model = CDRNet(latent_dim=8, num_classes=1).cuda()

    outs = model(A, B)

    for o in outs:
        print(o.shape)

