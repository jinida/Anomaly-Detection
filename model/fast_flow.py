from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import timm
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock

def conv_func(kernel_size, hidden_ratio=1.0):
    def conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True),
        )
    return conv

class FastFlow(nn.Module):
    def __init__(self, flow_step, input_size=256, hidden_ratio=1.0):
        super(FastFlow, self).__init__()
        self.feature_extractor = timm.create_model(
            'resnet18',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)
        )
        
        channels = self.feature_extractor.feature_info.channels()
        scales = self.feature_extractor.feature_info.reduction()
        self.norms = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [channel, int(input_size / scale), int(input_size / scale)]
                )
            )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        
        self.flows = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            nodes = SequenceINN(channel, int(input_size / scale), int(input_size / scale))
            
            for _ in range(flow_step):
                nodes.append(
                    AllInOneBlock,
                    subnet_constructor=conv_func(3, hidden_ratio),
                    affine_clamping=2.0,
                    permute_soft=False,
                )
            self.flows.append(nodes)
            
        self.input_size = input_size
        
    def forward(self, x):
        self.feature_extractor.eval()
        
        features = self.feature_extractor(x)
        features = [norm(feature) for norm, feature in zip(self.norms, features)]
        
        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}
        
        if not self.training:
            anomaly_map_list = []
            
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
                
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            pred_socre = torch.amax(anomaly_map, dim=(2, 3))
            ret["anomaly_map"] = anomaly_map
            ret["pred_score"] = pred_socre
            
        return ret
        