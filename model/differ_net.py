import torch
import torch.nn as nn
import timm

from FrEIA.framework import InputNode, OutputNode, ReversibleGraphNet, Node
from FrEIA.modules import PermuteRandom, GLOWCouplingBlock


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, internal_size_factor: int = 3):
        super(Projection, self).__init__()
        internal_size = in_channels * internal_size_factor
        
        self.net = nn.Sequential(
            nn.Linear(in_channels, internal_size),
            nn.SiLU(),
            nn.Linear(internal_size, internal_size),
            nn.SiLU(),
            nn.Linear(internal_size, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class DifferNet(nn.Module):
    def __init__(self, in_dim, num_coupling_blocks: int = 4, num_scales: int = 3):
        super(DifferNet, self).__init__()
        self.feature_extractor = timm.create_model('efficientnet_b0',
                                                   pretrained=True,
                                                   features_only=True,
                                                   out_indices=(4,))
        self.feature_extractor.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        nodes = list()
        nodes.append(InputNode(in_dim, name='input'))
        for i in range(num_coupling_blocks):
            nodes.append(Node(
                nodes[-1], PermuteRandom, {'seed': i}, name=f'permute_{i}')
            )
            nodes.append(Node(
                nodes[-1], GLOWCouplingBlock, {
                    'subnet_constructor': Projection,
                    'clamp': 2.0
                }, name=f'coupling_{i}')
            )
        nodes.append(OutputNode(nodes[-1], name='output'))
        self.flow = ReversibleGraphNet(nodes, verbose=False)
        self.num_scales = num_scales
        
    def forward(self, x):
        h, w = x.size()[2:]
        y = []
        for i in range(self.num_scales):
            x_scaled = nn.functional.interpolate(
                x, size=(h // (2 ** i), w // (2 ** i)), mode='bilinear'
            ) if i > 0 else x
            
            feature = self.feature_extractor(x_scaled)[-1]
            y.append(self.pool(feature).view(feature.size(0), -1))
            
        y = torch.cat(y, dim=1)
        z, log_jac_det = self.flow(y)
        
        return z, log_jac_det
        