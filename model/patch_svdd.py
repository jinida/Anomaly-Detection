import torch
import torch.nn as nn

from model.module import Conv

class PatchSVDD(nn.Module):
    def __init__(self, dimension: int = 64):
        super(PatchSVDD, self).__init__()
        self.dimension = dimension
        
        self.layer1 = nn.Sequential(
            Conv(3, 32, 3, 2, 0, bias=False),
            Conv(32, 64, 3, 1, 0, bias=False),
            Conv(64, 128, 3, 1, 0, bias=False),
            Conv(128, 128, 3, 1, 0, bias=False),
            Conv(128, 64, 3, 1, 0, bias=False),
            Conv(64, 32, 3, 1, 0, bias=False),
            Conv(32, 32, 3, 1, 0, bias=False),
            Conv(32, dimension, 3, 1, 0, bias=True, act=nn.Tanh, bn=False),
        )
        
        self.layer2 = nn.Sequential(
            Conv(dimension, 128, 2, 1, 0, bias=False),
            Conv(128, dimension, 1, 1, 0, bias=True, act=nn.Tanh, bn=False),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(dimension, dimension * 2, bias=False),
            nn.BatchNorm1d(dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension * 2, bias=False),
            nn.BatchNorm1d(dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, 8, bias=True),
        )
        
        self.position_loss = nn.CrossEntropyLoss()
        
    def _forward_hier(self, x, K):
        K_2 = K // 2
        n = x.size(0)
        x1 = x[..., :K_2, :K_2]
        x2 = x[..., :K_2, K_2:]
        x3 = x[..., K_2:, :K_2]
        x4 = x[..., K_2:, K_2:]
        xx = torch.cat([x1, x2, x3, x4], dim=0)
        hh = self.layer1(xx)
        h1, h2, h3, h4 = hh[:n], hh[n:2*n], hh[2*n:3*n], hh[3*n:]
        h12 = torch.cat([h1, h2], dim=3)
        h34 = torch.cat([h3, h4], dim=3)
        h = torch.cat([h12, h34], dim=2)
        return h
    
    def forward(self, x):
        if isinstance(x, dict):
            pos_p1_32, pos_p2_32, pos_32 = x['pos_32']
            pos_p1_64, pos_p2_64, pos_64 = x['pos_64']
            svdd_p1_64, svdd_p2_64 = x['svdd_64']
            svdd_p1_32, svdd_p2_32 = x['svdd_32']

            pos1_64_features = self.layer2(self._forward_hier(pos_p1_64, 64))
            pos2_64_features = self.layer2(self._forward_hier(pos_p2_64, 64))
            pos_64_features = self.fc((pos1_64_features - pos2_64_features).view(-1, self.dimension))
            pos_loss_64 = self.position_loss(pos_64_features, pos_64)
            
            pos1_32_features = self.layer1(pos_p1_32)
            pos2_32_features = self.layer1(pos_p2_32)
            pos_32_features = self.fc((pos1_32_features - pos2_32_features).view(-1, self.dimension))
            pos_loss_32 = self.position_loss(pos_32_features, pos_32)
            
            svdd_1_64_feature = self.layer2(self._forward_hier(svdd_p1_64, 64))
            svdd_2_64_feature = self.layer2(self._forward_hier(svdd_p2_64, 64))
            svdd_1_32_feature = self.layer1(svdd_p1_32)
            svdd_2_32_feature = self.layer1(svdd_p2_32)
            
            svdd_loss_64 = (svdd_1_64_feature - svdd_2_64_feature).norm(dim=1).mean()
            svdd_loss_32 = (svdd_1_32_feature - svdd_2_32_feature).norm(dim=1).mean()
            
            return {
                'pos_64': pos_loss_64,
                'pos_32': pos_loss_32,
                'svdd_64': svdd_loss_64,
                'svdd_32': svdd_loss_32
            }
            
        else:
            features = self.layer2(self._forward_hier(x, 64))
            return features
    
    