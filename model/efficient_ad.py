import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pdn_network(out_channels=384):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

class EfficientAD(nn.Module):
    def __init__(self, out_channels=384):
        super(EfficientAD, self).__init__()
        self.teacher = get_pdn_network(out_channels=out_channels)
        self.student = get_pdn_network(out_channels=2*out_channels)
        self.teacher_mean: torch.Tensor
        self.teacher_std: torch.Tensor
        self.register_buffer('teacher_mean', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('teacher_std', torch.ones(1, out_channels, 1, 1))
        self.q_st_start: torch.Tensor
        self.q_st_end: torch.Tensor
        self.q_ae_start: torch.Tensor
        self.q_ae_end: torch.Tensor
        
        self.register_buffer('q_st_start', torch.tensor(0.0))
        self.register_buffer('q_st_end', torch.tensor(0.0))
        self.register_buffer('q_ae_start', torch.tensor(0.0))
        self.register_buffer('q_ae_end', torch.tensor(0.0))

        self.out_channels = out_channels
        
        self.ae = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
            nn.Upsample(size=3, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                    padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=8, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                    padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=15, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                    padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=32, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                    padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=63, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                    padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=127, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                    padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=56, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                    padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1)
        )
    
    def set_feature_params(self, loader):
        means_outputs = []
        means_distances = []
        for data in loader:
            image, _ = data
            inputs = image.cuda()
            with torch.no_grad():
                features = self.teacher(inputs)
                means_outputs.append(torch.mean(features, dim=[0, 2, 3]))
        self.teacher_mean = torch.mean(torch.stack(means_outputs), dim=0)[None, :, None, None]
        
        for data in loader:
            image, _ = data
            inputs = image.cuda()
            
            with torch.no_grad():
                features = self.teacher(inputs)
                distances = torch.mean((features - self.teacher_mean) ** 2, dim=[0, 2, 3])
                means_distances.append(distances)
        channel_var = torch.mean(torch.stack(means_distances), dim=0)[None, :, None, None]
        
        self.teacher_std = torch.sqrt(channel_var + 1e-6)
    
    def forward(self, x):
        if isinstance(x, dict):
            input = x['input'].cuda()
            ae_input = x['ae_input'].cuda()
            
            with torch.no_grad():
                teacher_out = (self.teacher(input) - self.teacher_mean) / self.teacher_std
            student_out = self.student(input)[:, :self.out_channels]
            distance_out = (teacher_out - student_out) ** 2
            d_hard = torch.quantile(distance_out.flatten(), 0.999)
            loss_hard = torch.mean(distance_out[distance_out >= d_hard])
            
            ae_teacher_out = (self.teacher(ae_input) - self.teacher_mean) / self.teacher_std
            ae_student_out = self.student(ae_input)[:, self.out_channels:]
            ae_out = self.ae(ae_input)
            
            distance_ae_out = (ae_teacher_out - ae_out) ** 2
            distance_stae_out = (ae_student_out - ae_out) ** 2
            loss_total = torch.mean(distance_ae_out) + torch.mean(distance_stae_out) + loss_hard
            
            return loss_total
        else:
            input = x.cuda()
            with torch.no_grad():
                teacher_out = (self.teacher(input) - self.teacher_mean) / self.teacher_std
            student_out = self.student(input)[:, :self.out_channels]
            ae_out = self.ae(input)
            map_st = torch.mean((teacher_out - student_out[:, :self.out_channels]) ** 2, dim=1, keepdim=True)
            map_ae = torch.mean((teacher_out - ae_out) ** 2, dim=1, keepdim=True)
            
            return map_st, map_ae
        