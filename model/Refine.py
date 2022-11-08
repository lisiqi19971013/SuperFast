import torch.nn as nn
import torch
from model import up, down, SizeAdapter, SynthesisModule
import torch.nn.functional as F


class RefineModule(nn.Module):
    def __init__(self, Channels=1):
        super().__init__()
        self._size_adapter = SizeAdapter(32)
        self.conv1 = nn.Conv2d(Channels*2+2, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, Channels, 1)

    def forward(self, input):
        # print('refine', input[0].device, input[-1].device)
        x = torch.cat(input, dim=1)
        x = self._size_adapter.pad(x)  # 1280, 736
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)  # 1280, 736
        s2 = self.down1(s1)  # 640, 368
        s3 = self.down2(s2)  # 320, 184
        s4 = self.down3(s3)  # 160, 92
        s5 = self.down4(s4)  # 80, 46
        x = self.down5(s5)  # 40, 23
        x = self.up1(x, s5)    #
        x = self.up2(x, s4)    #
        x = self.up3(x, s3)    #
        x = self.up4(x, s2)    #
        x = self.up5(x, s1)    #
        x = self.conv3(x)      #
        x = self._size_adapter.unpad(x)
        return x


class RefineModel(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], channel=1, pretrain_ckpt=''):
        super(RefineModel, self).__init__()
        self.Sythesis = SynthesisModule(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho, channel, pretrain_ckpt)
        self.Refine = RefineModule(channel)
        self.pretrain_ckpt = pretrain_ckpt

    def load_prev_state(self):
        model_stat = torch.load(self.pretrain_ckpt)
        print('loading model from: ', self.pretrain_ckpt)
        self.Sythesis.load_state_dict(model_stat['state_dict'])

    def forward(self, events_forward, events_backward, left_image, right_image, weight, n_left, n_right, surface):
        output_left, output_right = self.Sythesis(events_forward, events_backward, left_image, right_image, weight, n_left, n_right)
        surface = torch.sum(surface, dim=-1)
        refine_left = self.Refine([left_image, output_left, surface])
        refine_right = self.Refine([right_image, output_right, surface])
        return refine_left, refine_right, output_left, output_right