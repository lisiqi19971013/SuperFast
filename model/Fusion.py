import torch.nn as nn
import torch
from model import up, down, SizeAdapter, RefineModel, SynthesisSlow
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    def __init__(self, Channels=1):
        super().__init__()
        self._size_adapter = SizeAdapter(32)
        self.conv1 = nn.Conv2d(Channels*3+1, 32, 7, stride=1, padding=3)
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


class FusionModel(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], channel=1, fast_ckpt='', slow_ckpt=''):
        super(FusionModel, self).__init__()
        self.Refine = RefineModel(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho, channel, fast_ckpt)
        self.SynthesisSlow = SynthesisSlow(channel)
        self.Fusion = AttentionFusion(channel)
        self.slow_ckpt = slow_ckpt
        self.fast_ckpt = fast_ckpt

    def load_prev_state(self):
        model_stat = torch.load(self.fast_ckpt)
        print('loading fast synthesis from: ', self.fast_ckpt)
        self.Refine.load_state_dict(model_stat['state_dict'])

        model_stat = torch.load(self.slow_ckpt)
        print('loading slow synthesis from: ', self.slow_ckpt)
        self.SynthesisSlow.load_state_dict(model_stat['state_dict'])    # Ours
        # self.SynthesisSlow.fusion_network.load_state_dict(model_stat['state_dict'])   # HSERGB

    def forward(self, events_forward, events_backward, left_image, right_image, weight, n_left, n_right, surface, left_voxel_grid, right_voxel_grid):
        bs, _, H, W = left_image.shape
        output_left, output_right, _, _ = self.Refine(events_forward, events_backward, left_image, right_image, weight, n_left, n_right, surface)
        output_slow = self.SynthesisSlow(left_voxel_grid, left_image, right_voxel_grid, right_image)
        output = self.Fusion([output_slow, output_left, output_right, weight.view(-1, 1, 1, 1).expand(bs, 1, H, W).type(output_right.type())])
        return output

