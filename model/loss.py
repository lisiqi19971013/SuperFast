import sys
from torch import nn
import torch
import torch.nn.functional as F
from model import down


class IE(nn.Module):
    def __init__(self, Channels=1, norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(Channels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5, norm=norm)
        self.down2 = down(64, 128, 3, norm=norm)
        self.down3 = down(128, 256, 3, norm=norm)
        self.down4 = down(256, 512, 3, norm=norm)
        self.norm = norm
        if norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1) if not self.norm else F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1) if not self.norm else F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        s2 = self.down1(s1)  # 640, 368
        s3 = self.down2(s2)  # 320, 184
        s4 = self.down3(s3)  # 160, 92
        s5 = self.down4(s4)  # 80, 46
        return [s1, s2, s3, s4, s5]


class LPIPS_Ours(nn.Module):
    def __init__(self, Channels, pretrain_ckpt, cal_compare=True):
        super(LPIPS_Ours, self).__init__()
        self.net = IE(Channels=Channels)
        if cal_compare:
            print('Set up Lpips and Comparison loss:', pretrain_ckpt)
        else:
            print('Set up Lpips loss:', pretrain_ckpt)
        self.net.load_state_dict(torch.load(pretrain_ckpt)['encoder_state_dict'], strict=False)
        self.L2 = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()
        for param in self.net.parameters():
            param.requires_grad = False
        self.cal_compare = cal_compare

        self.net.eval()

    def forward(self, output, gt_pos, gt_neg=None, mask=None, weight=[1/32, 1/16, 1/8, 1/4, 1]):
        if len(output.shape) == 3:
            output = output.unsqueeze(0)
        if len(gt_pos.shape) == 3:
            gt_pos = gt_pos.unsqueeze(0)

        fea_out = self.net(output)
        if not self.cal_compare or gt_neg == None:
            with torch.no_grad():
                fea_pos = self.net(gt_pos)
            Lpips = self.L2(fea_out[0], fea_pos[0])
            return Lpips
        else:
            with torch.no_grad():
                fea_pos = self.net(gt_pos)
                fea_neg = self.net(gt_neg)
            Lpips = self.L2(fea_out[0], fea_pos[0])

            mask_list = []
            mask_list.append(torch.repeat_interleave(mask.unsqueeze(1), 32, dim=1))
            mask_list.append(torch.repeat_interleave(F.max_pool2d(mask_list[0].float(), 2, stride=2), 2, dim=1).bool())
            mask_list.append(torch.repeat_interleave(F.max_pool2d(mask_list[1].float(), 2, stride=2), 2, dim=1).bool())
            mask_list.append(torch.repeat_interleave(F.max_pool2d(mask_list[2].float(), 2, stride=2), 2, dim=1).bool())
            mask_list.append(torch.repeat_interleave(F.max_pool2d(mask_list[3].float(), 2, stride=2), 2, dim=1).bool())

            d_ap = self.L1(fea_out[0][mask_list[0]], fea_pos[0][mask_list[0]]) * weight[0]
            d_an = self.L1(fea_out[0][mask_list[0]], fea_neg[0][mask_list[0]]) * weight[0]
            for i in range(1, 5):
                d_ap += self.L1(fea_out[i][mask_list[i]], fea_pos[i][mask_list[i]]) * weight[i]
                d_an += self.L1(fea_out[i][mask_list[i]], fea_neg[i][mask_list[i]]) * weight[i]
            ComparisonLoss = d_ap/d_an

            return Lpips, ComparisonLoss
