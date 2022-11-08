import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    def __init__(self, norm=False):
        super(FeatureFusion, self).__init__()
        self.norm = norm
        self.fusion = nn.ModuleList([self.build_layer(32), self.build_layer(64), self.build_layer(128),
                                     self.build_layer(256), self.build_layer(512), self.build_layer(512)])

    def build_layer(self, channel):
        if not self.norm:
            return nn.Sequential(
                nn.Conv2d(channel*2, int(channel/2), kernel_size=1, stride=1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/2), int(channel/4), kernel_size=7, padding=3), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), int(channel/4), kernel_size=7, padding=3), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), channel, kernel_size=1)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(channel*2, int(channel/2), kernel_size=1, stride=1), nn.BatchNorm2d(int(channel/2)), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/2), int(channel/4), kernel_size=7, padding=3), nn.BatchNorm2d(int(channel/4)), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), int(channel/4), kernel_size=7, padding=3), nn.BatchNorm2d(int(channel/4)), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(int(channel/4), channel, kernel_size=1), nn.BatchNorm2d(channel)
            )

    def forward(self, image_fea, event_fea):
        output = []
        for k in range(len(image_fea)):
            x = torch.cat([image_fea[k], event_fea[k]], dim=1)
            res = self.fusion[k](x)
            output.append(image_fea[k] + res)
            # output.append(F.leaky_relu( (image_fea[k] + res) , negative_slope=0.1))
        return output

