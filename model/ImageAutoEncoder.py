from torch import nn
from model import down
import torch.nn.functional as F
import torch


class up1(nn.Module):
    def __init__(self, inChannels, outChannels, norm=False):
        super(up1, self).__init__()
        bias = False if norm else True
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(3 * outChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(outChannels)
            self.bn2 = nn.BatchNorm2d(outChannels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(outChannels, track_running_stats=True)
            self.bn2 = nn.InstanceNorm2d(outChannels, track_running_stats=True)

    def forward(self, x, skpCn1, skpCn2):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv1(x)
        if self.norm:
            x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)

        x = self.conv2(torch.cat((x, skpCn1, skpCn2), 1))
        if self.norm:
            x = F.leaky_relu(self.bn2(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, SizeAdapter, Channels=1, norm=False):
        super().__init__()
        self._size_adapter = SizeAdapter
        self.conv1 = nn.Conv2d(Channels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5, norm=norm)
        self.down2 = down(64, 128, 3, norm=norm)
        self.down3 = down(128, 256, 3, norm=norm)
        self.down4 = down(256, 512, 3, norm=norm)
        self.down5 = down(512, 512, 3, norm=norm)
        self.norm = norm
        if norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)

    def forward(self, image):
        x = self._size_adapter.pad(image)  # 1280, 736
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1) if not self.norm else F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1) if not self.norm else F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        s2 = self.down1(s1)  # 640, 368
        s3 = self.down2(s2)  # 320, 184
        s4 = self.down3(s3)  # 160, 92
        s5 = self.down4(s4)  # 80, 46
        x = self.down5(s5)  # 40, 23
        return [s1, s2, s3, s4, s5, x]


class ImageDecoder(nn.Module):
    def __init__(self, SizeAdapter, Channels=1, norm=False):
        super(ImageDecoder, self).__init__()
        self.up1 = up1(512, 512, norm=norm)
        self.up2 = up1(512, 256, norm=norm)
        self.up3 = up1(256, 128, norm=norm)
        self.up4 = up1(128, 64, norm=norm)
        self.up5 = up1(64, 32, norm=norm)
        self.conv3 = nn.Conv2d(32, Channels, 1)
        self._size_adapter = SizeAdapter

    def forward(self, input, img_fea):
        [s1, s2, s3, s4, s5, x] = input
        [i1, i2, i3, i4, i5, _] = img_fea
        x = self.up1(x, s5, i5)    #
        x = self.up2(x, s4, i4)    #
        x = self.up3(x, s3, i3)    #
        x = self.up4(x, s2, i2)    #
        x = self.up5(x, s1, i1)    #
        x = self.conv3(x)      #
        x = self._size_adapter.unpad(x)
        return x


class ImageAE(nn.Module):
    def __init__(self, Channels, norm=False):
        super(ImageAE, self).__init__()
        from model import SizeAdapter
        self._size_adapter = SizeAdapter(32)
        self.Encoder = ImageEncoder(self._size_adapter, Channels, norm=norm)
        self.Decoder = ImageDecoder(self._size_adapter, Channels, norm=norm)

    def forward(self, image):
        img_fea = self.Encoder(image)
        fusion_fea = [torch.zeros(img_fea[k].shape).type(img_fea[k].type()) for k in range(len(img_fea))]
        output = self.Decoder(fusion_fea, img_fea)
        return output