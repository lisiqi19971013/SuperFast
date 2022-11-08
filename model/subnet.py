import torch
import torch.nn.functional as F
import datetime
import math
from torch import nn
import numpy as np


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)


class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """
    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        # print(self._pixels_pad_to_height, self._pixels_pad_to_width)
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


class TimeRecorder(object):
    def __init__(self, total_epoch, iter_per_epoch):
        self.total_epoch = total_epoch
        self.iter_per_epoch = iter_per_epoch
        self.start_train_time = datetime.datetime.now()
        self.start_epoch_time = datetime.datetime.now()
        self.t_last = datetime.datetime.now()

    def get_iter_time(self, epoch, iter):
        dt = (datetime.datetime.now() - self.t_last).__str__()
        self.t_last = datetime.datetime.now()
        remain_time = self.cal_remain_time(epoch, iter, self.total_epoch, self.iter_per_epoch)
        end_time = (datetime.datetime.now() + datetime.timedelta(seconds=remain_time)).strftime("%Y-%m-%d %H:%S:%M")
        remain_time = datetime.timedelta(seconds=remain_time).__str__()
        return dt, remain_time, end_time

    def cal_remain_time(self, epoch, iter, total_epoch, iter_per_epoch):
        t_used = (datetime.datetime.now() - self.start_train_time).total_seconds()
        time_per_iter = t_used / (epoch * iter_per_epoch + iter + 1)
        remain_iter = total_epoch * iter_per_epoch - (epoch * iter_per_epoch + iter + 1)
        remain_time_second = time_per_iter * remain_iter
        return remain_time_second


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def skip_mul(x1, x2):
    return x1 * (x2+1)


class up(nn.Module):
    def __init__(self, inChannels, outChannels, norm=False):
        super(up, self).__init__()
        bias = False if norm else True
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(outChannels)
            self.bn2 = nn.BatchNorm2d(outChannels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(outChannels, track_running_stats=True)
            self.bn2 = nn.InstanceNorm2d(outChannels, track_running_stats=True)

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x =self.conv1(x)
        if self.norm:
            x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)

        x = self.conv2(torch.cat((x, skpCn), 1))
        if self.norm:
            x = F.leaky_relu(self.bn2(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize, norm=False):
        super(down, self).__init__()
        bias = False if norm=='BN' else True
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2), bias=bias)
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2), bias=bias)
        self.norm = norm
        if norm == 'BN':
            print('Norm: BN')
            self.bn1 = nn.BatchNorm2d(outChannels)
            self.bn2 = nn.BatchNorm2d(outChannels)
        elif norm == 'IN':
            print('Norm: IN')
            self.bn1 = nn.InstanceNorm2d(outChannels, track_running_stats=True)
            self.bn2 = nn.InstanceNorm2d(outChannels, track_running_stats=True)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.conv1(x)
        if self.norm:
            x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)

        x = self.conv2(x)
        if self.norm:
            x = F.leaky_relu(self.bn2(x), negative_slope=0.1)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayer, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = torch.nn.LeakyReLU(negative_slope=0.1)   # torch.activation, 默认relu
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Attention_block(nn.Module):
    def __init__(self, input_channel=64):
        super(Attention_block, self).__init__()
        self.fc1 = nn.Linear(input_channel, int(input_channel/8))
        self.fc2 = nn.Linear(int(input_channel/8), input_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, infeature):
        b, c, h, w = infeature.shape
        max_f = F.max_pool2d(infeature, kernel_size=[h, w]).reshape(b, 1, c)
        avg_f = F.avg_pool2d(infeature, kernel_size=[h, w]).reshape(b, 1, c)

        cha_f = torch.cat([max_f, avg_f], dim=1)
        out1 = self.fc2(self.relu(self.fc1(cha_f)))
        channel_attention = self.sigmoid(out1[:, 0, :] + out1[:, 1, :]).reshape(b, c, 1, 1)
        feature_with_channel_attention = infeature * channel_attention
        return feature_with_channel_attention


class Metric(object):
    def __init__(self):
        self.L1Loss_this_epoch = []
        self.Lpips_this_epoch = []
        self.FeaLoss_this_epoch = []
        self.total_this_epoch = []

        self.L1Loss_history = []
        self.Lpips_history = []
        self.FeaLoss_history = []
        self.total_history = []

    def update(self, L1Loss, Lpips, FeaLoss, total):
        self.L1Loss_this_epoch.append(L1Loss.item())
        self.Lpips_this_epoch.append(Lpips.item())
        self.FeaLoss_this_epoch.append(FeaLoss.item())
        self.total_this_epoch.append(total.item())

    def update_epoch(self):
        avg = self.get_average_epoch()
        self.L1Loss_history.append(avg[0])
        self.Lpips_history.append(avg[1])
        self.FeaLoss_history.append(avg[2])
        self.total_history.append(avg[3])
        self.new_epoch()

    def new_epoch(self):
        self.L1Loss_this_epoch = []
        self.Lpips_this_epoch = []
        self.FeaLoss_this_epoch = []
        self.total_this_epoch = []

    def get_average_epoch(self):
        return np.average(self.L1Loss_this_epoch), np.average(self.Lpips_this_epoch), np.average(self.FeaLoss_this_epoch), np.average(self.total_this_epoch)



