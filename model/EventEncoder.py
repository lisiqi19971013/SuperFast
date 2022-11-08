from torch import nn
import slayerSNN as snn
import torch
from model import down, SizeAdapter
import torch.nn.functional as F


def getNeuronConfig(type: str='SRMALPHA', theta: float=10., tauSr: float=1., tauRef: float=1., scaleRef: float=2., tauRho: float=0.3, scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }


class SnnEncoder(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4], scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100]):
        super(SnnEncoder, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0], scaleRho=scaleRho[0]))
        self.neuron_config.append(getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1], scaleRho=scaleRho[1]))
        self.neuron_config.append(getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2], scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams)
        self.slayer2 = snn.layer(self.neuron_config[1], netParams)
        self.slayer3 = snn.layer(self.neuron_config[2], netParams)

        self.conv1 = self.slayer1.conv(2, 16, kernelSize=3, padding=1)
        self.conv2 = self.slayer2.conv(18, 16, kernelSize=3, padding=1)
        self.conv3 = self.slayer3.conv(18, hidden_number, kernelSize=1, padding=0)

    def forward(self, spikeInput):
        # Bs x channel x H x W x ts
        psp0 = self.slayer1.psp(spikeInput)
        psp1 = self.conv1(psp0)
        spikes_1 = self.slayer1.spike(psp1)

        psp2 = torch.cat([self.slayer2.psp(spikes_1), psp0], dim=1)
        psp2 = self.conv2(psp2)
        spikes_2 = self.slayer2.spike(psp2)

        psp3 = torch.cat([self.slayer3.psp(spikes_2), psp0], dim=1)
        psp3 = self.conv3(psp3)
        return psp3


class EventEncoder(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], norm=False):
        super(EventEncoder, self).__init__()
        self.SNN = SnnEncoder(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        self.IN = nn.InstanceNorm2d(hidden_number)
        self._size_adapter = SizeAdapter(minimum_size=32)
        self.conv1 = nn.Conv2d(hidden_number, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        print(netParams)
        self.down1 = down(32, 64, 5, norm=norm)
        self.down2 = down(64, 128, 3, norm=norm)
        self.down3 = down(128, 256, 3, norm=norm)
        self.down4 = down(256, 512, 3, norm=norm)
        self.down5 = down(512, 512, 3, norm=norm)
        self.norm = norm
        if norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)

    def forward(self, events, n_left, n_right, forward=True):
        bs, _, H, W, Ts = events.shape
        snn_fea = self.SNN(events)
        snn_fea_left_list = []
        snn_fea_right_list = []
        if forward:
            for b in range(bs):
                assert snn_fea[b, ..., n_left[b]:].shape[-1] == n_right[b]
                snn_fea_left_list.append(torch.sum(snn_fea[b, ..., :n_left[b]], dim=-1).unsqueeze(0) / n_left[b].cuda())
                snn_fea_right_list.append(torch.sum(snn_fea[b, ..., n_left[b]:], dim=-1).unsqueeze(0) / n_right[b].cuda())
        else:
            for b in range(bs):
                assert snn_fea[b, ..., n_right[b]:].shape[-1] == n_left[b]
                snn_fea_left_list.append(torch.sum(snn_fea[b, ..., n_right[b]:], dim=-1).unsqueeze(0) / n_left[b].cuda())
                snn_fea_right_list.append(torch.sum(snn_fea[b, ..., :n_right[b]], dim=-1).unsqueeze(0) / n_right[b].cuda())

        snn_fea_left = torch.cat(snn_fea_left_list, dim=0)
        snn_fea_right = torch.cat(snn_fea_right_list, dim=0)  # b x C x H x W
        snn_fea = torch.cat([snn_fea_left, snn_fea_right], dim=0)  # 2b x C x H x W

        snn_fea = self.IN(snn_fea)
        x = self._size_adapter.pad(snn_fea)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1) if not self.norm else F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)  if not self.norm else F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        return [s1[:bs, ...], s2[:bs, ...], s3[:bs, ...], s4[:bs, ...], s5[:bs, ...], x[:bs, ...]], \
               [s1[bs:, ...], s2[bs:, ...], s3[bs:, ...], s4[bs:, ...], s5[bs:, ...], x[bs:, ...]]