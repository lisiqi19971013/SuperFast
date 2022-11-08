import torch.nn as nn
import torch
from model import SizeAdapter, UNet
from model import EventEncoder, ImageEncoder, ImageDecoder, FeatureFusion


class SynthesisModule(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], channel=1, pretrain_ckpt=''):
        super(SynthesisModule, self).__init__()
        self._size_adapter = SizeAdapter(32)
        self.EventEncoder = EventEncoder(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        self.ImageEncoder = ImageEncoder(self._size_adapter, channel)
        self.ImageDecoder = ImageDecoder(self._size_adapter, channel)
        self.FeatureFusion_forward = FeatureFusion()
        self.FeatureFusion_backward = FeatureFusion()
        self.pretrain_ckpt = pretrain_ckpt

    def load_pretrain_model(self):
        pretrain_img_model = torch.load(self.pretrain_ckpt)
        print('loading pretrain model from: ', self.pretrain_ckpt)
        self.ImageEncoder.load_state_dict(pretrain_img_model['encoder_state_dict'])
        self.ImageDecoder.load_state_dict(pretrain_img_model['decoder_state_dict'])

    def forward(self, events_forward, events_backward, left_image, right_image, weight, n_left, n_right):
        bs, _, H, W = left_image.shape
        [event_fea_forward_left, _] = self.EventEncoder(events_forward, n_left, n_right, True)
        [_, event_fea_backward_right] = self.EventEncoder(events_backward, n_left, n_right, False)

        img_fea_left = self.ImageEncoder(left_image)
        img_fea_right = self.ImageEncoder(right_image)
        fusion_fea_forward = self.FeatureFusion_forward(img_fea_left, event_fea_forward_left)
        fusion_fea_backward = self.FeatureFusion_backward(img_fea_right, event_fea_backward_right)
        output_left = self.ImageDecoder(fusion_fea_forward, img_fea_left)
        output_right = self.ImageDecoder(fusion_fea_backward, img_fea_right)

        return output_left, output_right


class SynthesisSlow(nn.Module):
    def __init__(self, channel=1, pretrain_ckpt=''):
        super(SynthesisSlow, self).__init__()
        self.fusion_network = UNet(2 * channel + 2 * 5, channel, False)
        self.pretrain_ckpt = pretrain_ckpt

    def load_pretrain_model(self):
        pretrain_img_model = torch.load(self.pretrain_ckpt)
        print('loading pretrain model from: ', self.pretrain_ckpt)
        self.fusion_network.load_state_dict(pretrain_img_model['state_dict'])

    def forward(self, left_voxel_grid, left_image, right_voxel_grid, right_image):
        input = torch.cat([left_voxel_grid, left_image, right_voxel_grid, right_image], dim=1)
        synthesis_slow = self.fusion_network(input)
        return synthesis_slow