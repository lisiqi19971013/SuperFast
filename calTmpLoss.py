import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import sys
sys.path.append('..')
import torch as th


def create_meshgrid(width, height, is_cuda):
    x, y = th.meshgrid([th.arange(0, width), th.arange(0, height)])
    x, y = (x.transpose(0, 1).float(), y.transpose(0, 1).float())
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
    return x, y


def compute_source_coordinates(y_displacement, x_displacement):
    """Retruns source coordinates, given displacements.

    Given traget coordinates (y, x), the source coordinates are
    computed as (y + y_displacement, x + x_displacement).

    Args:
        x_displacement, y_displacement: are tensors with indices [example_index, 1, y, x]
    """
    width, height = y_displacement.size(-1), y_displacement.size(-2)
    x_target, y_target = create_meshgrid(width, height, y_displacement.is_cuda)
    x_source = x_target + x_displacement.squeeze(1)
    y_source = y_target + y_displacement.squeeze(1)
    out_of_boundary_mask = ((x_source.detach() < 0) | (x_source.detach() >= width) |
                            (y_source.detach() < 0) | (y_source.detach() >= height))
    return y_source, x_source, out_of_boundary_mask


def backwarp_2d(source, y_displacement, x_displacement):
    """Returns warped source image and occlusion_mask.
    Value in location (x, y) in output image in taken from
    (x + x_displacement, y + y_displacement) location of the source image.
    If the location in the source image is outside of its borders,
    the location in the target image is filled with zeros and the
    location is added to the "occlusion_mask".

    Args:
        source: is a tensor with indices [example_index, channel_index, y, x].
        x_displacement, y_displacement: are tensors with indices [example_index, 1, y, x].
    Returns:
        target: is a tensor with indices [example_index, channel_index, y, x].
        occlusion_mask: is a tensor with indices [example_index, 1, y, x].
    """
    width, height = source.size(-1), source.size(-2)
    y_source, x_source, out_of_boundary_mask = compute_source_coordinates(y_displacement, x_displacement)
    x_source = (2.0 / float(width - 1)) * x_source - 1
    y_source = (2.0 / float(height - 1)) * y_source - 1
    x_source = x_source.masked_fill(out_of_boundary_mask, 0)
    y_source = y_source.masked_fill(out_of_boundary_mask, 0)
    grid_source = th.stack([x_source, y_source], -1)
    target = th.nn.functional.grid_sample(source, grid_source, align_corners=True)
    out_of_boundary_mask = out_of_boundary_mask.unsqueeze(1)
    target.masked_fill_(out_of_boundary_mask.expand_as(target), 0)
    return target, out_of_boundary_mask


outputPath = './ckpt/THU-HSEVI/'
datasetPath = './dataset/THU-HSEVI'

al = [-0.05, -0.35, -0.7, -0.06, -3, -25]
cl = ['cup_breaking', 'water_bomb', 'balloon_bursting', 'chessboard_waving', 'hard_disk_spining', 'firing']

for n in range(6):
    cate = cl[n]
    opFolder = os.path.join(outputPath, cate, cate+'1')
    alpha = al[n]

    lossList = []
    lossGtList = []

    gtFolder = os.path.join(datasetPath, cate, cate + '1', 'frame')

    for i in range(2599):

        if not os.path.exists(os.path.join(opFolder, '%d_output.png'%i)):
            continue
        if not os.path.exists(os.path.join(opFolder, '%d_output.png'%(i+1))):
            continue

        if i % 198 == 0 or i % 199 == 0 or i % 200 == 0:
            continue

        flow = np.load(os.path.join('./optical_flow', cate, 'flow_%d.npy'%i))
        flow = torch.from_numpy(flow)
        I0_gt = torch.unsqueeze(transforms.ToTensor()(Image.open(os.path.join(gtFolder, '%d.jpg'%i))), dim=0)
        I0_output = torch.unsqueeze(transforms.ToTensor()(Image.open(os.path.join(opFolder, '%d_output.png'%i))), dim=0)

        I1_gt = torch.unsqueeze(transforms.ToTensor()(Image.open(os.path.join(gtFolder, '%d.jpg'%(i+1)))), dim=0)
        I1_output = torch.unsqueeze(transforms.ToTensor()(Image.open(os.path.join(opFolder, '%d_output.png'%(i+1)))), dim=0)

        warped_gt, _ = backwarp_2d(source=I0_gt, y_displacement=flow[:, 0, ...], x_displacement=flow[:, 1, ...])
        warped_output, _ = backwarp_2d(source=I0_output, y_displacement=flow[:, 0, ...], x_displacement=flow[:, 1, ...])

        M = np.exp(alpha * torch.nn.MSELoss(reduction='sum')(I1_gt, warped_gt))
        Ltc = M * torch.nn.L1Loss(reduction='sum')(I1_output, warped_output)
        LtcGt = M * torch.nn.L1Loss(reduction='sum')(I1_gt, warped_gt)
        lossList.append(Ltc.item())
        lossGtList.append(LtcGt.item())
        print(cate, i, M.item(), Ltc.item(), LtcGt.item())

    print(sum(lossList)/len(lossList), sum(lossGtList)/len(lossGtList))
    with open(os.path.join(outputPath, 'res_tmp.txt'), 'a+') as f:
        f.writelines('%s\toutput Ltmp:%f, gt Ltmp:%f\n'%(cate, sum(lossList)/len(lossList), sum(lossGtList)/len(lossGtList)))



