import sys
sys.path.append('..')
from model import FusionModel
from tools import hybrid_storage, representation
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


def calssim(gt, pred):
    return structural_similarity(gt, pred, data_range=gt.max() - gt.min(), multichannel=False, gaussian_weights=True)


class UHSEDataset:
    def __init__(self, p, nb_of_timebin=5):
        folder = p.strip('\n')
        self.storage = [hybrid_storage.HybridStorage.from_folders(event_folder=folder,
                                                                  gt_image_folder=os.path.join(folder, 'frame'),
                                                                  image_file_template="*.jpg",
                                                                  gt_img_timestamps_file='../ts_frame.txt',
                                                                  event_name='event.npy')]

        self.idx = []
        for k in range(len(self.storage)):
            self.idx += [k] * (13 * 198)
        self.start_idx = [k*13*198 for k in range(len(self.storage))]
        self.nb_of_time_bin = nb_of_timebin
        self.name = os.path.join(os.path.split(folder)[-1][:-1], os.path.split(folder)[-1])

    def __len__(self):
        length = 0
        for k in range(len(self.storage)):
            length += 13*198
        return length

    def __getitem__(self, idx1):
        sample_idx = self.idx[idx1]
        start_idx = self.start_idx[sample_idx]

        idx0 = idx1 - start_idx

        idx = int(idx0/198)*200+idx0%198+1

        t = self.storage[sample_idx]._gtImages._timestamps[idx]

        idx_r = int(idx0/198 + 1)*200 - 1
        idx_l = int(idx0/198)*200

        left_image = self.storage[sample_idx]._gtImages._images[idx_l]
        right_image = self.storage[sample_idx]._gtImages._images[idx_r]

        t_left = self.storage[sample_idx]._gtImages._timestamps[idx_l]
        t_right = self.storage[sample_idx]._gtImages._timestamps[idx_r]

        gt_image = self.storage[sample_idx]._gtImages._images[idx]
        duration_left = t-t_left
        duration_right = t_right-t

        event_left = self.storage[sample_idx]._events.filter_by_timestamp(t_left, duration_left)
        event_right = self.storage[sample_idx]._events.filter_by_timestamp(t, duration_right)

        n_left = n_right = self.nb_of_time_bin
        event_left_forward = representation.to_count_map(event_left, n_left).clone()
        event_right_forward = representation.to_count_map(event_right, n_right).clone()
        left_voxel_grid = representation.to_voxel_grid(event_left, nb_of_time_bins=5)
        right_voxel_grid = representation.to_voxel_grid(event_right, nb_of_time_bins=5)

        event_right.reverse()
        event_left.reverse()
        event_left_backward = representation.to_count_map(event_left, n_left)
        event_right_backward = representation.to_count_map(event_right, n_right)
        events_forward = np.concatenate((event_left_forward, event_right_forward), axis=-1)
        events_backward = np.concatenate((event_right_backward, event_left_backward), axis=-1)

        weight = duration_left / (duration_left + duration_right)

        surface = self.storage[sample_idx]._events.filter_by_timestamp(t-100, 200)
        surface = representation.to_count_map(surface)

        return events_forward, events_backward, left_image, right_image, gt_image, weight, n_left, n_right, surface, left_voxel_grid, right_voxel_grid, idx


def showMessage(message, file):
    print(message)
    with open(file, 'a') as f:
        f.writelines(message + '\n')


def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    cv2.imwrite(path, np.array(img[0, 0].cpu() * 255))


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    import torch
    from torch import nn
    from model import Metric

    device = 'cuda'
    input_img_channel = 1
    nb_of_time_bin = 15
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    theta = [3, 5, 10]
    tauSr = [1, 2, 4]
    tauRef = [1, 2, 4]
    scaleRef = [1, 1, 1]
    tauRho = [1, 1, 10]
    scaleRho = [1, 1, 10]

    ckpt_path = './ckpt/ckpt_THU_HSEVI.pth'
    run_dir = os.path.split(ckpt_path)[0]
    print('rundir:', run_dir)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    with open('./dataset/THU-HSEVI/test.txt', 'r') as f:
        lines = f.readlines()

    op_dir = os.path.join(run_dir, 'HSEVI')
    os.makedirs(op_dir, exist_ok=True)
    with open(os.path.join(op_dir, 'ckpt.txt'), 'w') as f:
        f.writelines(ckpt_path+'\n')

    testFolder = [UHSEDataset(lines[k], nb_of_timebin=nb_of_time_bin) for k in range(len(lines))]
    testLoader = [torch.utils.data.DataLoader(testFolder[k], batch_size=1, shuffle=False, pin_memory=True, num_workers=1) for k in range(len(lines))]

    model = FusionModel(netParams, hidden_number=32, theta=theta, tauSr=tauSr, tauRef=tauRef, scaleRef=scaleRef,
                        tauRho=tauRho, scaleRho=scaleRho, channel=1)
    model = nn.DataParallel(model)
    model = model.to(device)
    print('==> loading existing model:', ckpt_path)
    model_info = torch.load(ckpt_path)
    model.load_state_dict(model_info['state_dict'])

    loss_l1 = nn.L1Loss()
    testMetirc = Metric()

    with torch.no_grad():
        model.eval()
        psnr, ssim = [], []
        for loader in testLoader:
            opFolder = os.path.join(op_dir, loader.dataset.name)
            os.makedirs(opFolder, exist_ok=True)
            count = 0
            for i, (events_forward, events_backward, left_image, right_image, gt_image, weight, n_left, n_right, surface, left_voxel_grid, right_voxel_grid, n) in enumerate(loader):
                events_forward = events_forward.cuda()
                events_backward = events_backward.cuda()
                left_image = left_image.cuda()
                right_image = right_image.cuda()
                gt_image = gt_image.cuda()
                weight = weight.cuda()
                surface = surface.cuda()
                left_voxel_grid = left_voxel_grid.cuda()
                right_voxel_grid = right_voxel_grid.cuda()

                output = model(events_forward, events_backward, left_image, right_image, weight, n_left, n_right, surface, left_voxel_grid, right_voxel_grid)

                saveImg(output, os.path.join(opFolder, '%d_output.png' % n))

                L1Loss = loss_l1(output, gt_image)
                testMetirc.update(L1Loss=L1Loss, Lpips=torch.tensor([0]), FeaLoss=torch.tensor([0]), total=L1Loss)

                avg = testMetirc.get_average_epoch()
                message = 'Test, Iter [%d]/[%d], L1:%f' % (n, 2600, avg[0])
                print(message)
                count += 1

