import sys
sys.path.append('..')
from model import FusionModel
from tools import representation, event
from torchvision import transforms
import cv2
from PIL import Image


class HSERGBDataset:
    def __init__(self, subset='close', mode='train', folder='', number_of_frames_to_skip=15, nb_of_time_bin=20):
        if mode not in ['train', 'test']:
            raise ValueError

        self.folder = os.path.join('/home/lisiqi/data/HSERGB', subset, mode, folder)
        self.number_of_frames_to_skip = number_of_frames_to_skip
        print('skip %d frame' % self.number_of_frames_to_skip)
        self.mode = mode
        self.nb_of_time_bin = nb_of_time_bin
        self.generate_data()
        self.folder = folder

    def __len__(self):
        return len(self.idx)

    def generate_data(self):
        self.left_image = []
        self.right_image = []
        self.gt_image = []
        self.event = []
        self.gt_timestamp = []
        self.idx = []
        self.lr_timestamp = []
        with open(os.path.join(self.folder, 'images_corrected', 'timestamp.txt'), 'r') as f:
            ts = [float(l.strip('\n')) for l in f.readlines()]
        N = len(ts)

        for k in range(int(N/(self.number_of_frames_to_skip+1))-2):
            rand = 0
            start = k*(self.number_of_frames_to_skip+1) + rand
            end = (k+1)*(self.number_of_frames_to_skip+1) + rand

            self.left_image.append(os.path.join(self.folder, 'images_corrected', '%d.png'%start))
            self.right_image.append(os.path.join(self.folder, 'images_corrected', '%d.png'%end))
            self.event.append([os.path.join(self.folder, 'events_aligned', '%d.npz'%k) for k in range(start, end+1)])
            self.gt_image.append([os.path.join(self.folder, 'images_corrected', '%d.png'%k) for k in range(start+1, end)])
            self.gt_timestamp.append([ts[k] for k in range(start+1, end)])
            self.lr_timestamp.append([ts[start], ts[end]])
        for k in range(len(self.left_image)):
            self.idx += [k] * len(self.gt_image[0])
        self.start_idx = [k * len(self.gt_image[0]) for k in range(len(self.left_image))]

    def __getitem__(self, idx):
        seq_idx = self.idx[idx]
        sample_idx = idx - self.start_idx[seq_idx]
        left_image = transforms.ToTensor()(Image.open(self.left_image[seq_idx]))
        right_image = transforms.ToTensor()(Image.open(self.right_image[seq_idx]))

        w, h = left_image.shape[2], left_image.shape[1]
        gt_image = transforms.ToTensor()(Image.open(self.gt_image[seq_idx][sample_idx]))

        events = event.EventSequence.from_npz_files(self.event[seq_idx], h, w)
        ts = self.gt_timestamp[seq_idx][sample_idx]

        duration_left = ts - self.lr_timestamp[seq_idx][0]
        duration_right = self.lr_timestamp[seq_idx][1] - ts

        e_left = events.filter_by_timestamp(events.start_time(), duration_left)
        e_right = events.filter_by_timestamp(ts, duration_right)

        event_left_forward = representation.to_count_map(e_left, self.nb_of_time_bin).clone()
        event_right_forward = representation.to_count_map(e_right, self.nb_of_time_bin).clone()
        left_voxel_grid = representation.to_voxel_grid(e_left, nb_of_time_bins=5)
        right_voxel_grid = representation.to_voxel_grid(e_right, nb_of_time_bins=5)

        e_right.reverse()
        e_left.reverse()
        event_left_backward = representation.to_count_map(e_left, self.nb_of_time_bin)
        event_right_backward = representation.to_count_map(e_right, self.nb_of_time_bin)
        events_forward = np.concatenate((event_left_forward, event_right_forward), axis=-1)
        events_backward = np.concatenate((event_right_backward, event_left_backward), axis=-1)

        weight = duration_left / (duration_left+duration_right)

        surface = events.filter_by_timestamp(ts-200, 400)
        surface = representation.to_count_map(surface)

        return events_forward, events_backward, left_image, right_image, gt_image, weight, \
               [self.nb_of_time_bin, self.nb_of_time_bin], surface, left_voxel_grid, right_voxel_grid, self.gt_image[seq_idx][sample_idx]


def showMessage(message, file):
    print(message)
    with open(file, 'a') as f:
        f.writelines(message + '\n')


def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    img = np.array(img[0].cpu() * 255)
    img1 = np.zeros([img.shape[1], img.shape[2], 3])
    img1[:,:,0] = img[2,:,:]
    img1[:,:,1] = img[1,:,:]
    img1[:,:,2] = img[0,:,:]
    cv2.imwrite(path, img1)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    import numpy as np
    import torch
    from torch import nn
    from model import Metric

    subset = 'close'  # 'far'
    number_of_frames_to_skip = 7  # 30

    assert subset in ['close', 'far']
    assert number_of_frames_to_skip in [7, 30]

    if subset == 'close':
        testList = ['confetti', 'fountain_bellevue2', 'water_bomb_eth_01', 'water_bomb_floor_01', 'spinning_plate',
                    'baloon_popping', 'candle', 'fountain_schaffhauserplatz_02', 'spinning_umbrella']
    else:
        testList = ['lake_01', 'bridge_lake_03', 'bridge_lake_01', 'lake_03', 'sihl_03',
                    'kornhausbruecke_letten_random_04']

    device = 'cuda'
    input_img_channel = 3
    nb_of_time_bin = 8
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}

    theta = [3, 5, 10]
    tauSr = [1, 2, 4]
    tauRef = [1, 2, 4]
    scaleRef = [1, 1, 1]
    tauRho = [1, 1, 10]
    scaleRho = [1, 1, 10]

    ckpt_path = f'./ckpt/ckpt_HSERGB_{subset}_x{number_of_frames_to_skip}.pth'

    run_dir = os.path.split(ckpt_path)[0]
    print('rundir:', run_dir)

    opF = os.path.join(run_dir, f'HSERGB_x{number_of_frames_to_skip}')
    os.makedirs(opF, exist_ok=True)

    with open(os.path.join(opF, 'ckpt.txt'), 'w') as f:
        f.writelines(ckpt_path+'\n')

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = True

    testFolder = [HSERGBDataset(subset, 'test', k, number_of_frames_to_skip, nb_of_time_bin) for k in testList]
    testLoader = [torch.utils.data.DataLoader(testFolder[k], batch_size=1, shuffle=False, pin_memory=False, num_workers=1) for k in range(len(testFolder))]
    total = sum([len(testLoader[k]) for k in range(len(testLoader))])
    model = FusionModel(netParams, hidden_number=32, theta=theta, tauSr=tauSr, tauRef=tauRef, scaleRef=scaleRef,
                        tauRho=tauRho, scaleRho=scaleRho, channel=input_img_channel, fast_ckpt='', slow_ckpt='')
    model = nn.DataParallel(model)
    model = model.to(device)
    print('==> loading existing model:', ckpt_path)
    model_info = torch.load(ckpt_path)
    model.load_state_dict(model_info['state_dict'])

    l1 = torch.nn.L1Loss()
    testMetirc = Metric()

    with torch.no_grad():
        model.eval()
        count = 0
        for loader in testLoader:
            opFolder = os.path.join(opF, subset, loader.dataset.folder, 'test')
            os.makedirs(opFolder, exist_ok=True)
            for i, (events_forward, events_backward, left_image, right_image, gt_image, weight, [n_left, n_right],
                    surface, left_voxel_grid, right_voxel_grid, name) in enumerate(loader):
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

                L1Loss = l1(output, gt_image)
                testMetirc.update(L1Loss=L1Loss, Lpips=torch.tensor([0]), FeaLoss=torch.tensor([0]), total=L1Loss)

                saveImg(output, os.path.join(opFolder, '%06d.png'%int(name[0].split('/')[-1].strip('.png'))))

                if count % 50 == 0:
                    avg = testMetirc.get_average_epoch()
                    message = 'Test, Iter [%d]/[%d], l1:%f' % (count, total, avg[0])
                    print(message)
                count += 1
