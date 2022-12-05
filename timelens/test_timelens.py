from timelens.common import event, representation, os_tools
from torchvision import transforms
from PIL import Image
import tqdm
import os
import numpy as np
from timelens.model import attention_average_network
from tools import hybrid_storage


class Dataset:
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
        idx = int(idx0 / 198) * 200 + idx0 % 198 + 1

        t = self.storage[sample_idx]._gtImages._timestamps[idx]
        idx_r = int(idx0 / 198 + 1) * 200 - 1
        idx_l = int(idx0 / 198) * 200

        left_image = self.storage[sample_idx]._gtImages._images[idx_l]
        right_image = self.storage[sample_idx]._gtImages._images[idx_r]

        t_left = self.storage[sample_idx]._gtImages._timestamps[idx_l]
        t_right = self.storage[sample_idx]._gtImages._timestamps[idx_r]

        gt_image = self.storage[sample_idx]._gtImages._images[idx]

        duration_left = t - t_left
        duration_right = t_right - t

        event_left = self.storage[sample_idx]._events.filter_by_timestamp(t_left, duration_left)
        event_left_r = event_left.copy()
        event_right = self.storage[sample_idx]._events.filter_by_timestamp(t, duration_right)

        event_right = representation.to_voxel_grid(event_right, self.nb_of_time_bin)
        event_left = representation.to_voxel_grid(event_left, self.nb_of_time_bin)

        event_left_r.reverse()
        event_left_r = representation.to_voxel_grid(event_left_r, self.nb_of_time_bin)

        # name = self.storage[sample_idx]._gtImages.names[idx]

        weight = (t - t_left) / (t_right - t)

        return event_left, event_left_r, event_right, left_image, right_image, gt_image, weight, idx



def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    cv2.imwrite(path, np.array(img[0, 0].cpu() * 255))


if __name__ == '__main__':
    from timelens.common import pytorch_tools
    import torch
    import torch.nn as nn
    import cv2

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    checkpoint_file = './ckpt_timelens.pth'
    op_dir = './output_timelens'
    os.makedirs(op_dir, exist_ok=True)

    device = 'cuda'

    model = attention_average_network.AttentionAverage(inChannel=1)
    a = torch.load(checkpoint_file)
    model.load_state_dict(a['state_dict'])

    with open('../dataset/THU-HSEVI/test.txt', 'r') as f:
        lines = f.readlines()

    folder_list = [p.strip('\n') for p in lines]
    pytorch_tools.set_fastest_cuda_mode()

    model = model.to(device)
    psnrList, ssimList = [], []
    for k in range(len(folder_list)):
        testFolder = Dataset(folder_list[k])
        testLoader = torch.utils.data.DataLoader(testFolder, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
        test_total_iter = len(testLoader)
        print(test_total_iter)
        op = os.path.join(op_dir, folder_list[k].split('/')[-2], folder_list[k].split('/')[-1])
        os.makedirs(op, exist_ok=True)

        with torch.no_grad():
            model.eval()
            for i, (event_left, event_left_r, event_right, left_image, right_image, gt_image, weight, idx) in enumerate(testLoader):
                example = {"before": {"rgb_image_tensor": left_image, "voxel_grid": event_left, "reversed_voxel_grid": event_left_r},
                    "middle": {"weight": list(np.array(weight))}, "after": {"rgb_image_tensor": right_image, "voxel_grid": event_right}}
                example = pytorch_tools.move_tensors_to_cuda(example)
                output, _ = model.run_attention_averaging(example)

                saveImg(output, os.path.join(op, '%d_output.png'%idx))
                print('%d/6, %d/2600'%(k, idx))

