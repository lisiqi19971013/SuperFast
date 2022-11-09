import glob
import os
import time
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import numpy as np


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def calssim(gt, pred):
    return structural_similarity(gt, pred, multichannel=True, gaussian_weights=True)


outputPath = './ckpt/HSERGB_x30'
datasetPath = '/home/lisiqi/data/HSERGB_CVPR_FULL'

subset = 'close' # 'far'

if subset == 'close':
    testList = ['confetti', 'fountain_bellevue2', 'water_bomb_eth_01', 'water_bomb_floor_01', 'spinning_plate',
                'baloon_popping', 'candle', 'fountain_schaffhauserplatz_02', 'spinning_umbrella']
else:
    testList = ['lake_01', 'bridge_lake_03', 'bridge_lake_01', 'lake_03', 'sihl_03', 'kornhausbruecke_letten_random_04']

f = open(os.path.join(outputPath, subset, 'res.txt'), 'w')
psnrList = []
ssimList = []

for c in testList:
    folder = os.path.join(outputPath, subset, c, 'test')
    imgList = os.listdir(folder)
    imgList.sort()
    psnr = []
    ssim = []
    for i in imgList:
        op = cv2.imread(os.path.join(folder, i))
        gt = cv2.imread(os.path.join(datasetPath, subset, 'test', c, 'images_corrected', i))
        p = calpsnr(gt, op)
        s = calssim(gt, op)
        psnr.append(p)
        ssim.append(s)
        psnrList.append(p)
        ssimList.append(s)
        print(c, int(i.strip('.png')), len(imgList), p, s)
    f.writelines('%s, psnr:%f, ssim:%f\n' % (c, sum(psnr) / len(psnr), sum(ssim) / len(ssim)))
f.writelines('Total, psnr:%f, ssim:%f\n' % (sum(psnrList) / len(psnrList), sum(ssimList) / len(ssimList)))
f.close()