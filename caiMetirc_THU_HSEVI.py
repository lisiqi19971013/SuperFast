import os
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def calpsnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred)


def calssim(gt, pred):
    return structural_similarity(gt, pred, multichannel=False, gaussian_weights=True)


cate = ['balloon_bursting', 'chessboard_waving', 'cup_breaking', 'hard_disk_spining', 'water_bomb', 'firing']
core = {'cup_breaking':[600,1000], 'water_bomb':[1400,1800], 'chessboard_waving':[0,100], 'balloon_bursting':[250,350],
        'hard_disk_spining':[0,400], 'firing':[300,500]}


outputPath = './ckpt/THU-HSEVI/'
datasetPath = './dataset/THU-HSEVI'

psnrList = []
ssimList = []

with open(outputPath+"/res.txt", 'a+') as f:
    f.writelines('Metrics on full dataset:\n')

for c in cate:
    if not os.path.exists(os.path.join(outputPath, c)):
        continue
    start, end = 0, 2599

    psnr, ssim = [], []
    for folder in os.listdir(os.path.join(outputPath, c)):
        path = os.path.join(outputPath, c, folder)

        for k in range(start, end+1):
            if not os.path.exists(os.path.join(path, '%d_output.png'%k)):
                continue
            op = cv2.imread(os.path.join(path, '%d_output.png' % k), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join('./dataset/THU-HSEVI', c, folder, 'frame', '%d.jpg'%k), cv2.IMREAD_GRAYSCALE)

            p = calpsnr(gt, op)
            s = calssim(gt, op)
            psnr.append(p)
            ssim.append(s)
            psnrList.append(p)
            ssimList.append(s)
            print(folder, k)

    psnr = sum(psnr) / len(psnr)
    ssim = sum(ssim) / len(ssim)
    message = c + ': PSNR:%f, SSIM:%f\n'%(psnr, ssim)
    print(message)
    with open(outputPath + "/res.txt", 'a+') as f:
        f.writelines(message)

message = '\nTotal: PSNR:%f, SSIM:%f\n'%(sum(psnrList)/len(psnrList), sum(ssimList)/len(ssimList))
print(message)
with open(outputPath+"/res.txt", 'a+') as f:
    f.writelines(message)


with open(outputPath+"/res.txt", 'a+') as f:
    f.writelines('\nMetrics on high-speed clips:\n')

psnrList = []
ssimList = []

for c in cate:
    if not os.path.exists(os.path.join(outputPath, c)):
        continue
    start, end = core[c]

    psnr, ssim = [], []
    for folder in os.listdir(os.path.join(outputPath, c)):
        path = os.path.join(outputPath, c, folder)

        for k in range(start, end + 1):
            if not os.path.exists(os.path.join(path, '%d_output.png' % k)):
                continue
            op = cv2.imread(os.path.join(path, '%d_output.png' % k), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join('./dataset/THU-HSEVI', c, folder, 'frame', '%d.jpg' % k), cv2.IMREAD_GRAYSCALE)

            p = calpsnr(gt, op)
            s = calssim(gt, op)
            psnr.append(p)
            ssim.append(s)
            psnrList.append(p)
            ssimList.append(s)
            print(folder, k)

    psnr = sum(psnr) / len(psnr)
    ssim = sum(ssim) / len(ssim)
    message = c + ': PSNR:%f, SSIM:%f\n' % (psnr, ssim)
    print(message)
    with open(outputPath + "/res.txt", 'a+') as f:
        f.writelines(message)

message = '\nTotal: PSNR:%f, SSIM:%f\n'%(sum(psnrList)/len(psnrList), sum(ssimList)/len(ssimList))
print(message)
with open(outputPath+"/res.txt", 'a+') as f:
    f.writelines(message)
