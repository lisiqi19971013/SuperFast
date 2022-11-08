# SuperFast: 200× Video Frame Interpolation via Event Camera
This repository is for the IEEE T-PAMI 2022 paper "SuperFast: 200× Video Frame Interpolation via Event Camera"

## Requirements

1. Python 3 with the following packages installed:
   * torch==1.9.0
   * torchvision==0.10.0
   * tqdm==4.62.3
   * numpy==1.21.2
   * imageio==2.9.0
   * Pillow==8.3.1
2. slayerPytorch
   - See https://github.com/bamsumit/slayerPytorch to install the slayerPytorch for the SNN simulation.
3. cuda
   - A **CUDA** enabled **GPU** is required for training any model. We test our code with CUDA 11.1 V11.1.105 on NVIDIA 3090 GPUs.



## Data preparing

1. Download our $\text{THU}^\text{HSEVI}$ dataset from: 
2. Download the HS-ERGB dataset from https://github.com/uzh-rpg/rpg_timelens.
3. Change the corresponding data path in each .py file



## Train and test

### Testing

After training, run the following code to generating results. **NOTICE: the default output path is (./dataset/N-MNIST/ResConv/HRPre)**.

```shell
>>> python testNmnist.py
```



### Calculate metrics

After generating results, run calRMSE.py to calculate the metrics. **NOTICE: the output path should be changed**.

```shell
>>> python calRMSE.py
```
