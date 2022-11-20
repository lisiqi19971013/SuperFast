# SuperFast: 200× Video Frame Interpolation via Event Camera
This repository is for the **IEEE T-PAMI** **2022** paper "*SuperFast: 200× Video Frame Interpolation via Event Camera*". Please cite and star our work if our code or dataset is used.

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

1. Download our $\text{THU}^\text{HSEVI}$ dataset from https://pan.baidu.com/s/13GVxDUc_81_DedRnIHIFSA (extract code: 69g5) or from https://drive.google.com/file/d/1im7Idfx7VP8c1krIdyS7zeiKTmgsU6qR/view?usp=sharing. 
2. Change the absolute path in the files "train.txt", "train1.txt", "test.txt", and "test1.txt". The files "train.txt" and "test.txt" are the common split of our dataset, and the files "train1.txt" and "test1.txt" are the split of our dataset according to the scenario (Tab. 7).
3. Download the HS-ERGB dataset from: https://github.com/uzh-rpg/rpg_timelens.
4. Download the pre-trained model on both our $\text{THU}^\text{HSEVI}$ dataset and HS-ERGB dataset from https://pan.baidu.com/s/1pzyXBRR3BIohaNvaUGbSCQ (extract code: 3rrf) or from https://drive.google.com/drive/folders/1-EqPqIQ9MyEz4Nj43jSkNfcdxikhRc1w?usp=sharing.




## Testing

### Test on $\text{THU}^\text{HSEVI}$ dataset

Change the variables "**ckpt_path**" and "**data_path**" in the file "**test_THU_HSEVI.py**" accordingly. For the common split of our $\text{THU}^\text{HSEVI}$ dataset, change the variable "**split_by_scenario**" to **False**. For the scenario-based split, change the variable "**split_by_scenario**" to **True**. 

Run the following code to generate output results.

```shell
>>> python test_THU_HSEVI.py
```

**NOTE: Default output paths are "./ckpt/THU-HSEVI" for the common split, "./ckpt/THU-HSEVI1" for the scenario-based split.**



Then, run the following code to calculate performance metrics. Don't forget to change the variables "**outputPath**" and "**datasetPath**" correspondingly. 

```shell
>>> python calMetirc_THU_HSEVI.py
```

The metrics will be written in the file "**res.txt**" in "**outputPath**".



As for the temporal error, download the optical flow from: https://pan.baidu.com/s/12W7n6OUbNYtg4JsPtAUXQg (extract code: 215c) or from: https://drive.google.com/file/d/1BlL8tJ2rxH7wjHDvjRupOIg_4U4Ghkhj/view?usp=share_link, and place it in the folder "**./optical_flow**". Then, change the variables "**outputPath**" and "**datasetPath**" correspondingly. Run the following code to calculate the temporal error.

```shell
>>> python calTmpLoss.py
```



### Test on HS-ERGB dataset

We test the performance of our proposed method on HS-ERGB dataset while interpolate 7 or 30 frames. Change the variables "**ckpt_path**" and "**data_path**" in the file "**test_HSERGB.py**" accordingly. Then, change the variables "**subset='close' or 'far'**" and "**number_of_frames_to_skip=7 or 30**".

Run the following code to generate output results.

```shell
>>> python test_HSERGB.py
```

**NOTE: Default output paths are "./ckpt/HSERGB_x7" and "./ckpt/HSERGB_x30".**

Then, run the following code to calculate performance metrics. Don't forget to change the variables "**outputPath**" and "**datasetPath**" correspondingly, and choose the "**subset**" from "close" and "far". 

```shell
>>> python calMetirc_HSERGB.py
```

The metrics will be written in the file "**res.txt**" in "**outputPath**".



## Citation

```bib
@article{SuperFast,
  author={Gao, Yue and Li, Siqi and Li, Yipeng and Guo, Yandong and Dai, Qionghai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{SuperFast: 200$\boldsymbol{\times}$ Video Frame Interpolation via Event Camera}}, 
  year={2022},
  volume={},
  number={},
  pages={},
  publisher={IEEE}
  }
```

