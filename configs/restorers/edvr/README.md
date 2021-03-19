# EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

## Introduction

[ALGORITHM]

```latex
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

## Results and Models

Evaluated on RGB channels.

The metrics are `PSNR / SSIM`.

|                                        Method                                         |       REDS4       |                                                                                                                  Download                                                                                                                   |
| :-----------------------------------------------------------------------------------: | :---------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [edvrm_wotsa_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py) | 30.3430 /  0.8664 | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522_141644.log.json) |
|       [edvrm_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_x4_g8_600k_reds.py)       | 30.4194 / 0.8684  |       [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622-ba4a43e4.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622_102544.log.json)       |





#### Results on compress test without multi-scale test
validation: final 20 clips compress
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_60k_small](/configs/restorers/edvr/edvr_g8_60k_compress.py)                         | 256x256 | 64 | 64  | 10 | 2e-4  | 5 | 30.8516 | 45.5h / 4 |
| [EDVR_120k_256_large](/configs/restorers/edvr/edvr_g8_120k_large_compress.py)             | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | 31.7988 | 31.7h / 8 |
| [EDVR_120k_384_large](/configs/restorers/edvr/edvr_g8_120k_384_large_compress.py)             | 384x384 | 16 | 128 | 40 | 2e-4  | 5 | 31.8575 | 78.0h / 4|
| [EDVR_60k_384_large](/configs/restorers/edvr/edvr_g8_60k_384_large_compress.py)           | 384x384 | 16 | 128 | 40 | 2e-4  | 5 | 31.7008 | 29.0h / 4 |
| [EDVR_60k_256_large](/configs/restorers/edvr/edvr_g8_60k_large_compress.py)               | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | 31.6701 | 14.5h / 8 | 
| [EDVR_60k_128_large](/configs/restorers/edvr/edvr_g8_60k_128_large_compress.py)           | 128x128 | 32 | 128 | 40 | 2e-4  | 5 | 31.3463   | 36.2h / 4 |
| [EDVR_60k_64_large](/configs/restorers/edvr/edvr_g8_60k_64_large_compress.py)             | 64x64   | 32 | 128 | 40 | 2e-4  | 5 | 31.2391   | 34.3h / 4|
| [EDVR_60k_large_lrx2](/configs/restorers/edvr/edvr_g8_60k_large_compress_lrx2.py)             | 256x256   | 32 | 128 | 40 | 4e-4  | 5 | failure   | - |
| [EDVR_60k_large_lrx4](/configs/restorers/edvr/edvr_g8_60k_large_compress_lrx4.py)             | 256x256   | 32 | 128 | 40 | 8e-4  | 5 | failure   | - |
| [EDVR_60k_256_f9_large](/configs/restorers/edvr/edvr_g8_60k_large_f9_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 9 | 31.6593 | 41.0h / 4| 
| [EDVR_60k_256_f7_large](/configs/restorers/edvr/edvr_g8_60k_large_f7_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 7 | 31.6745 | 44.0h / 4| 
| [EDVR_60k_256_f5_large](/configs/restorers/edvr/edvr_g8_60k_large_f5_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.6495 | 19.0h / 4 | 
| [EDVR_60k_256_f3_large](/configs/restorers/edvr/edvr_g8_60k_large_f3_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 3 | 31.6029 | 16.5h / 4 | 
| [EDVR_600k_large](/configs/restorers/edvr/edvr_g8_600k_large_compress.py)                       | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | <b>31.9174</b>  | 148h / 8 | 
| [EDVR_600k_large](/configs/restorers/edvr/edvr_g8_600k_large_compress.py)                       | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | <b>31.6241(c)</b>  | 148h / 8 | 
| [EDVR_600k_large_twice](/configs/restorers/edvr/edvr_g8_600k_large_compress.py)                       | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | 31.1380 | - | 
| [EDVR_600k_384_large](/configs/restorers/edvr/edvr_g8_600k_384_large_compress.py)                       | 384x384 | 32 | 128 | 40 | 2e-4  | 5 | 31.8942 | | 
| [EDVR_750k_384_large](/configs/restorers/edvr/edvr_g8_600k_384_large_compress.py)                       | 384x384 | 32 | 128 | 40 | 2e-4  | 5 | 31.8734 | | 
| [EDVR_600k_496_large](/configs/restorers/edvr/edvr_g8_600k_496_large_compress.py)                       | 496x496 | 32 | 128 | 40 | 2e-4  | 5 | doing | | 
| [EDVR_600k_large_s2_l](/configs/restorers/edvr/edvr_g8_600k_large_s2_l_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | <b>32.0961</b> | 190h / 4 | 
| [600k_large_s2_l_twice](/configs/restorers/edvr/edvr_g8_600k_large_s2_l_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 32.0878 | - | 
 [EDVR_600k_large_wopre](/configs/restorers/edvr/edvr_g8_600k_large_wopre_compress.py)                       | 128x128 | 16 | 128 | 40 | 2e-4  | 5 | doing(crop) | 323h / 8| 
 [EDVR_600k_large_wopre2](/configs/restorers/edvr/edvr_g8_600k_large_wopre2_compress.py)                       | 128x128 | 16 | 128 | 40 | 2e-4  | 5 | doing(crop) | 323h / 8| 

validation: final 20 clips compress3
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_60k_256_large_3](/configs/restorers/edvr/edvr_g8_60k_large_compress3.py)               | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | 29.9823 | 28.0h / 4| 
| [EDVR_60k_256_f9_large](/configs/restorers/edvr/edvr_g8_60k_large_f9_compress3.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 9 | 30.4767 | 41.0h / 4| 
| [EDVR_60k_256_f7_large](/configs/restorers/edvr/edvr_g8_60k_large_f7_compress3.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 7 | - | 44.0h / 4| 
| [EDVR_60k_256_f5_large](/configs/restorers/edvr/edvr_g8_60k_large_f5_compress3.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 30.4670 | 24h / 4 | 
| [EDVR_60k_256_f3_large](/configs/restorers/edvr/edvr_g8_60k_large_f3_compress3.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 3 | 30.0370 | 17.5h / 4 | 
| [EDVR_600k_384_large](/configs/restorers/edvr/edvr_g8_600k_384_large_compress3.py)                       | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 30.5028 | 323h / 4| 
| [EDVR_600k_384_large](/configs/restorers/edvr/edvr_g8_600k_384_large_compress3.py)                       | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 30.1336(c) | 323h / 4| 
 [EDVR_600k_large_wopre](/configs/restorers/edvr/edvr_g8_600k_large_wopre_compress3.py)                       | 128x128 | 16 | 128 | 40 | 2e-4  | 5 | doing(crop) | 323h / 8| 



validation: final 20 clips compress croped 512
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_60k_256_f5_large](/configs/restorers/edvr/edvr_g8_60k_large_f5_compress_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.3648 |  | 
| [EDVR_60k_256_f5_large_car](/configs/restorers/edvr/edvr_g8_60k_large_f5_car_compress_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.2359 |  | 
| [EDVR_60k_256_f5_large_nonlocal](/configs/restorers/edvr/edvr_g8_60k_large_f5_nonlocal_compress_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 30.5610 |  15h / 4| 
| [EDVR_60k_256_f5_large_car_nonlocal](/configs/restorers/edvr/edvr_g8_60k_large_f5_car_nonlocal_compress_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.0659 |  | 
| [EDVR_600k_256_f5_large_car_nonlocal](/configs/restorers/edvr/edvr_g8_60k_large_f5_car_nonlocal_compress_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | doing |  | 



validation: final 20 clips compress no tsa warmup
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_60k_256_f9_large](/configs/restorers/edvr/edvr_g8_60k_large_f9_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 9 | 30.9062 | 41.0h / 4| 
| [EDVR_60k_256_f7_large](/configs/restorers/edvr/edvr_g8_60k_large_f7_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 7 | 31.7535 | 44.0h / 4| 
| [EDVR_60k_256_f5_large](/configs/restorers/edvr/edvr_g8_60k_large_f5_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.2660 | 19.0h / 4 | 
| [EDVR_60k_256_f3_large](/configs/restorers/edvr/edvr_g8_60k_large_f3_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 3 | 30.8115 | 16.5h / 4 | 
| [EDVR_60k_384_large](/configs/restorers/edvr/edvr_g8_60k_384_large_compress.py)           | 384x384 | 16 | 128 | 40 | 2e-4  | 5 | 31.7640 | 29.0h / 4 |
| [EDVR_60k_256_large](/configs/restorers/edvr/edvr_g8_60k_large_compress.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.7556 | 14.5h / 8 | 
| [EDVR_60k_128_large](/configs/restorers/edvr/edvr_g8_60k_128_large_compress.py)           | 128x128 | 16 | 128 | 40 | 2e-4  | 5 | 31.5561   | 36.2h / 4 |
| [EDVR_60k_64_large](/configs/restorers/edvr/edvr_g8_60k_64_large_compress.py)             | 64x64   | 16 | 128 | 40 | 2e-4  | 5 | 30.8148   | 34.3h / 4|

validation: final 20 clips compress train 30 clips
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_100k_256_f9_large](/configs/restorers/edvr/edvr_g8_100k_large_f9_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 9 | doing | 41.0h / 4| 
| [EDVR_100k_256_f7_large](/configs/restorers/edvr/edvr_g8_100k_large_f7_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 7 | doing | 44.0h / 4| 
| [EDVR_100k_256_f5_large](/configs/restorers/edvr/edvr_g8_100k_large_f5_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | doing | 19.0h / 4 | 
| [EDVR_100k_256_f3_large](/configs/restorers/edvr/edvr_g8_100k_large_f3_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 3 | 31.3290 | 16.5h / 4 | 
| [EDVR_100k_384_large](/configs/restorers/edvr/edvr_g8_100k_384_large_compress_m.py)           | 384x384 | 16 | 128 | 40 | 2e-4  | 5 |  - | 29.0h / 4 |
| [EDVR_100k_256_large](/configs/restorers/edvr/edvr_g8_100k_256_large_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.3312 | 14.5h / 8 | 
| [EDVR_100k_128_large](/configs/restorers/edvr/edvr_g8_100k_128_large_compress_m.py)           | 128x128 | 16 | 128 | 40 | 2e-4  | 5 |  31.2906  | 36.2h / 4 |
| [EDVR_100k_64_large](/configs/restorers/edvr/edvr_g8_100k_64_large_compress_m.py)             | 64x64   | 16 | 128 | 40 | 2e-4  | 5 | -| 34.3h / 4|


validation: final 20 clips compress train 30 clips no tsa warmup
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_100k_256_f9_large](/configs/restorers/edvr/edvr_g8_100k_large_f9_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 9 | 31.3222 | 41.0h / 4| 
| [EDVR_100k_256_f7_large](/configs/restorers/edvr/edvr_g8_100k_large_f7_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 7 | 31.3983 | 44.0h / 4| 
| [EDVR_100k_256_f5_large](/configs/restorers/edvr/edvr_g8_100k_large_f5_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.3554 | 19.0h / 4 | 
| [EDVR_100k_256_f3_large](/configs/restorers/edvr/edvr_g8_100k_large_f3_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 3 | 31.3204 | 16.5h / 4 | 
| [EDVR_100k_384_large](/configs/restorers/edvr/edvr_g8_100k_384_large_compress_m.py)           | 384x384 | 16 | 128 | 40 | 2e-4  | 5 |  31.3168| 29.0h / 4 |
| [EDVR_100k_256_large](/configs/restorers/edvr/edvr_g8_100k_256_large_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.3554 | 14.5h / 8 | 
| [EDVR_100k_256_large_b8](/configs/restorers/edvr/edvr_g8_100k_256_large_compress_m.py)               | 256x256 | 32 | 128 | 40 | 2e-4  | 5 | 31.3334 | 14.5h / 8 | 
| [EDVR_100k_256_large_b2](/configs/restorers/edvr/edvr_g8_100k_256_large_compress_m.py)               | 256x256 | 8 | 128 | 40 | 2e-4  | 5 | 31.3634 | 14.5h / 8 | 
| [EDVR_100k_128_large](/configs/restorers/edvr/edvr_g8_100k_128_large_compress_m.py)           | 128x128 | 16 | 128 | 40 | 2e-4  | 5 |  31.2790  | 36.2h / 4 |
| [EDVR_100k_64_large](/configs/restorers/edvr/edvr_g8_100k_64_large_compress_m.py)             | 64x64   | 16 | 128 | 40 | 2e-4  | 5 | 31.2343 | 34.3h / 4|
| [EDVR_100k_256_large_interval1_2](/configs/restorers/edvr/edvr_g8_100k_256_large_interval1_2_compress_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.3403 | 14.5h / 8 | 
| [EDVR_100k_256_f5_large_car](/configs/restorers/edvr/edvr_g8_100k_256_large_car_compress_m_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.3858 |  | 
 | [EDVR_100k_256_f7_car_large](/configs/restorers/edvr/edvr_g8_100k_large_f7_car_compress_m.py)                       | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.4170 | 323h / 4| 
 | [EDVR_100k_256_f7_car_large_b2](/configs/restorers/edvr/edvr_g8_100k_large_f7_car_compress_m.py)                       | 256x256 | 8 | 128 | 40 | 2e-4  | 5 | doing | 323h / 4| 

validation: final 20 clips compress train 30 clips no tsa warmup croped 512
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_100k_256_f5_large](/configs/restorers/edvr/edvr_g8_100k_256_large_compress_m_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.0789(temp) /31.0806 |  | 
| [EDVR_100k_256_f5_large_car](/configs/restorers/edvr/edvr_g8_100k_256_large_car_compress_m_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.1022 |  | 
| [EDVR_200k_256_f5_large_car](/configs/restorers/edvr/edvr_g8_100k_256_large_car_compress_m_crop2.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.1172 |  | 
| [EDVR_100k_256_f5_large_nonlocal](/configs/restorers/edvr/edvr_g8_100k_256_large_nonlocal_compress_m_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 30.9610 |  | 
| [EDVR_100k_256_f5_large_car_nonlocal](/configs/restorers/edvr/edvr_g8_100k_256_large_car_nonlocal_compress_m_crop.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 30.9306 |  | 
| [EDVR_100k_256_f5_large_car_nonlocal2](/configs/restorers/edvr/edvr_g8_100k_256_large_car_nonlocal_compress_m_crop2.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.0478 |  | 
| [EDVR_50k_256_f5_large_car_nonlocal3](/configs/restorers/edvr/edvr_g8_100k_256_large_car_nonlocal_compress_m_crop3.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.0995 |  | 
| [EDVR_50k_256_f5_large_car_nonlocal4](/configs/restorers/edvr/edvr_g8_100k_256_large_car_nonlocal_compress_m_crop5.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.1315 |  | 
| [EDVR_50k_256_f5_large_car_nonlocal5](/configs/restorers/edvr/edvr_g8_100k_256_large_car_nonlocal_compress_m_crop4.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 31.1260 |  | 
validation: final 20 clips compress3 train 30 clips no tsa warmup croped 512
| Arch | input_size | batch_size| mid_channel | blocks_reconst | lr | frames| PSNR | Time|
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | 
| [EDVR_100k_256_f7_large_car](/configs/restorers/edvr/edvr_g8_100k_256_large_f7_car_compress3_m.py)               | 256x256 | 16 | 128 | 40 | 2e-4  | 5 | 29.5176 |  |


# Traning Setting
defalut: 

- input size: 256 x 256
- total batchsize: 32
- gpu nums: 8
- batch per gpu: 4


```
input size * batch per gpu = 256 x 256 x 4. This is a constant.
If change input size or batch per gpu, you will scale the loss weight correspondingly.
```
