## Prerequisites

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3 or higher
- CUDA 9.0 or higher 
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)(script will install mmcv automatically)

## A from-scratch setup script(just do it)

Here is a full script for setting up mmediting with conda. 

```shell
conda create -n open-mmlab python=3.7 -y
source activate open-mmlab

conda install pytorch cudatoolkit=10.1 torchvision -c pytorch

git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip install -r requirements.txt
pip install -v -e .
```

### Test a dataset

MMEditing implements **distributed** testing with `MMDistributedDataParallel`.

#### Test with single/multiple GPUs

You can use the following commands to test a dataset with single/multiple GPUs.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```
For example,

```shell
# single-gpu testing
python tools/test.py configs/example_config.py work_dirs/example_exp/example_model_20200202.pth --out work_dirs/example_exp/results.pkl

# multi-gpu testing
./tools/dist_test.sh configs/example_config.py work_dirs/example_exp/example_model_20200202.pth gpu_num --save-path work_dirs/example_exp/results/  
```
## Data and Model Preperation
You should download the data from the [wedsite](https://drive.google.com/drive/folders/1cBuwi-8LYrsLZ3lD9ao3H6Wt5XOoyOwo?usp=sharing), and put it in ``data`` floder. 
You also should download the model from the [wedsite](https://drive.google.com/drive/folders/1xLEr5Vqr36DKIx5sPthdUampwUJms_oA?usp=sharing), and put it in ``works_dir`` floder. 
May be you should change ``config.py`` 'lq', 'gt' and 'anno' path.
After have 3 results, you should ensemble it.
``./tools/image_merge_cal_psnr.py``
## Compress1 && Compress2  test cmd

```

./tools/dist_test.sh configs/restorers/edvr/edvr_g8_600k_large_finetune_compress.py work_dirs/edvr_g8_600k_large_fintune_compress/iter_150000.pth 8 --save-path=./work_dirs/edvr_g8_600k_large_finetune_compress/results_test/ --multi-scale
```

## Compress3 test cmd
```
./tools/dist_test.sh configs/restorers/edvr/edvr_g8_600k_large_finetune_compress3.py work_dirs/edvr_g8_600k_large_fintune_compress3/iter_150000.pth 8 --save-path=./work_dirs/edvr_g8_600k_large_compress3/results_test/ --multi-scale
```


## Detailed Installation

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g. 1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```python
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

you can use more CUDA versions such as 9.0.

c. Clone the mmediting repository.

```shell
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
```

d. Install build requirements and then install mmediting.

```shell
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
```

If you build mmediting on macOS, replace the last command with

```
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' pip install -e .
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmedit with a different CUDA/PyTorch version.

    ```
    pip uninstall mmedit
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

2. Following the above instructions, mmediting is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

4. Some models (such as EDVR in restorers) depend on CUDA ops in `mmcv-full` which is listed in `requirements.txt`. Install it with the default command `pip install -r requirements.txt` need to compile CUDA ops locally and it may take up to 10 mins. Another option is to install pre-compiled `mmcv-full`, visit [MMCV github page](https://github.com/open-mmlab/mmcv#install-with-pip) for concrete instructions. Moreover, if the model you intend to use does not depend on CUDA ops, you could also install the lite version of mmcv with `pip install mmcv` in which CUDA ops is excluded.
