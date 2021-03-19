import numpy as np
import os
import os.path as osp
import shutil
from mmcv.utils import check_file_exist, is_str, mkdir_or_exist
import argparse
import cv2
from mmcv.fileio import FileClient
import mmcv


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


if __name__ == "__main__":
    filepath_train = '/root/cwt1/ntire2021/work_dirs/edvr_g8_600k_large_s2_l_compress/results_600k_train/'
    filepath_train_h = '/root/cwt1/ntire2021/work_dirs/edvr_g8_600k_large_s2_l_compress/results_600k_train_h/'
    filepath_train_w = '/root/cwt1/ntire2021/work_dirs/edvr_g8_600k_large_s2_l_compress/results_600k_train_w/'
    filepath_train_wh = '/root/cwt1/ntire2021/work_dirs/edvr_g8_600k_large_s2_l_compress/results_600k_train_wh/'
    filepath_merge = '/root/cwt1/ntire2021/work_dirs/edvr_g8_600k_large_s2_l_compress/results_600k_train_merge/'
    filepath_gt = '/root/cwt1/ntire2021/data/video_compress_track2/images/train_raw/'
    frame_names = range(1, 21)
    count = 0
    psnr_mean = []
    for frame in frame_names:
        print("frame:", frame)
        _train_path = os.path.join(filepath_train, f'{frame:03d}')
        _train_path_h = os.path.join(filepath_train_h, f'{frame:03d}')
        _train_path_w = os.path.join(filepath_train_w, f'{frame:03d}')
        _train_path_wh = os.path.join(filepath_train_wh, f'{frame:03d}')
        _gt_path = os.path.join(filepath_gt, f'{frame:03d}')
        _save_path = os.path.join(filepath_merge, f'{frame:03d}')
        mkdir_or_exist(_save_path)
        imagenames = os.listdir(_train_path)

        for imagename in imagenames:
            count += 1
            if count % 100 == 0:
                print(count)
            # if count > 10:
            #     break

            img_train = cv2.imread(os.path.join(_train_path, imagename))
            img_train_h = cv2.imread(os.path.join(_train_path_h, imagename))
            img_train_w = cv2.imread(os.path.join(_train_path_w, imagename))
            img_train_wh = cv2.imread(os.path.join(_train_path_wh, imagename))

            img_train = img_train.astype(np.float32)
            img_train_h = cv2.flip(img_train_h, 0).astype(np.float32)
            img_train_w = cv2.flip(img_train_w, 1).astype(np.float32)
            img_train_wh = cv2.flip(img_train_wh, -1).astype(np.float32)
            image_save = (img_train + img_train_w + img_train_h +
                          img_train_wh) / 4
            image_save = image_save.astype(np.uint8)
            imagename_save_path = os.path.join(_save_path, imagename)
            cv2.imwrite(imagename_save_path, image_save)
