import numpy as np
import os
import os.path as osp
import shutil
from mmcv.utils import check_file_exist, is_str, mkdir_or_exist
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('save_path', help='path to save restoration result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    save_path = args.save_path
    for i in range(1, 21):
        print(i)
        _img_path = os.path.join(img_path, f'{i:03d}')
        _save_path = os.path.join(save_path, f'{i:03d}')
        mkdir_or_exist(_save_path)
        imagenames = os.listdir(_img_path)
        for imagename in imagenames:
            if int(imagename.split('.')[0]) % 10 == 0:
                new_filename_path = os.path.join(_save_path, imagename)
                # print(new_filename_path)
                old_filename_path = os.path.join(_img_path, imagename)
                shutil.copy(old_filename_path, new_filename_path)


if __name__ == "__main__":
    main()
