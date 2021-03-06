import argparse

import mmcv
import torch
import numpy as np
from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('save_path', help='path to save restoration result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    output = restoration_inference(model, args.img_path)
    output = tensor2img(output)
    # print(np.shape(output))
    mmcv.imwrite(output, args.save_path)
    if args.imshow:
        mmcv.imshow(output, 'predicted restoration result')


if __name__ == '__main__':
    main()
