import numbers
import os.path as osp

import mmcv

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer
import numpy as np
import torch
import torchvision.transforms.functional as F


@MODELS.register_module()
class EDVR(BasicRestorer):
    """EDVR model for video super-resolution.

    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EDVR, self).__init__(generator, pixel_loss, train_cfg, test_cfg,
                                   pretrained)
        self.with_tsa = generator.get('with_tsa', False)
        self.step_counter = 0  # count training steps

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        if self.step_counter == 0 and self.with_tsa:
            if self.train_cfg is None or (self.train_cfg is not None and
                                          'tsa_iter' not in self.train_cfg):
                raise KeyError(
                    'In TSA mode, train_cfg must contain "tsa_iter".')
            # only train TSA module at the beginging if with TSA module
            for k, v in self.generator.named_parameters():
                if 'fusion' not in k:
                    v.requires_grad = False

        if self.with_tsa and (self.step_counter == self.train_cfg.tsa_iter):
            # train all the parameters
            for v in self.generator.parameters():
                v.requires_grad = True

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        Args:
            imgs (Tensor): Input images.

        Returns:
            Tensor: Restored image.
        """
        out = self.generator(imgs)
        return out

    def multi_scale_test(self, lq):
        # output = self.test_crop_forward(lq)
        output_f = self.flipx4_forward(lq)
        output_r = self.rotx4_forward(lq)
        output = (output_r + output_f) / 2

        # output = self.rotx4_forward(lq)
        return output

    def flipx4_forward(self, lq):
        """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
        Args:
            model (PyTorch model)
            inp (Tensor): inputs defined by the model

        Returns:
            output (Tensor): outputs of the model. float
        """
        # normal
        output_f = self.generator(lq)

        # flip W
        output = self.generator(torch.flip(lq, (-1, )))
        output_f = output_f + torch.flip(output, (-1, ))
        # flip H
        output = self.generator(torch.flip(lq, (-2, )))
        output_f = output_f + torch.flip(output, (-2, ))
        # flip both H and W
        output = self.generator(torch.flip(lq, (-2, -1)))
        output_f = output_f + torch.flip(output, (-2, -1))

        return output_f / 4

    def rotx4_forward(self, lq):
        """Flip testing with rotation self ensemble, i.e., normal,90, 180, 270
        Args:
            model (PyTorch model)
            inp (Tensor): inputs defined by the model

        Returns:
            output (Tensor): outputs of the model. float
        """
        # normal
        output_r = self.generator(lq)
        lq_90 = torch.rot90(lq, 1, [-1, -2])
        lq_180 = torch.rot90(lq_90, 1, [-1, -2])
        lq_270 = torch.rot90(lq_180, 1, [-1, -2])
        # counter-clockwise 90
        output = self.generator(lq_90)
        output_r = output_r + torch.rot90(output, 1, [-2, -1])
        # counter-clockwise 180
        output = self.generator(lq_180)
        output_r = output_r + torch.rot90(
            torch.rot90(output, 1, [-2, -1]), 1, [-2, -1])
        # counter-clockwise 270
        output = self.generator(lq_270)
        output_r = output_r + torch.rot90(
            torch.rot90(torch.rot90(output, 1, [-2, -1]), 1, [-2, -1]), 1,
            [-2, -1])
        return output_r / 4

    def test_crop_forward(self, lq, patch_size=512, stride=500):
        """Testing crop forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w). (n, t, c, h, w).
        """
        n, _, c, height, width = lq.size()
        output = torch.zeros(n, c, height, width).cuda()
        inference_count = torch.zeros(height, width).cuda()
        start_h = 0
        start_w = 0

        for h in list(range(start_h, height - patch_size + 1,
                            stride)) + [height - patch_size]:
            for w in list(range(start_w, width - patch_size + 1,
                                stride)) + [width - patch_size]:
                patch = lq[:, :, :, h:h + patch_size, w:w + patch_size]
                # output_patch = self.flipx4_forward(patch)
                output_patch = self.generator(patch)
                output[:, :, h:h + patch_size, w:w +
                       patch_size] = output_patch[:, :, :, :] + output[:, :,
                                                                       h:h +
                                                                       patch_size,
                                                                       w:w +
                                                                       patch_size]
                inference_count[h:h + patch_size, w:w +
                                patch_size] = inference_count[h:h + patch_size,
                                                              w:w +
                                                              patch_size] + 1

        output = output / inference_count
        return output

    # def forward_test(self,
    #                  lq,
    #                  gt=None,
    #                  meta=None,
    #                  save_image=False,
    #                  save_path=None,
    #                  iteration=None,
    #                  multi_scale=False):
    #     """Testing forward function.

    #     Args:
    #         lq (Tensor): LQ Tensor with shape (n, c, h, w). (n, t, c, h, w).
    #         gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
    #         save_image (bool): Whether to save image. Default: False.
    #         save_path (str): Path to save image. Default: None.
    #         iteration (int): Iteration for the saving image name.
    #             Default: None.

    #     Returns:
    #         dict: Output results.
    #     """
    #     # todo add multi_scale test code
    #     if multi_scale:
    #         output = self.multi_scale_test(lq)
    #     else:
    #         output = self.generator(lq)
    #         output_w = self.generator(torch.flip(lq, (-1, )))
    #         output_h = self.generator(torch.flip(lq, (-2, )))
    #         output_wh = self.generator(torch.flip(lq, (-2, -1)))
    #     if self.test_cfg is not None and self.test_cfg.get('metrics', None):
    #         assert gt is not None, (
    #             'evaluation with metrics must have gt images.')
    #         results = dict(eval_result=self.evaluate(output, gt))
    #     else:
    #         results = dict(lq=lq.cpu(), output=output.cpu())
    #         if gt is not None:
    #             results['gt'] = gt.cpu()

    #     # save image
    #     if save_image:
    #         gt_path = meta[0]['gt_path'][0]
    #         folder_name = meta[0]['key'].split('/')[0]
    #         frame_name = osp.splitext(osp.basename(gt_path))[0]
    #         center_frame_idx = int(frame_name)
    #         # if center_frame_idx % 10 == 0:
    #         if True:
    #             if isinstance(iteration, numbers.Number):
    #                 save_path = osp.join(
    #                     save_path, folder_name,
    #                     f'{frame_name}-{iteration + 1:06d}.png')
    #             elif iteration is None:
    #                 save_path = osp.join(save_path, folder_name,
    #                                      f'{frame_name}.png')
    #                 save_path_w = osp.join(save_path + '_flipw', folder_name,
    #                                        f'{frame_name}.png')
    #                 save_path_h = osp.join(save_path + '_fliph', folder_name,
    #                                        f'{frame_name}.png')
    #                 save_path_wh = osp.join(save_path + '_flipwh', folder_name,
    #                                         f'{frame_name}.png')
    #             else:
    #                 raise ValueError('iteration should be number or None, '
    #                                  f'but got {type(iteration)}')
    #             output = tensor2img(output)
    #             output_w = tensor2img(output_w)
    #             output_h = tensor2img(output_h)
    #             output_wh = tensor2img(output_wh)
    #             # print(np.shape(output))
    #             output_crop = output[:-8, :, :]
    #             output_w_crop = output_w[:-8, :, :]
    #             output_h_crop = output_h[:-8, :, :]
    #             output_wh_crop = output_wh[:-8, :, :]
    #             # print(np.shape(output_crop))
    #             mmcv.imwrite(output_crop, save_path)
    #             mmcv.imwrite(output_w_crop, save_path_w)
    #             mmcv.imwrite(output_h_crop, save_path_h)
    #             mmcv.imwrite(output_wh_crop, save_path_wh)

    #     return results

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None,
                     multi_scale=False):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w). (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        # todo add multi_scale test code
        if multi_scale:
            output = self.multi_scale_test(lq)
        else:
            output = self.generator(lq)
            # output = self.generator(torch.flip(lq, (-1, )))  # w
            # output = self.generator(torch.flip(lq, (-2, )))  # h
            # output = self.generator(torch.flip(lq, (-2, -1)))  # wh
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            gt_path = meta[0]['gt_path'][0]
            folder_name = meta[0]['key'].split('/')[0]
            frame_name = osp.splitext(osp.basename(gt_path))[0]
            center_frame_idx = int(frame_name)
            # if center_frame_idx % 10 == 0:
            if True:
                if isinstance(iteration, numbers.Number):
                    save_path = osp.join(
                        save_path, folder_name,
                        f'{frame_name}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path = osp.join(save_path, folder_name,
                                         f'{frame_name}.png')

                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                output = tensor2img(output)

                # print(np.shape(output))
                # output_crop = output[:536, :, :]
                output_crop = output[:-8, :, :]

                # print(np.shape(output_crop))
                mmcv.imwrite(output_crop, save_path)

        return results
