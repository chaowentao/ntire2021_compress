import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
from torch.nn.modules.utils import _pair

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  CAResidualBlockNoBN, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from abc import ABCMeta


class Base_NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 sperate=None,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 **kwargs):
        super(Base_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        self.sperate = sperate
        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.mode != 'gaussian':
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(
                self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU'))

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                normal_init(m.conv, std=std)
        else:
            normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm_cfg is None:
                constant_init(self.conv_out.conv, 0)
            else:
                constant_init(self.conv_out.norm, 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out.conv, std=std)
            else:
                normal_init(self.conv_out.norm, std=std)

    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        if self.sperate == 'hw':
            g_x = self.g(x).view(n, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
        if self.sperate == 'c':
            g_x = self.g(x).view(n, self.inter_channels, -1)
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
            phi_x = phi_x.permute(0, 2, 1)
        if self.sperate == 't':
            g_x = self.g(x).view(n, self.inter_channels, -1)
            g_x = g_x.permute(1, 0, 2)
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(1, 0, 2)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
            phi_x = phi_x.permute(1, 2, 0)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal2d_sp(Base_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    _abbr_ = 'nonlocal_block'

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 sperate='hw',
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super(NonLocal2d_sp, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample
        self.sperate = sperate
        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocalModule(nn.Module):

    def __init__(self, mid_channels=64, num_frames=5, center_frame_idx=2):
        super(NonLocalModule, self).__init__()

        self.inputChannelWH = NonLocal2d_sp(mid_channels, True, sperate='hw')
        self.inputChannelC = NonLocal2d_sp(mid_channels, False, sperate='c')
        self.inputChannelT = NonLocal2d_sp(mid_channels, False, sperate='t')

    def forward(self, aligned_feat):
        n, t, c, h, w = aligned_feat.size()
        nonlocal_input = aligned_feat.reshape(n * t, c, h, w)

        Ahw = self.inputChannelWH(nonlocal_input)
        Ac = self.inputChannelC(nonlocal_input)
        At = self.inputChannelT(nonlocal_input)

        Ahw = Ahw.reshape(n, t, c, h, w)
        Ac = Ac.reshape(n, t, c, h, w)
        At = At.reshape(n, t, c, h, w)

        aligned_feat = Ahw + Ac + At + aligned_feat
        return aligned_feat


class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super(ModulatedDCNPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVRNet_Test.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        deform_groups (int): Deformable groups. Defaults: 8.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 deform_groups=8,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvModule(
                mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
            if i == 3:
                self.offset_conv2[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            else:
                self.offset_conv2[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg)
                self.offset_conv3[level] = ConvModule(
                    mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            self.dcn_pack[level] = ModulatedDCNPack(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=deform_groups)

            if i < 3:
                act_cfg_ = act_cfg if i == 2 else None
                self.feat_conv[level] = ConvModule(
                    mid_channels * 2,
                    mid_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg_)

        # Cascading DCN
        self.cas_offset_conv1 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_offset_conv2 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(
            mid_channels,
            mid_channels,
            3,
            padding=1,
            deform_groups=deform_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neighbor_feats, ref_feats):
        """Forward function for PCDAlignment.

        Align neighboring frames to the reference frame in the feature level.

        Args:
            neighbor_feats (list[Tensor]): List of neighboring features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).
            ref_feats (list[Tensor]): List of reference features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # The number of pyramid levels is 3.
        assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
            'The length of neighbor_feats and ref_feats must be both 3, '
            f'but got {len(neighbor_feats)} and {len(ref_feats)}')

        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([neighbor_feats[i - 1], ref_feats[i - 1]],
                               dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](
                    torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)

            feat = self.dcn_pack[level](neighbor_feats[i - 1], offset)
            # changed
            if i == 3:
                feat = self.lrelu(feat)
            else:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))

            if i > 1:
                # upsample offset and features
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feats[0]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet_Test.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        num_frames (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self,
                 mid_channels=64,
                 num_frames=5,
                 center_frame_idx=2,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(
            num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(
            mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(
            mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(
            mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(
            mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(
            mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """Forward function for TSAFusion.

        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)  # (n, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (n, h, w)
            corr_l.append(corr.unsqueeze(1))  # (n, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (n, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)  # (n, t*c, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob

        # fusion
        feat = self.feat_fusion(aligned_feat)

        # spatial attention
        attn = self.spatial_attn1(aligned_feat)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(
            torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        mid_channels (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self,
                 num_in_ch=3,
                 mid_channels=64,
                 hr_in=False,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = ConvModule(
            num_in_ch, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = ConvModule(
                mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
            self.stride_conv_hr2 = ConvModule(
                mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)

        # generate feature pyramid
        self.stride_conv_l2 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.stride_conv_l3 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)

        self.resblock_l3 = ResidualBlockNoBN(mid_channels=mid_channels)
        self.resblock_l2_1 = ResidualBlockNoBN(mid_channels=mid_channels)
        self.resblock_l2_2 = ResidualBlockNoBN(mid_channels=mid_channels)
        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(mid_channels=mid_channels) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.conv_first(x)
        if self.hr_in:
            feat_l1 = self.stride_conv_hr1(feat_l1)
            feat_l1 = self.stride_conv_hr2(feat_l1)

        # generate feature pyramid
        feat_l2 = self.stride_conv_l2(feat_l1)
        feat_l3 = self.stride_conv_l3(feat_l2)

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


@BACKBONES.register_module()
class EDVRNet_Test(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether to use TSA module. Default: False.
        with_nonlocal (bool): Whether to use nonlocal module. Default: False.
        with_car (bool): Whether to use CAResidualBlockNoBN instead of ResidualBlockNoBN. Default: False. 
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=False,
                 with_nonlocal=False,
                 with_car=False):
        super(EDVRNet_Test, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa
        self.with_nonlocal = with_nonlocal
        self.with_car = with_car

        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(
                mid_channels=mid_channels, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(mid_channels, mid_channels, 1, 1)
        else:
            self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # extract pyramid features
        if self.with_car:
            self.feature_extraction = make_layer(
                CAResidualBlockNoBN,
                num_blocks_extraction,
                mid_channels=mid_channels)
        else:
            self.feature_extraction = make_layer(
                ResidualBlockNoBN,
                num_blocks_extraction,
                mid_channels=mid_channels)

        self.feat_l2_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        # pcd alignment
        self.pcd_alignment = PCDAlignment(
            mid_channels=mid_channels, deform_groups=deform_groups)
        # tsa fusion
        if self.with_tsa:
            self.fusion = TSAFusion(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1,
                                    1)
        # non local module
        if self.with_nonlocal:
            self.non_local = NonLocalModule(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)

        # reconstruction
        if self.with_car:
            self.reconstruction = make_layer(
                CAResidualBlockNoBN,
                num_blocks_reconstruction,
                mid_channels=mid_channels)
        else:
            self.reconstruction = make_layer(
                ResidualBlockNoBN,
                num_blocks_reconstruction,
                mid_channels=mid_channels)
        # upsample
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        # we fix the output channels in the last few layers to 64.
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """Forward function for EDVRNet_Test.

        Args:
            x (Tensor): Input tensor with shape (n, t, c, h, w).

        Returns:
            Tensor: SR center frame with shape (n, c, h, w).
        """
        # x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        return x_center

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            if self.with_tsa:
                for module in [
                        self.fusion.feat_fusion, self.fusion.spatial_attn1,
                        self.fusion.spatial_attn2, self.fusion.spatial_attn3,
                        self.fusion.spatial_attn4, self.fusion.spatial_attn_l1,
                        self.fusion.spatial_attn_l2,
                        self.fusion.spatial_attn_l3,
                        self.fusion.spatial_attn_add1
                ]:
                    kaiming_init(
                        module.conv,
                        a=0.1,
                        mode='fan_out',
                        nonlinearity='leaky_relu',
                        bias=0,
                        distribution='uniform')
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
