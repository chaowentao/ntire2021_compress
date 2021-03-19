exp_name = 'edvr_g8_600k_large_finetune_compress'

# model settings
model = dict(
    type='EDVR',
    generator=dict(
        type='EDVRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=128,  # large 128
        num_frames=5,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=40,  # 40
        center_frame_idx=2,
        hr_in=True,
        with_predeblur=True,
        with_tsa=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=2.0, reduction='sum'))
# model training and testing settings
train_cfg = dict(tsa_iter=0)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'SRREDSDataset'
val_dataset_type = 'SRREDSDataset'
train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(
        type='LoadImageFromFileListTest',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileListTest',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=512, random=False),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

data = dict(
    # train
    samples_per_gpu=2,
    workers_per_gpu=2,
    drop_last=True,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='./data/video_compress_track2/test_youtube/images/train',
            gt_folder=
            './data/video_compress_track2/test_youtube/images/train_raw',
            ann_file=
            './data/video_compress_track2/test_youtube/images/meta_info_Compress_min_extra_test.txt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=1,
            val_partition='val_1',
            test_mode=False)),
    # val
    val_samples_per_gpu=1,
    val_workers_per_gpu=1,
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/video_compress_track2/test_youtube/images/train',
        gt_folder='./data/video_compress_track2/test_youtube/images/train_raw',
        ann_file=
        './data/video_compress_track2/test_youtube/images/meta_info_Compress_min_extra_test.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=1,
        val_partition='test_extra_1',
        test_mode=True),
    test=dict(
        # type=val_dataset_type,
        # # lq_folder='./data/video_compress_track2/images/val',
        # # gt_folder='./data/video_compress_track2/images/val',
        # # ann_file='./data/video_compress_track2/meta_info_Compress_Val.txt',
        # lq_folder='./data/video_compress_track2/test_youtube/images/train',
        # gt_folder='./data/video_compress_track2/test_youtube/images/train_raw',
        # ann_file=
        # './data/video_compress_track2/test_youtube/images/meta_info_Compress_extra_test.txt',
        # num_input_frames=5,
        # pipeline=test_pipeline,
        # scale=1,
        # val_partition='test_extra',
        # test_mode=True),
        type=val_dataset_type,
        # lq_folder='./data/video_compress_track2/images/val',
        # gt_folder='./data/video_compress_track2/images/val',
        # ann_file='./data/video_compress_track2/meta_info_Compress_Val.txt',
        lq_folder='./data/video_compress_track2/images/test',
        gt_folder='./data/video_compress_track2/images/test',
        ann_file='./data/video_compress_track2/meta_info_Compress_min_Test.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=1,
        val_partition='test_1',
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 150000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[50000, 50000, 50000, 100000],
    restart_weights=[1, 0.5, 0.5, 0.5],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = './work_dirs/edvr_g8_600k_large_finetune_compress/iter_50000.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
