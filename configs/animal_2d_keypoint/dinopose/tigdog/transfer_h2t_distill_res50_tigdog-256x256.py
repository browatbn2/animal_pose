_base_ = ['../../../_base_/default_runtime.py']

randomness=dict(seed=0)

# runtime
train_cfg = dict(max_epochs=200, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-4,
))

# learning policy
# param_scheduler = [
#     dict(
#         type='LinearLR', begin=0, end=500, start_factor=0.001,
#         by_epoch=False),  # warm-up
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=210,
#         milestones=[30, 60],
#         gamma=0.1,
#         by_epoch=True)
# ]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=40),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

# codec settings
codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

num_keypoints = 18
embedding_dim = 128
dino_dim = 768

# model settings
model = dict(
    type='mmpose.DinoPoseEstimator',
    distill=False,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=embedding_dim),
    # head_hr=dict(
    #     type='HRNet',
    #     in_channels=embedding_dim,
    #     extra=dict(
    #         stage1=dict(
    #             num_modules=1,
    #             num_branches=1,
    #             block='BASIC',
    #             num_blocks=(4,),
    #             num_channels=(64,)),
    #         stage2=dict(
    #             num_modules=1,
    #             num_branches=2,
    #             block='BASIC',
    #             num_blocks=(4, 4),
    #             num_channels=(32, 64)),
    #         stage3=dict(
    #             num_modules=1,
    #             num_branches=2,
    #             block='BASIC',
    #             num_blocks=(4, 4),
    #             num_channels=(32, 64)),
    #         stage4=dict(
    #             num_modules=1,
    #             num_branches=2,
    #             block='BASIC',
    #             num_blocks=(4, 4),
    #             num_channels=(32, 64)),
    #     ),
    # ),
    # head=dict(
    #     type='HeatmapHead',
    #     in_channels=32,
    #     out_channels=num_keypoints,
    #     deconv_out_channels=None,
    #     loss=dict(type='KeypointMSELoss', use_target_weight=True),
    #     decoder=codec),
    head=dict(
        type='HeatmapHead',
        in_channels=embedding_dim,
        out_channels=num_keypoints,
        deconv_out_channels=None,
        conv_out_channels=[128, 64],
        conv_kernel_sizes=[7, 7],
        # conv_out_channels=[64, 64, 64],
        # conv_kernel_sizes=[7, 7, 7],
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    dino_decoder=dict(
        type='HeatmapHead',
        in_channels=embedding_dim,
        out_channels=dino_dim,
        deconv_out_channels=None,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
    init_cfg=dict(type='Pretrained', checkpoint='/home/browatbn/dev/csl/animal_pose/work_dirs/distill_res50_tigdog-256x256/epoch_90.pth'),
)

# base dataset settings
dataset_type = 'TigDogDataset'
data_mode = 'topdown'
data_root = '/home/browatbn/dev/datasets/animal_data/behaviorDiscovery2.0/'

pixel_augmentations = dict(
    type='Albumentation',
    transforms=[
        dict(type='CoarseDropout',
             min_width=8, min_height=8,
             max_width=64, max_height=64,
             min_holes=1,
             max_holes=1,
             p=1.00),
        dict(type='RandomBrightnessContrast', brightness_limit=0.3, contrast_limit=0.3),
        dict(type='HueSaturationValue', hue_shift_limit=20*2, sat_shift_limit=30*2, val_shift_limit=20*2),
        dict(type='RGBShift'),
        dict(type='RandomGamma'),
        dict(type='Blur', blur_limit=11, p=0.1),
        dict(type='MotionBlur', blur_limit=11, p=0.1),
        dict(type='GaussNoise', p=0.1),
        dict(type='CLAHE', p=0.1),
    ])

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='TopdownAffineDino', input_size=codec['input_size'], input_size_dino=codec['heatmap_size']),
    dict(type='GenerateTarget', encoder=codec),
]

train_pipeline.append(pixel_augmentations)

train_pipeline.append(
    dict(type='PackPoseInputs',
         meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape', 'img_shape',
        'input_size', 'input_center', 'input_scale', 'flip', 'flip_direction', 'flip_indices',
        'raw_ann_info', 'dataset_name', 'dino_warp_mat', 'mask'),
         pack_transformed=True)
)

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='TopdownAffineDino', input_size=codec['input_size'], input_size_dino=codec['heatmap_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs',
         meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape', 'img_shape',
                    'input_size', 'input_center', 'input_scale', 'flip', 'flip_direction', 'flip_indices',
                    'raw_ann_info', 'dataset_name', 'dino_warp_mat', 'mask'),
         pack_transformed=True)
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='train_horse.json',
            data_prefix=dict(img='.'),
            pipeline=train_pipeline,
        )
    )


val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='valid_tiger.json',
        data_prefix=dict(img='.'),
        test_mode=True,
        pipeline=val_pipeline,
    )
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    # persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='valid_tiger.json',
        data_prefix=dict(img='.'),
        test_mode=True,
        pipeline=val_pipeline,
    )
)

# evaluators
evaluator = dict(type='PCKAccuracy', thr=0.05)
val_evaluator = evaluator
test_evaluator = evaluator
