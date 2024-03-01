_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=200, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-5,
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

distill = False

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=40),
    checkpoint=dict(save_best='coco/AP', rule='greater'),
)

if distill:
    default_hooks.update(dict(load_dino=dict(type='LoadDinoHook')))

# codec settings
codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)
# codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(128, 128), sigma=4)

embedding_dim = 128
dino_channels = 384
# dino_channels = 1024

# model settings
model = dict(
    type='DinoPoseEstimator',
    distill=distill,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=embedding_dim,
        deconv_out_channels=None,
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=embedding_dim,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    student_neck=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=embedding_dim,
        deconv_out_channels=None,
    ),
    student_head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        # deconv_out_channels=(32,),
        # deconv_kernel_sizes=(4,),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    student_decoder=dict(
        type='HeatmapHead',
        in_channels=embedding_dim,
        out_channels=dino_channels,
        deconv_out_channels=None,
    ),
    # student_head_hr=dict(
    #     type='HeatmapHead',
    #     in_channels=embedding_dim,
    #     out_channels=32,
    #     deconv_out_channels=None,
    #     conv_out_channels=(128, 128, 128, 128),
    #     conv_kernel_sizes=(3, 3, 3, 3),
    # ),
    student_head_hr=dict(
        type='HRNet',
        in_channels=embedding_dim,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BASIC',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=1,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=1,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            # stage4=dict(
            #     num_modules=3,
            #     num_branches=4,
            #     block='BASIC',
            #     num_blocks=(4, 4, 4, 4),
            #     num_channels=(32, 64, 128, 256))
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
    # init_cfg=dict(type='Pretrained', checkpoint='/home/browatbn/dev/csl/animal_pose/work_dirs/distill_res50_ap10k-256x256/epoch_15.pth'),
    # init_cfg=dict(type='Pretrained', checkpoint='/home/browatbn/dev/csl/animal_pose/work_dirs/distill_hrnet_ap10k-256x256/epoch_25.pth'),
    init_cfg=dict(type='Pretrained', checkpoint='/home/browatbn/dev/csl/animal_pose/work_dirs/fewshot_distill_hrnet_ap10k-256x256/epoch_160.pth'),
)

# base dataset settings
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = '/home/browatbn/dev/datasets/animal_data/ap-10k/'

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

# if distill:
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


train_supercategory = 'Bovidae'
# test_supercategory = 'Bovidae'
test_supercategory = None
# test_category = 'chimpanzee'
test_category = 'gorilla'
# test_category = 'deer'
# test_category = None

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
        ann_file='annotations/ap10k-train-split1.json',
        data_prefix=dict(img='data/'),
        pipeline=train_pipeline,
        supercategory=train_supercategory
    ))
if distill:
    val_cfg = None
    test_cfg = None
else:
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
            ann_file='annotations/ap10k-val-split1.json',
            data_prefix=dict(img='data/'),
            test_mode=True,
            pipeline=val_pipeline,
            supercategory=test_supercategory,
            category = test_category
    ))
    test_dataloader = dict(
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_mode=data_mode,
            ann_file='annotations/ap10k-val-split1.json',
            data_prefix=dict(img='data/'),
            test_mode=True,
            pipeline=val_pipeline,
            supercategory=test_supercategory,
            category=test_category
        ))
    # evaluators
    val_evaluator = dict(type='CocoMetric',
                         # ann_file=data_root + 'annotations/ap10k-val-split1.json'
                         )
    test_evaluator = dict(type='CocoMetric',
                          # ann_file=data_root + 'annotations/ap10k-val-split1.json'
                          )
