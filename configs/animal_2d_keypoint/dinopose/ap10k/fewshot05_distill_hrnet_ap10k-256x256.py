_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=1)

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

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=40),
    checkpoint=dict(save_best='coco/AP', rule='greater'),
)

# codec settings
codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

embedding_dim = 128
dino_channels = 384
# dino_channels = 1024

# model settings
model = dict(
    type='DinoPoseEstimator',
    distill=False,
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

indices_fs05 = [20026, 20057, 20077, 20067, 20045, 20104, 20155, 20204, 20211, 20198, 20319, 20318, 20244, 20281, 20280, 20493, 20338, 20380, 20450, 20343, 20608, 20622, 20643, 20679, 20624, 23092, 22937, 22848, 23159, 23359, 105, 359, 192, 228, 404, 1107, 1099, 1096, 1039, 1079, 1117, 1193, 1115, 1458, 1138, 2193, 1998, 2061, 2030, 2080, 5000, 4706, 5111, 4916, 2577, 7756, 7076, 7377, 7714, 8123, 48663, 48595, 48656, 48655, 48680, 19939, 19870, 19847, 19875, 19880, 49381, 49383, 49312, 49355, 49386, 50219, 50029, 50230, 50209, 50185, 51848, 51807, 51885, 51802, 51808, 51931, 51918, 52060, 52100, 51981, 44364, 44266, 44341, 44902, 44890, 54873, 54978, 54866, 54817, 54817, 50842, 50864, 50994, 51181, 50839, 48787, 48819, 48772, 48732, 48810, 49029, 48978, 48920, 48949, 48924, 20782, 20756, 20726, 20692, 20803, 22082, 22046, 22050, 22116, 22047, 50313, 50334, 50319, 50343, 50257, 23660, 23449, 23572, 23635, 23484, 55635, 55653, 55643, 55665, 55655, 55853, 55783, 55682, 55822, 55823, 56723, 56655, 56601, 56733, 56644, 58570, 58447, 58522, 58430, 58492, 27092, 27103, 29840, 26603, 26826, 31632, 31721, 31466, 31055, 31199, 45002, 45043, 45083, 45025, 45003, 46216, 46276, 46261, 46205, 46292, 43209, 43150, 43229, 43101, 43152, 32039, 31994, 32058, 31966, 31920, 40412, 40795, 33006, 32551, 40235, 36362, 36511, 36456, 36360, 36472, 37590, 37666, 37578, 37532, 37858, 37894, 37889, 37878, 37888, 37874, 37977, 37945, 38497, 38011, 37983, 38913, 38848, 38730, 38912, 38855, 39855, 39856, 39834, 39808, 39751, 40860, 40853, 40846, 40907, 40916, 41040, 41025, 40952, 41067, 40962, 47652, 47585, 47510, 47463, 47595, 11012, 9944, 10350, 14157, 10323, 18112, 17669, 17856, 17715, 18292, 19463, 19335, 19313, 19350, 19325]
indices_train = indices_fs05

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
        indices=indices_train,
    ))
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
    ))
# evaluators
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/ap10k-val-split1.json')
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/ap10k-val-split1.json')
