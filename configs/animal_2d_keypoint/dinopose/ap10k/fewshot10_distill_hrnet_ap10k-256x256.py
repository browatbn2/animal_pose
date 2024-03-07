_base_ = ['../../../_base_/default_runtime.py']

randomness=dict(seed=0)

# runtime
train_cfg = dict(max_epochs=150, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=2e-5,
))

# learning policy
# param_scheduler = [
#     # dict(
#     #     type='LinearLR', begin=0, end=500, start_factor=0.001,
#     #     by_epoch=False),  # warm-up
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=150,
#         milestones=[100],
#         gamma=0.2,
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
dino_channels = 1024

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
        conv_out_channels=[64, 64],
        conv_kernel_sizes=[7, 7],
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    dino_decoder=dict(
        type='HeatmapHead',
        in_channels=embedding_dim,
        out_channels=dino_channels,
        deconv_out_channels=None,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
    init_cfg=dict(type='Pretrained', checkpoint='/home/browatbn/dev/csl/animal_pose/work_dirs/distill_hrnet_ap10k-256x256/epoch_200.pth'),
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

indices_fs10 = [31973, 32030, 32075, 32002, 31934, 31969, 31949, 32019, 32013, 32027, 40398, 34250, 34661, 33394, 40783, 40592, 32740, 40403, 40047, 35828, 36429, 36415, 36371, 36393, 36392, 36499, 36402, 36494, 36344, 36390, 37700, 37626, 37668, 37624, 37843, 37601, 37671, 37728, 37658, 37851, 37889, 37878, 37883, 37891, 37882, 37892, 37884, 37874, 37895, 37876, 38286, 38541, 37927, 37949, 38319, 37952, 37918, 38716, 38027, 38297, 38755, 38787, 38848, 38779, 38755, 38749, 38824, 38829, 38871, 38724, 39807, 39906, 39820, 39747, 39780, 39873, 39816, 39908, 39862, 39772, 40859, 40922, 40897, 40929, 40886, 40930, 40905, 40901, 40884, 40854, 40996, 41084, 41146, 41120, 40987, 41067, 41142, 41129, 41008, 41013, 44922, 44331, 44266, 44902, 44339, 44275, 44914, 44299, 44344, 44276, 20054, 20061, 20044, 20068, 20018, 20076, 20039, 20036, 20067, 20047, 20191, 20192, 20106, 20208, 20136, 20193, 20170, 20234, 20195, 20189, 20330, 20303, 20307, 20286, 20270, 20286, 20290, 20297, 20329, 20279, 20564, 20408, 20533, 20578, 20557, 20343, 20478, 20551, 20452, 20542, 20620, 20656, 20634, 20677, 20607, 20645, 20608, 20613, 20629, 20651, 44943, 44962, 45080, 45061, 44962, 45014, 45031, 44995, 45085, 45046, 46214, 46299, 46183, 46122, 46146, 46241, 46132, 46201, 46233, 46121, 49340, 49259, 49381, 49406, 49257, 49407, 49379, 49299, 49317, 49366, 50073, 50046, 50101, 50010, 50144, 50219, 49978, 50194, 50128, 50009, 233, 337, 107, 282, 198, 227, 119, 604, 125, 38, 1050, 1092, 1096, 1031, 1044, 1028, 1103, 1096, 1043, 1067, 1591, 1924, 1213, 1177, 1122, 1208, 1214, 1187, 1768, 1175, 2105, 1980, 2051, 2059, 2112, 2005, 2019, 2144, 2189, 2088, 4237, 5156, 5289, 2455, 4768, 3594, 3109, 5666, 2729, 2455, 7939, 8036, 8012, 7756, 7274, 7446, 7446, 7409, 6108, 7952, 48908, 48873, 48785, 48816, 48787, 48880, 48822, 48758, 48832, 48787, 49052, 49002, 49102, 48913, 48917, 48980, 49000, 48973, 48933, 49124, 51833, 51865, 51862, 51805, 51857, 51840, 51845, 51806, 51854, 51802, 51920, 52046, 52010, 52136, 51921, 52105, 51899, 51963, 52012, 52140, 22718, 22782, 22734, 23417, 22667, 22703, 22676, 22751, 23226, 23215, 55642, 55660, 55634, 55646, 55659, 55653, 55664, 55643, 55626, 55657, 55822, 55698, 55710, 55850, 55793, 55767, 55721, 55856, 55782, 55823, 56636, 56691, 56666, 56567, 56631, 56571, 56667, 56719, 56601, 56704, 58447, 58485, 58626, 58604, 58452, 58454, 58517, 58478, 58526, 58554, 51197, 50967, 50797, 51043, 51216, 51216, 50798, 50853, 50844, 50808, 43206, 43097, 43109, 43273, 43216, 43077, 43185, 43091, 43092, 43126, 20767, 20819, 20762, 20700, 20734, 20858, 20738, 20727, 20791, 20863, 22653, 22081, 22285, 22053, 22507, 22307, 22061, 22075, 22618, 22015, 10077, 12854, 10796, 8821, 12091, 8851, 18400, 9417, 17392, 18848, 17732, 17647, 18268, 17689, 17740, 17742, 18101, 17738, 17714, 18303, 19452, 19619, 19355, 19350, 19574, 19302, 19541, 19346, 19508, 19350, 54828, 54875, 55009, 54829, 54856, 54820, 54978, 54817, 54976, 54969, 19863, 19849, 19976, 19901, 19886, 19991, 19987, 19927, 19915, 19935, 50385, 50274, 50436, 50328, 50377, 50323, 50297, 50277, 50428, 50330, 47644, 47487, 47640, 47468, 47587, 47556, 47607, 47627, 47512, 47520, 23572, 23472, 23513, 23624, 23588, 23436, 23707, 23536, 23483, 23479, 29152, 28605, 29313, 29473, 29546, 30228, 29239, 26792, 29374, 26670, 31055, 31632, 30507, 31555, 31477, 30695, 31621, 31144, 30651, 30981, 48685, 48559, 48671, 48556, 48595, 48641, 48623, 48716, 48611, 48626]
indices_train = indices_fs10

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
