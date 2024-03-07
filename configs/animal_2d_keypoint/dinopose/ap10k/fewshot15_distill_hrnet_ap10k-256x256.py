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

indices_fs15 = [30106, 27825, 26981, 29546, 29189, 30261, 26814, 29374, 26976, 29962, 29485, 26792, 27865, 26881, 27025, 30418, 31044, 30884, 31666, 30806, 31355, 30954, 30980, 30978, 31199, 30957, 30975, 30999, 30940, 31832, 49302, 49257, 49308, 49236, 49317, 49399, 49274, 49285, 49405, 49315, 49363, 49337, 49283, 49322, 49323, 50037, 50190, 50108, 50132, 50218, 50049, 50214, 50026, 50135, 50018, 50162, 50034, 50136, 50238, 50003, 19969, 19989, 19886, 20002, 19937, 19838, 19832, 19897, 19917, 19870, 19913, 19848, 19983, 19925, 19939, 32046, 31906, 31924, 32044, 32020, 31913, 32050, 32052, 31950, 31956, 32078, 32035, 32033, 32063, 31985, 40246, 33006, 36206, 40783, 40405, 32973, 33505, 40825, 32739, 33739, 35517, 32895, 33283, 34727, 35272, 36331, 36338, 36351, 36485, 36454, 36422, 36340, 36419, 36393, 36373, 36387, 36500, 36408, 36434, 36409, 37658, 37859, 37626, 37861, 37606, 37642, 37733, 37532, 37559, 37785, 37764, 37843, 37679, 37792, 37648, 37888, 37867, 37876, 37895, 37878, 37881, 37866, 37871, 37877, 37885, 37893, 37889, 37886, 37894, 37891, 38697, 38497, 37993, 38710, 37997, 38047, 38021, 38003, 38018, 38330, 37974, 38564, 38019, 38231, 37957, 38869, 38859, 38785, 38773, 38726, 38821, 38879, 38850, 38762, 38775, 38882, 38862, 38783, 38797, 38736, 39808, 39817, 39919, 39923, 39856, 39811, 39870, 39840, 39746, 39892, 39841, 39778, 39770, 39757, 39804, 40905, 40906, 40849, 40845, 40901, 40869, 40880, 40913, 40930, 40870, 40872, 40860, 40923, 40846, 40868, 41015, 41040, 40971, 41009, 41135, 41034, 41049, 40974, 41067, 40941, 41139, 41024, 40947, 41057, 41012, 44904, 44689, 44910, 44286, 44777, 44411, 44339, 44265, 44357, 44364, 44300, 44285, 44711, 44566, 44257, 604, 238, 46, 135, 71, 243, 232, 185, 104, 105, 187, 207, 150, 165, 154, 1099, 1053, 1075, 1112, 1084, 1025, 1030, 1028, 1098, 1089, 1060, 1091, 1033, 1044, 1038, 1143, 1144, 1164, 1145, 1280, 1135, 1213, 1228, 1146, 1160, 1335, 1209, 1220, 1168, 1149, 2158, 2023, 1999, 2083, 2151, 2093, 2119, 2081, 2111, 2160, 2173, 2048, 2027, 2141, 2159, 4682, 4607, 5111, 3223, 5219, 3443, 5589, 4238, 5655, 2821, 4434, 4336, 2710, 2688, 5544, 8395, 7199, 7914, 7776, 7003, 7557, 8012, 8147, 7714, 6868, 7736, 7286, 8408, 7039, 6111, 48688, 48592, 48582, 48597, 48596, 48606, 48578, 48637, 48708, 48647, 48587, 48589, 48717, 48569, 48584, 20860, 20778, 20885, 20762, 20805, 20814, 20720, 20759, 20848, 20764, 20692, 20782, 20861, 20727, 20741, 22631, 22072, 22407, 22053, 22041, 22638, 22076, 22081, 22658, 22070, 22042, 22396, 22207, 22114, 22440, 50780, 50797, 50821, 50912, 50853, 50870, 51163, 51256, 50815, 51192, 50847, 51151, 51218, 50790, 51103, 23624, 23602, 23512, 23635, 23523, 23623, 23550, 23690, 23695, 23625, 23584, 23448, 23471, 23687, 23706, 45079, 45065, 45036, 45028, 45033, 45025, 45003, 44927, 45070, 45051, 45112, 44997, 45076, 45002, 44966, 46119, 46177, 46181, 46166, 46164, 46261, 46105, 46222, 46121, 46151, 46251, 46156, 46291, 46140, 46265, 47530, 47491, 47628, 47517, 47469, 47643, 47594, 47588, 47555, 47492, 47639, 47485, 47464, 47587, 47470, 54986, 54974, 54875, 54944, 54845, 54988, 54855, 54948, 54939, 54953, 54967, 55006, 54979, 54852, 54917, 10276, 17391, 10345, 14968, 10360, 9279, 8828, 9330, 11036, 8814, 18406, 8884, 10454, 18760, 18330, 17653, 17714, 17707, 17990, 17890, 18288, 17645, 17733, 18156, 18279, 17751, 18201, 17669, 17708, 17740, 19326, 19787, 19808, 19719, 19619, 19342, 19799, 19306, 19288, 19329, 19308, 19788, 19375, 19272, 19541, 22937, 22716, 22696, 22679, 22701, 23413, 22717, 23015, 22661, 22766, 23420, 22729, 22710, 22765, 23004, 43091, 43262, 43244, 43089, 43126, 43161, 43266, 43182, 43154, 43087, 43169, 43224, 43106, 43160, 43206, 50366, 50379, 50389, 50382, 50415, 50300, 50363, 50279, 50328, 50400, 50426, 50394, 50285, 50309, 50374, 20022, 20013, 20051, 20055, 20045, 20076, 20065, 20067, 20018, 20058, 20062, 20023, 20068, 20069, 20072, 20186, 20183, 20082, 20119, 20212, 20128, 20211, 20103, 20208, 20166, 20141, 20152, 20081, 20135, 20198, 20285, 20308, 20303, 20259, 20318, 20323, 20266, 20292, 20276, 20312, 20298, 20249, 20253, 20263, 20277, 20552, 20541, 20478, 20371, 20374, 20587, 20385, 20376, 20416, 20520, 20369, 20384, 20397, 20439, 20480, 20606, 20638, 20684, 20641, 20675, 20598, 20624, 20669, 20672, 20613, 20649, 20656, 20634, 20631, 20604, 48754, 48793, 48808, 48801, 48826, 48780, 48807, 48734, 48747, 48797, 48874, 48790, 48908, 48758, 48761, 49102, 48940, 48911, 49091, 49058, 48983, 48924, 49011, 49012, 49095, 49003, 49025, 48919, 49063, 49028, 51885, 51872, 51801, 51804, 51816, 51837, 51846, 51865, 51841, 51838, 51845, 51811, 51829, 51819, 51859, 51985, 51962, 52030, 52026, 51906, 52145, 51963, 52102, 51919, 51973, 51924, 52132, 52156, 51914, 52100, 55660, 55651, 55624, 55667, 55645, 55631, 55661, 55625, 55647, 55630, 55640, 55662, 55669, 55663, 55664, 55757, 55758, 55873, 55771, 55822, 55901, 55710, 55775, 55699, 55803, 55802, 55704, 55879, 55705, 55795, 56634, 56567, 56606, 56716, 56694, 56736, 56713, 56728, 56638, 56691, 56621, 56622, 56677, 56629, 56684, 58430, 58573, 58564, 58517, 58514, 58528, 58537, 58488, 58439, 58619, 58543, 58511, 58475, 58522, 58463]
indices_train = indices_fs15

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
