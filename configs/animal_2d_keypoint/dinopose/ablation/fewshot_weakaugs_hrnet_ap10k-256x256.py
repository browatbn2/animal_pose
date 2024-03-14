_base_ = ['../../../_base_/default_runtime.py']

randomness=dict(seed=0)

# runtime
train_cfg = dict(max_epochs=150, val_interval=5)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=2e-5,
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
    # checkpoint=dict(type='CheckpointHook', interval=5),
)

# codec settings
codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

embedding_dim = 128
dino_dim = 1024

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
        out_channels=dino_dim,
        deconv_out_channels=None,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
    init_cfg=dict(type='Pretrained', checkpoint='/home/browatbn/dev/csl/animal_pose/work_dirs/weakaugs_distill_hrnet_ap10k-256x256/epoch_100.pth'),
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

indices_fs20 = [22669, 22705, 23420, 22661, 23417, 22738, 22687, 22937, 22837, 22702, 23418, 22667, 22696, 23425, 22746,
                22793, 23203, 23292, 22724, 22689, 27936, 29387, 29851, 29017, 27070, 26837, 26803, 30306, 29684, 26970,
                28936, 30028, 29079, 29491, 29773, 28234, 26748, 29227, 26848, 26603, 30740, 30996, 30773, 30629, 31843,
                31677, 30986, 30651, 30507, 30981, 30595, 31666, 30995, 30817, 31266, 30529, 30984, 30551, 31432, 31288,
                44294, 44318, 44811, 44577, 44466, 44600, 44248, 44332, 44319, 44280, 44689, 44357, 44331, 44897, 44314,
                44411, 44589, 44268, 44922, 44389, 54997, 54817, 54882, 54875, 54969, 54928, 54818, 54938, 54836, 54831,
                54839, 54852, 54899, 55008, 54906, 54845, 54944, 54992, 54995, 54854, 48872, 48753, 48789, 48873, 48752,
                48763, 48844, 48838, 48807, 48788, 48834, 48862, 48882, 48908, 48846, 48756, 48888, 48856, 48779, 48854,
                49031, 49074, 49094, 49002, 48946, 48973, 49036, 48913, 49037, 49084, 48933, 49041, 49061, 48988, 48999,
                48971, 48978, 49004, 49001, 48995, 55627, 55648, 55651, 55631, 55659, 55663, 55664, 55629, 55634, 55657,
                55640, 55643, 55636, 55625, 55626, 55653, 55660, 55641, 55624, 55635, 55825, 55743, 55748, 55803, 55674,
                55820, 55793, 55783, 55767, 55735, 55828, 55821, 55868, 55764, 55859, 55876, 55694, 55889, 55850, 55686,
                56703, 56617, 56680, 56730, 56731, 56574, 56752, 56726, 56567, 56651, 56606, 56628, 56721, 56750, 56577,
                56684, 56657, 56693, 56747, 56718, 58541, 58623, 58459, 58567, 58624, 58487, 58511, 58527, 58612, 58432,
                58454, 58515, 58435, 58577, 58467, 58585, 58461, 58445, 58458, 58569, 45098, 45095, 45060, 45069, 45083,
                45037, 45000, 44952, 44933, 45066, 45070, 44945, 45005, 45092, 44965, 45077, 45076, 45013, 44934, 45067,
                46176, 46197, 46196, 46129, 46155, 46111, 46231, 46148, 46229, 46102, 46187, 46290, 46103, 46284, 46240,
                46154, 46281, 46132, 46134, 46121, 20863, 20879, 20853, 20866, 20761, 20805, 20877, 20769, 20721, 20704,
                20833, 20749, 20695, 20705, 20880, 20733, 20789, 20860, 20796, 20689, 22131, 22418, 22631, 22076, 22075,
                22008, 22341, 22625, 22081, 22263, 22028, 21999, 22048, 22496, 22019, 22092, 22363, 22047, 22452, 22073,
                32047, 32017, 32039, 31952, 32035, 31977, 32058, 32054, 31966, 31936, 31917, 31929, 31892, 32064, 31963,
                31906, 31974, 31938, 32052, 31954, 40803, 32706, 36283, 40386, 33006, 35295, 40831, 40414, 40778, 40408,
                33295, 33472, 40147, 40380, 34694, 35095, 39947, 32595, 35850, 40821, 36497, 36491, 36387, 36338, 36482,
                36458, 36432, 36344, 36321, 36477, 36505, 36493, 36320, 36401, 36445, 36466, 36471, 36413, 36500, 36352,
                37532, 37699, 37696, 37682, 37851, 37653, 37787, 37816, 37598, 37584, 37526, 37645, 37654, 37684, 37548,
                37821, 37859, 37708, 37599, 37591, 37866, 37891, 37895, 37878, 37874, 37876, 37885, 37894, 37888, 37881,
                37892, 37877, 37886, 37871, 37872, 37883, 37893, 37889, 37882, 37870, 38564, 37910, 37918, 38275, 38037,
                38108, 38519, 38041, 38047, 37909, 38080, 38709, 37997, 37993, 38035, 38708, 38186, 37968, 37966, 38586,
                38779, 38884, 38907, 38782, 38803, 38854, 38826, 38831, 38776, 38732, 38811, 38840, 38808, 38737, 38823,
                38748, 38730, 38836, 38768, 38859, 39789, 39820, 39790, 39871, 39855, 39857, 39897, 39878, 39862, 39751,
                39792, 39763, 39736, 39852, 39920, 39750, 39772, 39816, 39899, 39886, 40856, 40868, 40906, 40921, 40884,
                40927, 40881, 40911, 40875, 40896, 40923, 40893, 40870, 40843, 40886, 40908, 40910, 40882, 40902, 40897,
                40969, 41020, 41116, 41122, 41101, 41039, 41070, 40980, 41143, 41148, 41044, 41066, 41009, 41023, 41003,
                41112, 41000, 41120, 41138, 40983, 20075, 20021, 20066, 20036, 20010, 20057, 20047, 20043, 20076, 20059,
                20077, 20038, 20018, 20073, 20013, 20039, 20051, 20071, 20069, 20055, 20143, 20164, 20129, 20177, 20214,
                20218, 20179, 20185, 20189, 20162, 20130, 20087, 20096, 20158, 20114, 20209, 20196, 20119, 20081, 20091,
                20252, 20296, 20288, 20303, 20285, 20262, 20336, 20250, 20329, 20330, 20328, 20249, 20304, 20276, 20332,
                20275, 20323, 20281, 20325, 20270, 20399, 20369, 20356, 20566, 20351, 20481, 20521, 20482, 20402, 20379,
                20425, 20434, 20533, 20551, 20365, 20426, 20380, 20524, 20386, 20530, 20645, 20676, 20669, 20636, 20607,
                20609, 20610, 20643, 20671, 20594, 20632, 20684, 20679, 20673, 20634, 20602, 20589, 20604, 20613, 20620,
                51859, 51872, 51811, 51863, 51825, 51848, 51812, 51845, 51837, 51846, 51810, 51877, 51834, 51849, 51835,
                51818, 51838, 51867, 51860, 51802, 51974, 51923, 52119, 52058, 51920, 51953, 51890, 52007, 52008, 51915,
                52073, 51949, 52142, 52140, 52003, 52095, 52156, 52086, 51958, 51985, 49226, 49299, 49266, 49293, 49317,
                49276, 49346, 49287, 49421, 49262, 49259, 49417, 49224, 49408, 49339, 49379, 49340, 49285, 49357, 49312,
                50022, 50094, 50017, 50207, 50106, 50108, 50170, 50141, 50183, 50088, 50156, 50125, 50213, 49991, 50223,
                49982, 50167, 50027, 50204, 50233, 43194, 43113, 43121, 43270, 43182, 43149, 43256, 43210, 43251, 43141,
                43108, 43153, 43183, 43101, 43091, 43250, 43087, 43231, 43191, 43096, 23720, 23619, 23519, 23453, 23660,
                23515, 23476, 23694, 23710, 23465, 23707, 23675, 23454, 23581, 23528, 23491, 23505, 23625, 23520, 23628,
                315, 116, 181, 482, 859, 123, 604, 107, 715, 191, 225, 37, 203, 173, 359, 992, 92, 150, 2, 126, 1050,
                1040, 1101, 1038, 1107, 1082, 1044, 1108, 1035, 1028, 1043, 1027, 1102, 1072, 1037, 1069, 1103, 1099,
                1109, 1039, 1150, 1178, 1130, 1175, 1280, 1524, 1558, 1145, 1591, 1224, 1978, 1336, 1146, 1176, 1149,
                1132, 1358, 1302, 1977, 1324, 1983, 2124, 2016, 2138, 2069, 2003, 2023, 2128, 1988, 2088, 2079, 2051,
                2118, 2146, 2154, 2108, 2112, 2005, 2047, 2037, 4422, 4665, 4238, 5022, 3346, 2776, 5144, 3109, 2677,
                4237, 2399, 5267, 5211, 2332, 5444, 5644, 5219, 5344, 5655, 5355, 8445, 6842, 8036, 5989, 7187, 7763,
                5856, 8000, 7773, 6979, 8135, 7557, 7335, 7052, 7274, 8159, 7409, 8210, 7434, 7213, 50272, 50297, 50294,
                50430, 50443, 50368, 50320, 50300, 50410, 50364, 50398, 50324, 50317, 50283, 50428, 50421, 50344, 50388,
                50330, 50446, 48548, 48612, 48673, 48714, 48542, 48560, 48700, 48706, 48625, 48584, 48569, 48707, 48695,
                48688, 48607, 48715, 48724, 48565, 48547, 48551, 47490, 47562, 47463, 47553, 47519, 47635, 47545, 47548,
                47627, 47639, 47563, 47580, 47482, 47595, 47498, 47646, 47484, 47640, 47524, 47557, 10780, 17057, 13374,
                18915, 9289, 10709, 18318, 8861, 10255, 10384, 8877, 18341, 8878, 10476, 9364, 18345, 9374, 9359, 9356,
                8828, 17743, 18090, 18285, 17665, 17715, 17747, 18145, 17661, 17658, 17979, 17709, 18288, 18272, 17698,
                17734, 18089, 18279, 17720, 17751, 17704, 19530, 19822, 19297, 19377, 19307, 19271, 19831, 19312, 19357,
                19552, 19315, 19339, 19374, 19287, 19816, 19802, 19696, 19269, 19299, 19817, 19917, 19999, 19959, 19935,
                20006, 19888, 19984, 19866, 19982, 19847, 19929, 19867, 19912, 19947, 19854, 19955, 19833, 19919, 19952,
                19884, 50762, 51197, 50839, 51048, 51079, 51103, 50800, 51068, 50841, 51246, 51012, 51099, 50817, 50768,
                50767, 51135, 50809, 51007, 51034, 50779]


# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/ap10k-train-split1.json',
        data_prefix=dict(img='data/'),
        pipeline=train_pipeline,
        indices=indices_fs20
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
    num_workers=5,
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
