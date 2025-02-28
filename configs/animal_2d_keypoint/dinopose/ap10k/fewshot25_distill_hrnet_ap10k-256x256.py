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

indices_fs25 = [48752, 48839, 48787, 48876, 48866, 48850, 48859, 48768, 48725, 48848, 48738, 48790, 48864, 48817, 48728, 48852, 48827, 48904, 48753, 48746, 48796, 48808, 48907, 48782, 48751, 49094, 49026, 49000, 48950, 49031, 48951, 49084, 48912, 49071, 48944, 48966, 48971, 49041, 49085, 48949, 49099, 48984, 49025, 49001, 49012, 49051, 49028, 49075, 49055, 49082, 28605, 26976, 29435, 29350, 30228, 29940, 29485, 29380, 28269, 30273, 27713, 30006, 29326, 29873, 29067, 26970, 27099, 29116, 30350, 29091, 29152, 27269, 28943, 29510, 28852, 30999, 30740, 31066, 30595, 30977, 30971, 31643, 31554, 30806, 30970, 30984, 30651, 30507, 31566, 30817, 31443, 31088, 30983, 31133, 30990, 31577, 31233, 30992, 31199, 31555, 49245, 49312, 49309, 49406, 49265, 49287, 49269, 49226, 49323, 49317, 49303, 49276, 49346, 49264, 49291, 49349, 49296, 49410, 49236, 49350, 49263, 49360, 49369, 49241, 49290, 49984, 50041, 50193, 50104, 50220, 50053, 50095, 50208, 50055, 50214, 49978, 50052, 50235, 50105, 50004, 50238, 49993, 50191, 50213, 50036, 50231, 50145, 50022, 49983, 50147, 200, 149, 103, 135, 237, 180, 108, 155, 58, 217, 282, 143, 234, 107, 226, 221, 96, 172, 260, 240, 125, 106, 715, 90, 84, 1079, 1038, 1075, 1092, 1083, 1050, 1072, 1060, 1073, 1091, 1030, 1061, 1098, 1103, 1028, 1108, 1036, 1040, 1025, 1037, 1109, 1064, 1031, 1058, 1049, 1513, 1218, 1935, 1680, 1193, 1977, 1128, 1122, 1197, 1217, 1813, 1182, 1144, 1132, 1236, 1117, 1702, 1126, 1175, 1143, 1114, 1222, 1967, 1224, 1172, 2156, 2109, 2106, 2158, 2151, 2104, 2111, 2015, 2173, 1988, 1983, 2174, 2138, 2069, 2099, 2072, 2160, 2095, 2112, 2123, 2146, 2088, 2075, 2150, 2159, 4521, 4331, 3346, 4238, 5033, 4657, 4086, 4781, 4768, 4422, 2821, 3717, 2688, 5433, 4879, 2455, 5096, 4706, 4109, 2610, 4373, 5444, 4853, 4817, 5011, 7102, 7249, 7753, 7745, 7773, 7286, 6966, 7409, 7274, 7213, 8050, 7446, 8063, 7964, 7336, 8469, 7728, 7557, 7734, 5822, 7763, 6111, 5862, 5989, 8246, 43256, 43101, 43109, 43203, 43179, 43211, 43126, 43193, 43189, 43250, 43087, 43105, 43107, 43113, 43093, 43174, 43213, 43095, 43226, 43188, 43244, 43210, 43092, 43106, 43231, 45128, 45039, 45087, 45098, 45001, 44976, 44943, 44926, 44969, 44960, 44953, 45082, 44962, 45015, 44998, 44935, 44930, 45014, 45086, 45116, 45020, 45013, 45113, 44986, 44981, 46201, 46273, 46157, 46215, 46251, 46286, 46271, 46165, 46183, 46193, 46168, 46256, 46233, 46176, 46238, 46209, 46228, 46290, 46178, 46268, 46284, 46265, 46109, 46149, 46151, 20054, 20033, 20072, 20075, 20040, 20038, 20066, 20020, 20023, 20049, 20055, 20044, 20078, 20043, 20017, 20067, 20039, 20057, 20032, 20051, 20076, 20026, 20010, 20013, 20061, 20165, 20191, 20234, 20103, 20155, 20100, 20180, 20184, 20122, 20172, 20127, 20235, 20158, 20242, 20213, 20230, 20237, 20173, 20104, 20194, 20124, 20116, 20176, 20139, 20121, 20329, 20280, 20250, 20252, 20328, 20278, 20245, 20336, 20277, 20246, 20270, 20279, 20332, 20275, 20330, 20265, 20290, 20253, 20272, 20305, 20298, 20311, 20249, 20304, 20292, 20399, 20455, 20354, 20376, 20568, 20338, 20368, 20427, 20452, 20496, 20535, 20385, 20493, 20403, 20437, 20475, 20566, 20581, 20543, 20339, 20557, 20538, 20411, 20480, 20422, 20682, 20639, 20665, 20610, 20592, 20670, 20655, 20646, 20601, 20632, 20645, 20607, 20634, 20609, 20661, 20642, 20684, 20683, 20630, 20672, 20643, 20599, 20635, 20602, 20620, 48578, 48706, 48607, 48710, 48546, 48649, 48692, 48713, 48696, 48708, 48639, 48567, 48587, 48542, 48584, 48678, 48637, 48707, 48571, 48724, 48547, 48716, 48570, 48588, 48615, 31989, 32064, 31957, 32017, 32013, 32054, 31950, 31930, 32065, 32072, 31904, 31919, 31969, 31952, 31923, 32047, 32073, 32005, 32029, 31954, 32002, 31892, 31932, 32011, 32071, 35695, 34250, 40406, 32828, 40436, 40808, 35616, 40782, 40047, 40658, 39933, 40778, 40269, 34461, 40492, 40428, 40470, 36017, 40413, 34984, 40812, 35550, 33306, 34906, 36239, 36434, 36338, 36481, 36482, 36367, 36319, 36394, 36428, 36397, 36365, 36477, 36420, 36511, 36342, 36361, 36493, 36487, 36470, 36321, 36341, 36333, 36307, 36362, 36404, 36349, 37590, 37591, 37728, 37559, 37826, 37679, 37713, 37833, 37584, 37815, 37522, 37697, 37805, 37626, 37534, 37601, 37618, 37526, 37588, 37698, 37736, 37685, 37785, 37681, 37733, 37888, 37889, 37891, 37892, 37893, 37894, 37895, 37866, 37867, 37870, 37871, 37872, 37874, 37876, 37877, 37878, 37881, 37882, 37883, 37884, 37885, 37886, 38497, 38286, 38041, 37993, 38674, 38353, 37979, 38675, 38031, 38486, 37909, 38035, 37953, 38036, 37901, 37968, 37910, 38408, 37948, 38704, 38032, 38017, 37997, 38047, 38023, 38752, 38832, 38785, 38739, 38855, 38840, 38748, 38914, 38728, 38836, 38737, 38819, 38721, 38863, 38792, 38811, 38860, 38825, 38852, 38738, 38900, 38843, 38731, 38745, 38751, 39908, 39779, 39863, 39746, 39768, 39892, 39857, 39889, 39862, 39923, 39836, 39803, 39770, 39852, 39781, 39813, 39871, 39841, 39802, 39920, 39751, 39918, 39780, 39745, 39915, 40870, 40894, 40934, 40912, 40846, 40868, 40849, 40904, 40909, 40928, 40921, 40930, 40848, 40887, 40897, 40929, 40860, 40916, 40886, 40882, 40893, 40875, 40880, 40850, 40925, 40945, 41080, 40961, 40964, 41120, 41131, 41039, 41051, 41138, 40966, 40952, 41102, 41003, 41054, 41045, 41078, 41011, 41104, 41116, 41041, 40969, 41067, 41097, 41114, 41020, 47652, 47533, 47541, 47673, 47507, 47486, 47638, 47558, 47627, 47456, 47595, 47510, 47498, 47502, 47599, 47484, 47572, 47611, 47602, 47505, 47590, 47589, 47639, 47647, 47493, 50272, 50350, 50326, 50258, 50279, 50309, 50427, 50378, 50292, 50259, 50348, 50355, 50385, 50308, 50341, 50382, 50334, 50373, 50325, 50278, 50409, 50417, 50262, 50426, 50359, 19982, 19976, 20004, 19921, 19858, 19953, 19893, 19926, 19947, 19901, 19984, 19999, 19985, 19952, 20007, 19937, 19881, 19865, 19835, 19912, 19973, 19948, 19890, 19916, 19918, 44248, 44356, 44302, 44294, 44345, 44331, 44280, 44357, 44722, 44511, 44600, 44317, 44589, 44298, 44244, 44308, 44343, 44909, 44285, 44711, 44292, 44500, 44262, 44273, 44655, 23699, 23645, 23483, 23579, 23710, 23624, 23724, 23523, 23610, 23732, 23690, 23628, 23703, 23534, 23435, 23706, 23566, 23568, 23720, 23684, 23604, 23455, 23603, 23485, 23707, 54982, 54901, 54913, 54906, 54851, 54828, 54829, 55000, 54951, 54987, 54947, 54949, 54955, 54918, 54856, 54935, 54817, 54892, 54816, 54838, 54831, 54917, 54985, 55004, 54952, 20762, 20795, 20724, 20756, 20790, 20721, 20699, 20850, 20860, 20742, 20686, 20792, 20863, 20700, 20738, 20703, 20789, 20879, 20866, 20770, 20869, 20874, 20858, 20778, 20716, 22642, 22058, 22330, 22649, 22646, 22082, 22640, 22274, 22016, 22067, 22015, 22068, 22076, 22022, 22627, 22196, 22285, 22218, 22085, 22152, 22207, 22429, 22363, 22653, 22115, 55645, 55653, 55631, 55655, 55664, 55669, 55627, 55657, 55630, 55647, 55652, 55659, 55661, 55634, 55624, 55626, 55667, 55642, 55663, 55625, 55635, 55648, 55665, 55658, 55641, 55884, 55892, 55782, 55899, 55893, 55709, 55764, 55699, 55901, 55859, 55783, 55704, 55873, 55862, 55748, 55705, 55755, 55674, 55767, 55820, 55902, 55839, 55692, 55750, 55896, 56696, 56745, 56707, 56557, 56580, 56656, 56627, 56655, 56714, 56608, 56693, 56670, 56748, 56692, 56625, 56675, 56642, 56597, 56561, 56702, 56687, 56673, 56605, 56728, 56571, 58535, 58492, 58528, 58558, 58450, 58592, 58463, 58569, 58470, 58476, 58523, 58546, 58480, 58447, 58506, 58497, 58475, 58474, 58575, 58510, 58520, 58618, 58504, 58478, 58577, 18324, 19159, 10785, 12869, 16735, 10364, 10359, 18412, 10347, 8883, 18406, 19026, 14079, 13409, 10498, 18959, 8857, 9014, 9248, 12721, 12862, 18393, 9409, 12717, 8830, 17709, 17658, 17682, 17680, 17646, 17755, 18167, 18201, 17720, 17655, 17690, 18156, 18303, 17713, 17707, 17879, 18045, 17747, 18001, 18280, 17714, 17708, 17660, 17669, 17665, 19792, 19350, 19816, 19552, 19652, 19441, 19826, 19789, 19810, 19306, 19797, 19267, 19812, 19619, 19279, 19260, 19258, 19269, 19290, 19363, 19719, 19831, 19300, 19264, 19302, 23292, 23419, 22774, 22730, 23409, 23004, 22691, 22993, 22715, 23303, 22717, 23337, 22751, 22702, 23215, 22704, 22662, 23370, 22837, 22735, 22772, 23429, 22737, 22709, 22763, 51809, 51876, 51853, 51829, 51870, 51807, 51812, 51878, 51863, 51814, 51805, 51825, 51854, 51822, 51813, 51835, 51818, 51808, 51827, 51881, 51856, 51802, 51880, 51819, 51850, 52086, 52068, 51983, 52130, 51982, 52084, 52059, 51903, 51899, 52150, 52058, 52038, 51939, 52096, 52103, 52137, 52032, 51896, 51993, 51925, 51953, 52007, 52034, 51980, 52071, 50813, 50815, 50825, 50857, 50957, 51068, 51128, 50790, 51020, 50858, 50820, 50954, 51112, 50929, 50870, 51078, 51246, 51024, 50913, 51228, 51192, 51179, 50967, 51086, 50760]
indices_train = indices_fs25

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
