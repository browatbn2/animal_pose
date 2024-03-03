dataset_info = dict(
    dataset_name='tigdog',
    paper_info=dict(
        author='todo',
        title='TigDog',
        container='todo',
        year='todo',
        homepage='todo',
    ),
    keypoint_info={
        0: dict(name='L_Eye', id=0, color=[0, 255, 0], type='upper', swap='R_Eye'),
        1: dict(name='R_Eye', id=1, color=[255, 128, 0], type='upper', swap='L_Eye'),
        2: dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),

        3: dict(name='L_F_Paw', id=3, color=[0, 255, 0], type='upper', swap='R_F_Paw'),
        4: dict(name='R_F_Paw', id=4, color=[0, 255, 0], type='upper', swap='L_F_Paw'),
        5: dict(name='L_B_Paw', id=5, color=[0, 255, 0], type='lower', swap='R_B_Paw'),
        6: dict(name='R_B_Paw', id=6, color=[0, 255, 0], type='lower', swap='L_B_Paw'),

        7: dict(name='Root of tail', id=4, color=[51, 153, 255], type='lower', swap=''),

        8: dict(name='L_Elbow', id=8, color=[51, 153, 255], type='upper', swap='R_Elbow'),
        9: dict(name='R_Elbow', id=9, color=[255, 128, 0], type='upper', swap='L_Elbow'),
        10: dict(name='L_Knee', id=10, color=[255, 128, 0], type='lower', swap='R_Knee'),
        11: dict(name='R_Knee', id=11, color=[0, 255, 0], type='lower', swap='L_Knee'),

        12: dict(name='L_Shoulder', id=5, color=[51, 153, 255], type='upper', swap='R_Shoulder'),
        13: dict(name='R_Shoulder', id=8, color=[0, 255, 0], type='upper', swap='L_Shoulder'),

        14: dict(name='L_F_Knee', id=14, color=[0, 255, 0], type='upper', swap='R_F_Knee'),
        15: dict(name='R_F_Knee', id=15, color=[0, 255, 0], type='upper', swap='L_F_Knee'),
        16: dict(name='L_B_Knee', id=16, color=[0, 255, 0], type='lower', swap='R_B_Knee'),
        17: dict(name='R_B_Knee', id=17, color=[0, 255, 0], type='lower', swap='L_B_Knee'),
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[0, 0, 255]),
        1: dict(link=('L_Eye', 'Nose'), id=1, color=[0, 0, 255]),
        2: dict(link=('R_Eye', 'Nose'), id=2, color=[0, 0, 255]),
        6: dict(link=('L_Shoulder', 'L_Elbow'), id=6, color=[0, 255, 255]),
        7: dict(link=('L_Elbow', 'L_F_Paw'), id=6, color=[0, 255, 255]),
        9: dict(link=('R_Shoulder', 'R_Elbow'), id=8, color=[6, 156, 250]),
        10: dict(link=('R_Elbow', 'R_F_Paw'), id=9, color=[6, 156, 250]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5, 1.0
    ],
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089, 0.089
    ])
