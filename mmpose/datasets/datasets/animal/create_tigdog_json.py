from __future__ import print_function, absolute_import

import random
import torch.utils.data as td

from scipy.io import loadmat
# import argparse
# import glob
# import imageio
# import cv2
from os.path import isfile
import numpy as np
import torch
import os

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *

class Real_Animal_All(td.Dataset):
    def __init__(self, is_train=True, is_aug=False, **kwargs):
        print()
        print("==> real_animal_all")
        self.img_folder = kwargs['image_path']  # root image folders
        self.is_train = is_train  # training set or test set
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.animal = ['horse', 'tiger'] if kwargs['animal'] == 'all' else [kwargs['animal']] # train on single or all animal categories
        self.train_on_all_cat = kwargs['train_on_all_cat']  # train on single or mul, decide mean file to load
        self.is_aug = is_aug

        # create train/val split
        self.train_img_set = []
        self.valid_img_set = []
        self.train_pts_set = []
        self.valid_pts_set = []
        self.train_cat_ids = []
        self.valid_cat_ids = []
        self.load_animal()
        # self.mean, self.std = self._compute_mean()


    def load_animal(self):
        animal_to_categ = {'horse': 0, 'tiger': 1}
        # generate train/val data
        for animal in sorted(self.animal):
            img_list = []  # img_list contains all image paths
            anno_list = []  # anno_list contains all anno lists
            range_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0/ranges', animal, 'ranges.mat')
            landmark_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0/landmarks', animal)
            range_file = loadmat(range_path)
            frame_num = 0

            train_idxs = np.load(os.path.join(self.img_folder, 'behaviorDiscovery2.0/real_animal/' + animal + '/train_idxs_by_video.npy'))
            valid_idxs = np.load(os.path.join(self.img_folder, 'behaviorDiscovery2.0/real_animal/' + animal + '/valid_idxs_by_video.npy'))
            for video in range_file['ranges']:
                # range_file['ranges'] is a numpy array [Nx3]: shot_id, start_frame, end_frame
                shot_id = video[0]
                landmark_path_video = os.path.join(landmark_path, str(shot_id) + '.mat')

                if not os.path.isfile(landmark_path_video):
                    continue
                landmark_file = loadmat(landmark_path_video)

                for frame in range(video[1], video[2] + 1):  # ??? video[2]+1
                    frame_id = frame - video[1]
                    img_name = animal + '/' + '0' * (8 - len(str(frame))) + str(frame) + '.jpg'
                    img_list.append([img_name, shot_id, frame_id])

                    coord = landmark_file['landmarks'][frame_id][0][0][0][0]
                    vis = landmark_file['landmarks'][frame_id][0][0][0][1]
                    landmark = np.hstack((coord, vis))
                    landmark_18 = landmark[:18, :]
                    if animal == 'horse':
                        anno_list.append(landmark_18)
                    elif animal == 'tiger':
                        landmark_18 = landmark_18[
                            np.array([1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 13, 14, 9, 10, 11, 12]) - 1]
                        anno_list.append(landmark_18)
                    frame_num += 1

            for idx in range(train_idxs.shape[0]):
                train_idx = train_idxs[idx]
                self.train_img_set.append(img_list[train_idx])
                self.train_pts_set.append(anno_list[train_idx])
                self.train_cat_ids.append(animal_to_categ[animal])
            for idx in range(valid_idxs.shape[0]):
                valid_idx = valid_idxs[idx]
                self.valid_img_set.append(img_list[valid_idx])
                self.valid_pts_set.append(anno_list[valid_idx])
                self.valid_cat_ids.append(animal_to_categ[animal])
            print('Animal:{}, number of frames:{}, train: {}, valid: {}'.format(animal, frame_num,
                                                                         train_idxs.shape[0], valid_idxs.shape[0]))
        print('Total number of frames:{}, train: {}, valid {}'.format(len(img_list), len(self.train_img_set),
                                                                      len(self.valid_img_set)))

    def _compute_mean(self):
        animal = 'all' if self.train_on_all_cat else self.animal[0]  # which mean file to load
        meanstd_file = './data/synthetic_animal/' + animal + '_combineds5r5_texture' + '/mean.pth.tar'

        if isfile(meanstd_file):
            print('load from mean file:', meanstd_file)
            meanstd = torch.load(meanstd_file)
        else:
            print("generate mean file")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train_list:
                a = self.img_list[index][0]
                img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', a)
                img = load_image_ori(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train_list)
            std /= len(self.train_list)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        print('  Real animal  mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('  Real animal  std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        img_list = self.train_img_set if self.is_train else self.valid_img_set
        anno_list = self.train_pts_set if self.is_train else self.valid_pts_set
        try:
            a = img_list[index][0]
        except IndexError:
            print(index)

        img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', a)
        img = load_image_ori(img_path)  # CxHxW
        pts = anno_list[index].astype(np.float32)
        x_vis = pts[:, 0][pts[:, 0] > 0]
        y_vis = pts[:, 1][pts[:, 1] > 0]

        try:
            # generate bounding box using keypoints
            height, width = img.size()[1], img.size()[2]
            y_min = float(max(np.min(y_vis) - 15, 0.0))
            y_max = float(min(np.max(y_vis) + 15, height))
            x_min = float(max(np.min(x_vis) - 15, 0.0))
            x_max = float(min(np.max(x_vis) + 15, width))
        except ValueError:
            print(img_path, index)
        # Generate center and scale for image cropping,
        # adapted from human pose https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/dataset/mpii.lua
        c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
        s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

        # For single-animal pose estimation with a centered/scaled figure
        nparts = pts.shape[0]
        pts = torch.Tensor(pts)
        r = 0

        # Prepare image and groundtruth map
        inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r)
        # inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        tpts_inpres = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                tpts_inpres[i, 0:2] = to_torch(transform(tpts_inpres[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], rot=r))
                target[i], vis = draw_labelmap_ori(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'target_weight': target_weight, 'pts_256': tpts_inpres}
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_img_set)
        else:
            return len(self.valid_img_set)


def real_animal_all(**kwargs):
    return Real_Animal_All(**kwargs)


real_animal_all.njoints = 18  # ugly but works


def get_bbox_from_pts(pts, height, width) -> list:
    x_vis = pts[:, 0][pts[:, 0] > 0]
    y_vis = pts[:, 1][pts[:, 1] > 0]

    try:
        # generate bounding box using keypoints
        # height, width = img.size()[1], img.size()[2]
        y_min = int(max(np.min(y_vis) - 15, 0.0))
        y_max = int(min(np.max(y_vis) + 15, height))
        x_min = int(max(np.min(x_vis) - 15, 0.0))
        x_max = int(min(np.max(x_vis) + 15, width))
    except ValueError:
        print(img_path)

    return [x_min, y_min, x_max, y_max]


if __name__ == '__main__':
    ds_root = '/home/browatbn/dev/datasets/animal_data'
    ds = Real_Animal_All(animal='all',
                         image_path=ds_root,
                         inp_res=256,
                         out_res=64,
                         scale_factor=0.25,
                         rot_factor=30,
                         sigma=1,
                         percentage=0,
                         stage=1,
                         train_on_all_cat=True,
                         label_type=''
                         )
    ann = {
        'images': [],
        'annotations': [],
        'categories': [],
        'info': {}
    }

    category_info_horse = {'supercategory': 'animal', 'id': 0, 'name': 'horse', 'keypoints': [], 'skeleton': [[]],}
    category_info_tiger = {'supercategory': 'animal', 'id': 1, 'name': 'tiger', 'keypoints': [], 'skeleton': [[]],}
    ann['categories'] = [category_info_horse, category_info_tiger]

    import json
    annotation_file = '/home/browatbn/dev/datasets/animal_data/ap-10k/annotations/ap10k-val-split1.json'

    # annotation_file = '/home/browatbn/dev/datasets/animal_data/animalpose_keypoint_new/keypoints.json'
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)

    image_root = os.path.join(ds_root, 'behaviorDiscovery2.0')

    num = 100
    for img_id, (image, pts, category_id) in enumerate(zip(ds.valid_img_set[:num],
                                                           ds.valid_pts_set[:num],
                                                           ds.valid_cat_ids[:num])):
        img_path = image[0]

        _img = cv2.imread(os.path.join(image_root, img_path))
        height, width = _img.shape[:2]

        ann['images'].append({
            'id': img_id,
            'file_name': img_path,
            'height': height,
            'width': width,
        })

        bbox = get_bbox_from_pts(pts.astype(np.float32), height, width)
        annot = {
            'id': img_id,
            'image_id': img_id,
            'bbox': bbox,
            'keypoints': pts.tolist(),
            'num_keypoints': real_animal_all.njoints,
            'category_id': category_id,
        }
        ann['annotations'].append(annot)

    outfile = os.path.join(image_root, 'valid.json')
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(ann, f, ensure_ascii=False, indent=4)


    exit()

    dl = td.DataLoader(ds, batch_size=5, shuffle=False, num_workers=0)

    import matplotlib.pyplot as plt
    for data in dl:
        images, heatmaps, metadata = data
        # plt.imshow(images[0, 0].numpy())
        # plt.show()
        # plt.imshow(heatmaps[0, 0].numpy())
        # plt.show()
        pts = metadata['pts']
        print("foo")



