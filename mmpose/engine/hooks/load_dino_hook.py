from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

from mmengine.dist import all_reduce_dict, get_dist_info
from mmengine.hooks import Hook
from torch import nn
import numpy as np
import time
import h5py
import os
import torch
import kornia

from mmpose.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


def get_dinov2_filepath(split):
    out_dir = '/home/browatbn/dev/data/dino'
    patch_size = 14
    arch = 'vits14'
    return os.path.join(out_dir, f'dino_{split}_{arch}_{patch_size}.h5')


def load_dino_hdf5(filepath: str) -> h5py.File:
    print(f"Loading DINO features from file {filepath}")
    return h5py.File(filepath, 'r')


@HOOKS.register_module()
class LoadDinoHook(Hook):
    """Load DINO features from disk."""

    def __init__(self):

        self.dino_hdf5 = None
        self.dino_features = {}

        if True:
            split = 'train-split1'
            dino_file = get_dinov2_filepath(split=split)
            if dino_file is not None:
                assert os.path.isfile(dino_file), f"Could not find DINO feature file {dino_file}"
                self.dino_hdf5 = load_dino_hdf5(dino_file)

    def _get_dino_features(self, idx: int) -> np.ndarray:
        key = str(idx)
        return np.array(self.dino_hdf5[key])

    def _load_dino_batch(self, data):
        # t = time.time()
        features = []
        for i, data_sample in enumerate(data['data_samples']):
            feat = self._get_dino_features(data_sample.id)
            if data_sample.flip:
                feat = feat[:, :, ::-1].copy()
            features.append(feat)

            # feat = torch.tensor(feat, device='cuda')
            # dino_warp_mat = torch.tensor(data_sample.dino_warp_mat, device='cuda')
            # feat = kornia.geometry.affine(feat, dino_warp_mat)

            # data_sample.set_data({'dino': feat})

        attentions = torch.tensor(np.array(features), device='cuda')
        dino_warp_mats = torch.tensor(np.array([s.dino_warp_mat for s in data['data_samples']]), device='cuda')
        attentions = kornia.geometry.affine(attentions, dino_warp_mats)
        for i, data_sample in enumerate(data['data_samples']):
            data_sample.set_data({'dino': attentions[i]})

        # print(f"t1={1000 * (time.time() - t):.0f}ms")
        return data

    def before_train_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None) -> None:
        self._load_dino_batch(data_batch)
