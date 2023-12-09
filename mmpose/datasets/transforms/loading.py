# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import time
from mmcv.transforms import LoadImageFromFile

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results


import h5py
import os.path as osp
from mmcv.transforms.base import BaseTransform


def load_dino_hdf5(filepath: str) -> h5py.File:
    print(f"Loading DINO features from file {filepath}")
    return h5py.File(filepath, 'r')


@TRANSFORMS.register_module()
class LoadDino(BaseTransform):

    def __init__(self, dino_file="/home/browatbn/dev/data/dino/dino_train-split1_vits14_14.h5") -> None:
        self.dino_file = dino_file
        assert osp.isfile(self.dino_file), f"Could not find DINO feature file {self.dino_file}"
        self.attentions = load_dino_hdf5(self.dino_file)

    def _get_dino_features(self, idx: int) -> np.ndarray:
        return np.array(self.attentions[str(idx)])

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        t = time.time()
        a = self._get_dino_features(results['id'])
        if results.get('flip', False):
        #     # a = a[:, :, ::-1].copy()
            a = np.flip(a, axis=2).copy()
        results['attentions'] = a
        # print(f"t_loaddino={1000 * (time.time() - t)}ms")
        return results