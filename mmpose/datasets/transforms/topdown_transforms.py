# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import kornia
import torch
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix

import matplotlib.pyplot as plt
from mmpose.structures.bbox import flip_bbox


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        results['mask'] = results['img'].sum(axis=2) > 0

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()
class TopdownAffineDino(TopdownAffine):
    def __init__(self,
                 input_size: Tuple[int, int],
                 input_size_dino: Tuple[int, int]) -> None:
        super().__init__(input_size, use_udp=False)
        self.input_size_dino = input_size_dino

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffineDino`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size_dino
        warp_size = (int(w), int(h))

        center = results['input_center']
        scale = results['input_scale']
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        _warp_mat = np.concatenate([warp_mat, [[0.0, 0.0, 1.0]]])

        # reshape bbox to fixed aspect ratio
        results['bbox_scale_orig'] = self._fix_aspect_ratio(
            results['bbox_scale_orig'], aspect_ratio=w / h)

        if results.get('flip', False):
            results['bbox_center_orig'] = flip_bbox(
                results['bbox_center_orig'],
                image_size=results['ori_shape'][::-1],
                bbox_format='center',
                direction=results['flip_direction'])

        warp_mat_orig = get_warp_matrix(results['bbox_center_orig'][0], results['bbox_scale_orig'][0],
                                        rot=0, output_size=self.input_size_dino)
        warp_mat_orig_inv = np.linalg.inv(np.concatenate([warp_mat_orig, [[0.0, 0.0, 1.0]]]))

        t = _warp_mat @ warp_mat_orig_inv

        results['dino_warp_mat'] = t[:2].astype(np.float32)
        # dino_wrp = kornia.geometry.affine(torch.tensor(results['attentions']), torch.tensor(t[:2]).float())

        # img_orig = cv2.imread(results['img_path'])
        # fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(results['img'])
        # # ax[1].imshow(att_wrp)
        # ax[2].imshow(img_orig)
        # ax[3].imshow(dino_wrp[0].numpy())
        # plt.show()

        # results['attentions'] = dino_wrp.numpy()

        return results

