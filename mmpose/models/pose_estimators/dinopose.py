from itertools import zip_longest
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict
from mmengine.optim import OptimWrapper
import torch
import numpy as np
import kornia

import time
import cv2
import os
from torch import Tensor
import h5py

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
from mmpose.utils.tensor_utils import to_numpy
from mmpose.evaluation.functional import pose_pck_accuracy
from mmengine.structures import PixelData
from mmpose.models.utils.tta import flip_heatmaps

import cv2
from mmpose.visualization.vis import create_keypoint_result_figure, Visualization


def get_dinov2_filepath(split):
    out_dir = '/home/browatbn/dev/data/dino'
    patch_size = 14
    arch = 'vits14'
    # return os.path.join(out_dir, f'pca3_dino_{split}_{arch}_{patch_size}.h5')
    # return os.path.join(out_dir, f'pca9_dino_{split}_{arch}_{patch_size}.h5')
    # return os.path.join(out_dir, f'pca16_dino_{split}_{arch}_{patch_size}.h5')
    # return os.path.join(out_dir, f'pca32_dino_{split}_{arch}_{patch_size}.h5')
    return os.path.join(out_dir, f'dino_{split}_{arch}_{patch_size}.h5')


def load_dino_hdf5(filepath: str) -> h5py.File:
    print(f"Loading DINO features from file {filepath}")
    return h5py.File(filepath, 'r')


total_curr_iter = 0


@MODELS.register_module()
class DinoPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 dino_encoder: ConfigType = None,
                 dino_neck: OptConfigType = None,
                 dino_decoder: ConfigType = None,
                 student_backbone: ConfigType = None,
                 student_neck: OptConfigType = None,
                 student_head: OptConfigType = None,
                 student_head_hr: OptConfigType = None,
                 student_decoder: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 distill=False):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        if dino_encoder is not None:
            self.dino_encoder = MODELS.build(dino_encoder)
        if dino_neck is not None:
            self.dino_neck = MODELS.build(dino_neck)
        if dino_decoder is not None:
            self.dino_decoder = MODELS.build(dino_decoder)

        # self.student_backbone = MODELS.build(student_backbone)
        self.student_neck = MODELS.build(student_neck)
        self.student_head = MODELS.build(student_head)
        self.student_head_hr = MODELS.build(student_head_hr)
        self.student_decoder = MODELS.build(student_decoder)

        self.dino_hdf5 = None
        self.dino_features = {}

        self.batch_idx = 0
        self.vi = Visualization()

        self.distill = distill

        if True:
            split = 'train-split1'
            dino_file = get_dinov2_filepath(split=split)
            if dino_file is not None:
                assert os.path.isfile(dino_file), f"Could not find DINO feature file {dino_file}"
                self.dino_hdf5 = load_dino_hdf5(dino_file)
                # t = time.time()
                # print("loading dino....")
                # for i, idx in enumerate(self.attentions):
                #     if i % 200 == 0:
                #         print(i)
                #     try:
                #         a = np.array(self.attentions[str(idx)]).astype(np.float16)
                #         # attentions[i] = torch.tensor(a, device=device)
                #         self.dino_features[idx] = a
                #     except:
                #         pass
                # self.dino_features = attentions
                # print(f"t_loadalldino={1000 * (time.time() - t):.0f}ms")
                # np.save('test_attentions.npy', self.dino_features)
                # t = time.time()
                # self.dino_features = np.load('test_attentions.npy')
                # print(f"t_loadalldino_numpy={1000 * (time.time() - t):.0f}ms")



    def _get_dino_features(self, idx: int) -> np.ndarray:
        key = str(idx)
        # if key not in self.dino_features:
        #     self.dino_features[key] = np.array(self.dino_hdf5[key])
        # return self.dino_features[key]
        return np.array(self.dino_hdf5[key])

    def _load_dino_batch(self, data):
        t = time.time()
        # dino_dim = 384
        # features = torch.zeros((len(data['data_samples']), dino_dim, 64, 64), device=self.data_preprocessor.device)
        # ids = [s.id for s in data['data_samples']]
        # flips = [s.flip for s in data['data_samples']]
        # for i, idx in enumerate(ids):
        #     a = self._get_dino_features(idx)
        #     # flip = l1.flip[i] if l1.flip is not None else False
        #     if flips[i]:
        #         a = a[:, :, ::-1].copy()
        #     features[i] = torch.tensor(a, device=features.device)
        for i, data_sample in enumerate(data['data_samples']):
            feat = self._get_dino_features(data_sample.id)
            # flip = l1.flip[i] if l1.flip is not None else False
            if data_sample.flip:
                feat = feat[:, :, ::-1].copy()
            data_sample.set_data({'dino': feat})
        # data['dino'] = features
        print(f"t1={1000 * (time.time() - t):.0f}ms")
        return data

    def get_dino_inputs(self, data_samples) -> Tensor | None:
        if 'dino' not in data_samples[0]:
            return None
        return torch.stack([s.dino for s in data_samples])

    def extract_emb(self, inputs: Tensor) -> Tuple[Tensor]:
        # t = time.time()
        # attentions = torch.tensor(np.array([s.dino for s in data_samples]), device=self.data_preprocessor.device)
        # dino_warp_mats = torch.tensor(np.array([s.dino_warp_mat for s in data_samples]), device=self.data_preprocessor.device)
        # attentions = kornia.geometry.affine(attentions, dino_warp_mats)
        # print(f"t_dino_warp={1000 * (time.time() - t):.0f}ms")
        x = self.dino_encoder(inputs)
        x = self.dino_neck(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return (x, )

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.student_backbone(inputs)
        # x = self.neck(x)
        # x = torch.nn.functional.normalize(x, dim=1)
        return (x, )

    def extract_student(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.student_backbone(inputs)
        x = self.student_neck(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return (x, )

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """

        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            _data = self.data_preprocessor(data, True)
            losses, _ = self.forward(**_data, train=True)  # type: ignore
            # data['data_samples'] = results
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        _data = self.data_preprocessor(data, True)
        losses, results = self.forward(**_data, train=False)  # type: ignore
        # data['data_samples'] = results
        _, log_vars = self.parse_losses(losses)  # type: ignore
        return [results, log_vars]

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        _data = self.data_preprocessor(data, True)
        losses, results = self.forward(**_data, train=False)  # type: ignore
        # data['data_samples'] = results
        _, log_vars = self.parse_losses(losses)  # type: ignore
        return [results, log_vars]
        # data_ = self.data_preprocessor(data, False)
        # losses, results = self.predict(**_data, train=False)  # type: ignore
        # data['data_samples'] = results

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        pass

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        pass

    def forward_backbone(self, backbone, inputs, train):
        with torch.set_grad_enabled(train):
            if not train and self.test_cfg.get('flip_test', False):
                _feats = backbone(inputs)
                _feats_flip = backbone(inputs.flip(-1))
                feats = [_feats, _feats_flip]
            else:
                feats = backbone(inputs)
        return feats

    def forward_head_student(self, head, feats, data_samples, train):
        pred_heatmaps = None
        if not train and self.test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _feats = torch.nn.functional.normalize(self.student_neck(_feats), dim=1)
            # # _pred_heatmaps = head.forward(_feats)
            # _pred_heatmaps = self.student_head.forward(_pred_heatmaps)
            # _pred_heatmaps_flip = head.forward(torch.nn.functional.normalize(self.student_neck(_feats_flip), dim=1))
            # _pred_heatmaps_flip = self.student_head.forward(_pred_heatmaps_flip)
            # _pred_heatmaps_flip = flip_heatmaps(
            #     _pred_heatmaps_flip,
            #     flip_mode=self.test_cfg.get('flip_mode', 'heatmap'),
            #     flip_indices=flip_indices,
            #     shift_heatmap=self.test_cfg.get('shift_heatmap', False))
            # pred_heatmaps = (_pred_heatmaps + _pred_heatmaps_flip) * 0.5
            input_dino_recon_rgb = self.student_decoder(_feats)
        else:
            feats = self.student_neck(feats)
            pred_heatmaps = self.student_head.forward(feats)
            input_dino_recon_rgb = self.student_decoder(feats)
        return pred_heatmaps, input_dino_recon_rgb

    def forward_neck(self, neck, feats, train):
        with torch.set_grad_enabled(train):
            if not train and self.test_cfg.get('flip_test', False):
                _feats = neck(feats[0])
                _feats_flip = neck(feats[1])
                feats = [_feats, _feats_flip]
            else:
                feats = neck(feats)
        return feats

    def forward_head(self, head, feats, data_samples, train):
        if not train and self.test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _pred_heatmaps = head.forward(_feats)
            _pred_heatmaps_flip = head.forward(_feats_flip)
            _pred_heatmaps_flip = flip_heatmaps(
                _pred_heatmaps_flip,
                flip_mode=self.test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=self.test_cfg.get('shift_heatmap', False))
            pred_heatmaps = (_pred_heatmaps + _pred_heatmaps_flip) * 0.5
        else:
            pred_heatmaps = head.forward(feats)
        return pred_heatmaps


    def merge_feats(self, feats1, feats2, train):
        if not train and self.test_cfg.get('flip_test', False):
            feats_merged = [(feats1[0][0] + feats2[0][0]) / 2.0, (feats1[1][0] + feats2[1][0]) / 2.0]
        else:
            feats_merged = [(feats1[0] + feats2[0]) / 2.0]
        return feats_merged

    def forward(self, inputs: Tensor, data_samples: SampleList, train=False) -> tuple:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        inputs_dino = self.get_dino_inputs(data_samples)

        masks_rgb = np.stack([s.mask for s in data_samples])

        masks = torch.tensor(masks_rgb, device='cuda').unsqueeze(1)
        masks = torch.nn.functional.interpolate(masks.float(), size=(64, 64)).bool()

        if inputs_dino is not None:
            masks_dino = inputs_dino.sum(axis=1, keepdims=True) != 0
            masks = masks_dino & masks

        m_feats = masks.repeat(1, 32, 1, 1)
        m_recon = masks.repeat(1, 384, 1, 1)

        losses = dict()

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in data_samples])

        # if not self.distill:
        #     self.student_backbone.eval()
        #     self.student_neck.eval()

        # embed original dino features
        # with torch.set_grad_enabled(distill and train and False):
        #     self.dino_encoder.eval()
        #     self.dino_neck.eval()
        #     if not train and self.test_cfg.get('flip_test', False):
        #         _feats_dino = self.extract_emb(inputs_dino)
        #         _feats_dino_flip = self.extract_emb(inputs_dino.flip(-1))
        #         feats_dino = [_feats_dino, _feats_dino_flip]
        #         input_dino_recon = self.dino_decoder(feats_dino[0][0])
        #     else:
        #         feats_dino = self.extract_emb(inputs_dino)
        #         input_dino_recon = self.dino_decoder(feats_dino[0])

        # predict keypoints from RGB (without DINO)
        # with torch.set_grad_enabled(train):
        #     if not train and self.test_cfg.get('flip_test', False):
        #         _feats = self.extract_feat(inputs)
        #         _feats_flip = self.extract_feat(inputs.flip(-1))
        #         feats = [_feats, _feats_flip]
        #     else:
        #         feats = self.extract_feat(inputs)
        #
        # # embed RGB images
        # with torch.set_grad_enabled(self.distill and train):
        #     if not train and self.test_cfg.get('flip_test', False):
        #         _feats = self.extract_student(inputs)
        #         _feats_flip = self.extract_student(inputs.flip(-1))
        #         feats_student = [_feats, _feats_flip]
        #         input_dino_recon_rgb = self.student_decoder(feats_student[0][0])
        #     else:
        #         feats_student = self.extract_student(inputs)
        #         input_dino_recon_rgb = self.student_decoder(feats_student[0])
        #         # feats_student = [feats_student[0].detach()]

        input_dino_recon_rgb = None
        pred_heatmaps_student = None
        pred_heatmaps_rgb = None
        pred_heatmaps = None
        results = None

        # predict backbone
        with torch.set_grad_enabled(self.distill and train):
            feats = self.forward_backbone(self.backbone, inputs, train)
            ft_student = self.forward_neck(self.student_neck, feats, train)

        if self.distill:
            ft_ = ft_student
            if not train:
                ft_ = ft_student[0]
            input_dino_recon_rgb = self.student_decoder(ft_)
            loss_recon_rgb = torch.nn.functional.mse_loss(input_dino_recon_rgb[m_recon], inputs_dino[m_recon]) * 0.1
            losses.update(loss_recon_rgb=loss_recon_rgb)
        else:
            # detach features
            if train:
                feats = [feats[0].detach()]
                ft_student = ft_student.detach()

            # predict keypoints from student branch
            x = self.forward_neck(self.student_head_hr, ft_student, train)
            pred_heatmaps_student = self.forward_head(self.student_head, x, data_samples, train)
            loss_kpt_student = self.student_head.loss_module(pred_heatmaps_student, gt_heatmaps, keypoint_weights) * 100.0
            losses.update(loss_kpt_student=loss_kpt_student)

            # predict keypoint from head branch
            pred_heatmaps_rgb = self.forward_head(
                self.head,
                self.forward_neck(self.neck, feats, train),
                data_samples,
                train
            )
            loss_kpt_rgb = self.head.loss_module(pred_heatmaps_rgb, gt_heatmaps, keypoint_weights) * 100.0
            losses.update(loss_kpt_rgb=loss_kpt_rgb)

            # pred_heatmaps = (pred_heatmaps_rgb + pred_heatmaps_student) / 2.0
            pred_heatmaps = pred_heatmaps_rgb
            # pred_heatmaps = pred_heatmaps_student

            preds = self.head.decode(pred_heatmaps)
            pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps.detach()]

            # calculate accuracy
            if self.train_cfg.get('compute_acc', True):
                _, avg_acc, _ = pose_pck_accuracy(
                    output=to_numpy(pred_heatmaps),
                    target=to_numpy(gt_heatmaps),
                    mask=to_numpy(keypoint_weights) > 0)

                acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
                losses.update(acc_pose=acc_pose)

            results = self.add_pred_to_datasample(preds, pred_fields, data_samples)

        #
        # Visualization of training progress
        #
        interval = 40

        if self.batch_idx % interval == 0:
            if input_dino_recon_rgb is not None:
                # if not train and self.test_cfg.get('flip_test', False):
                    # feats_rgb = to_numpy(feats[0][0])
                    # feats_dino = to_numpy(feats_dino[0][0])
                    # feats_student = to_numpy(feats_student[0][0])
                    # feats_merged = to_numpy(feats_merged[0])
                # else:
                    # feats_rgb = to_numpy(feats[0])
                    # feats_dino = to_numpy(feats_dino[0])
                    # feats_student = to_numpy(feats_student[0])
                    # feats_merged = to_numpy(feats_merged[0])
                # self.vi.fit_pca(feats_student[:4])
                disp_dino = self.vi.visualize_batch(images=inputs,
                                                    attentions=to_numpy(inputs_dino),
                                                    # attentions_recon=to_numpy(input_dino_recon),
                                                    attentions_recon_rgb=to_numpy(input_dino_recon_rgb),
                                                    feats=[
                                                        # feats_dino,
                                                        # feats_rgb,
                                                        # feats_merged,
                                                        # feats_student
                                                    ],
                                                    masks=masks)
                cv2.imshow("Batch", cv2.cvtColor(disp_dino, cv2.COLOR_RGB2BGR))

            if pred_heatmaps_rgb is not None:
                preds = self.head.decode(pred_heatmaps_rgb)
                pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps_rgb.detach()]
                self.add_pred_to_datasample(preds, pred_fields, data_samples)
                disp_keypoints = create_keypoint_result_figure(inputs, data_samples)
                cv2.imshow("Predicted Keypoints RGB", cv2.cvtColor(disp_keypoints, cv2.COLOR_RGB2BGR))

            if pred_heatmaps_student is not None:
                preds = self.head.decode(pred_heatmaps_student)
                pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps_student.detach()]
                self.add_pred_to_datasample(preds, pred_fields, data_samples)
                disp_keypoints = create_keypoint_result_figure(inputs, data_samples)
                cv2.imshow("Predicted Keypoints Student", cv2.cvtColor(disp_keypoints, cv2.COLOR_RGB2BGR))

            # preds = self.head.decode(pred_heatmaps_dino)
            # pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps_dino.detach()]
            # self.add_pred_to_datasample(preds, pred_fields, data_samples)
            # disp_keypoints = create_keypoint_result_figure(inputs, data_samples)
            # cv2.imshow("Predicted Keypoints Dino", cv2.cvtColor(disp_keypoints, cv2.COLOR_RGB2BGR))

            cv2.waitKey(int(5))

        self.batch_idx += 1
        return losses, results

    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """
        assert len(batch_pred_instances) == len(batch_data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices],
                                              key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
