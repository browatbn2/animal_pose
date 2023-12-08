from itertools import zip_longest
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict
from mmengine.optim import OptimWrapper
import torch
import numpy as np

import cv2
from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .base import BasePoseEstimator
from mmpose.utils.tensor_utils import to_numpy
from mmpose.evaluation.functional import pose_pck_accuracy


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
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
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
            data = self.data_preprocessor(data, True)
            losses, outputs = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        log_vars['outputs'] = outputs
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def _predict_batch_dino(self, data_samples):
        attentions = torch.tensor(np.array([s.attentions for s in data_samples]), device=self.data_preprocessor.device)
        x = self.dino_encoder(attentions)
        attentions_emb = self.dino_neck(x)
        attentions_emb = torch.nn.functional.normalize(attentions_emb, dim=1)
        return attentions_emb

    def loss(self, inputs: Tensor, data_samples: SampleList) -> tuple:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """

        attentions_emb = self._predict_batch_dino(data_samples)
        # feats = self.extract_feat(inputs)
        feats = attentions_emb

        pred_fields = self.head.forward([feats])
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.head.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        if self.train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        # losses.update(
        #     self.head.loss(feats, data_samples, train_cfg=self.train_cfg))

        return losses, pred_fields

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [[_feats], [_feats_flip]]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        return results

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
