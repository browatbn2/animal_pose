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
from mmpose.visualization.vis import create_keypoint_result_figure, Visualization, create_skeleton_result_figure


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

from mmengine.model import BaseModule, constant_init
import torch.nn as nn


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim * 2, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim * 2, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim * 2, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=in_dim * 2, num_heads=8, batch_first=True)

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        # proj_key = proj_key.permute(0, 2, 1)
        # proj_value = proj_value.permute(0, 2, 1)
        # out, attention = self.multihead_attn(proj_query, proj_key, proj_value)
        # out = out.permute(0, 2, 1)
        # out = out.view(m_batchsize, -1, width, height)

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, width, height)
        out = self.gamma * out + x

        return out, attention


from mmcv.cnn import build_conv_layer, build_upsample_layer

@MODELS.register_module()
class SelfAttentionNeck(BaseModule):
    def __init__(self):

        super().__init__()
        # embed_dim = 64 * 64
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.self_attention = SelfAttention(in_dim=128, activation=nn.ReLU())
        # out_cfg = dict(type='Conv2d', in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        out_cfg = dict(type='deconv', in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.out = build_conv_layer(out_cfg)


    def forward(self, x):
        x = self.conv1(x)
        x, attn = self.self_attention(x)
        x = self.out(x)
        return x


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
                 head_hr: OptConfigType = None,
                 dino_encoder: ConfigType = None,
                 dino_neck: OptConfigType = None,
                 dino_decoder: ConfigType = None,
                 student_backbone: ConfigType = None,
                 student_neck: OptConfigType = None,
                 student_head: OptConfigType = None,
                 student_head_hr: OptConfigType = None,
                 student_head_attn: OptConfigType = None,
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
        if head_hr is not None:
            self.head_hr = MODELS.build(head_hr)

        # self.student_backbone = MODELS.build(student_backbone)
        # self.student_neck = MODELS.build(student_neck)
        # self.student_head = MODELS.build(student_head)
        # self.student_decoder = MODELS.build(student_decoder)
        # if student_head_hr is not None:
        #     self.student_head_hr = MODELS.build(student_head_hr)

        self.dino_hdf5 = None
        self.dino_features = {}

        self.batch_idx = 0
        self.vi = Visualization()

        self.distill = distill

        # if True:
        #     split = 'train-split1'
        #     dino_file = get_dinov2_filepath(split=split)
        #     if dino_file is not None:
        #         assert os.path.isfile(dino_file), f"Could not find DINO feature file {dino_file}"
        #         self.dino_hdf5 = load_dino_hdf5(dino_file)

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

        heatmap_size = 64

        def count_parameters(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        # print(count_parameters(self.backbone))
        # print(count_parameters(self.student_head_attn))
        # exit()

        inputs_dino = self.get_dino_inputs(data_samples)

        masks_rgb = np.stack([s.mask for s in data_samples])

        masks = torch.tensor(masks_rgb, device='cuda').unsqueeze(1)
        masks = torch.nn.functional.interpolate(masks.float(), size=(64, 64)).bool()

        if inputs_dino is not None:
            masks_dino = inputs_dino.sum(axis=1, keepdims=True) != 0
            masks = masks_dino & masks
            m_recon = masks.repeat(1, inputs_dino.shape[1], 1, 1)
        else:
            m_recon = None

        m_feats = masks.repeat(1, 32, 1, 1)

        losses = dict()

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in data_samples])

        dino_recon = None
        pred_heatmaps = None
        results = None

        freeze_backbone = False

        # predict backbone
        with torch.set_grad_enabled((not freeze_backbone or self.distill) and train):
            feats = self.forward_backbone(self.backbone, inputs, train)
            feats = self.forward_neck(self.neck, feats, train)

        if self.distill:
            ft_ = feats
            if not train:
                ft_ = feats[0]
            dino_recon = self.dino_decoder(ft_)
            loss_recon_dino = torch.nn.functional.mse_loss(dino_recon[m_recon], inputs_dino[m_recon]) * 0.1
            losses.update(loss_recon_dino=loss_recon_dino)
        else:
            # detach features
            if train and freeze_backbone:
                feats = feats.detach()

            # predict keypoints with small HRNet
            # x = self.forward_neck(self.head_hr, feats, train)
            # pred_heatmaps = self.forward_head(self.head, x, data_samples, train)

            # predict keypoints directly from dino features
            pred_heatmaps = self.forward_head(self.head, feats, data_samples, train)

            loss_kpt_student = self.head.loss_module(pred_heatmaps, gt_heatmaps, keypoint_weights) * 100.0
            losses.update(loss_kpt_student=loss_kpt_student)

            preds = self.head.decode(pred_heatmaps)
            pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps.detach().cpu()]

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

        create_figure = True

        if create_figure:
            interval = 1
            wait = 0
        else:
            interval = 40
            wait = 5

        username = os.environ.get('USER')
        if self.batch_idx % interval == 0 and username == 'browatbn':
            if dino_recon is not None:
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
                                                    attentions_recon_rgb=to_numpy(dino_recon),
                                                    feats=[
                                                        # feats_dino,
                                                        # feats_rgb,
                                                        # feats_merged,
                                                        # feats_student
                                                    ],
                                                    masks=masks)
                cv2.imshow("Batch", cv2.cvtColor(disp_dino, cv2.COLOR_RGB2BGR))

            if pred_heatmaps is not None:
                preds = self.head.decode(pred_heatmaps)
                pred_fields = [PixelData(heatmaps=hm) for hm in pred_heatmaps.detach()]
                self.add_pred_to_datasample(preds, pred_fields, data_samples)
                disp_keypoints = create_keypoint_result_figure(inputs, data_samples)
                cv2.imshow("Predicted Keypoints RGB", cv2.cvtColor(disp_keypoints, cv2.COLOR_RGB2BGR))

                if create_figure:
                    disp_skeletons = create_skeleton_result_figure(inputs, data_samples, groundtruth=False)
                    cv2.imshow("Predicted Results", cv2.cvtColor(disp_skeletons, cv2.COLOR_RGB2BGR))

                    disp_skeletons = create_skeleton_result_figure(inputs, data_samples, groundtruth=True)
                    cv2.imshow("Groudtruth Results", cv2.cvtColor(disp_skeletons, cv2.COLOR_RGB2BGR))

                    disp_features = self.vi.visualize_batch(images=inputs, feats=[feats[0]], datasamples=datasamples, nimgs=len(inputs), horizontal=False)
                    cv2.imshow("Feature Maps", cv2.cvtColor(disp_features, cv2.COLOR_RGB2BGR))

            cv2.waitKey(wait)

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
