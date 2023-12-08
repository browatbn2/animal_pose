import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor
import cv2
import joblib
from sklearn.decomposition import PCA
import sklearn

from mmpose.utils.typing import SampleList
from mmpose.codecs.utils import get_heatmap_maximum

from mmpose.utils import nn
from mmpose.utils.nn import Batch, to_numpy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize(tensor):
    if tensor.shape[1] == 3:
        tensor[:, 0:3] *= np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
        tensor[:, 0:3] += np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    elif tensor.shape[-1] == 3:
        tensor[..., 0:3] *= IMAGENET_STD
        tensor[..., 0:3] += IMAGENET_MEAN


def denormalized(tensor):
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    denormalize(t)
    return t


def color_map(data, vmin=None, vmax=None, cmap=plt.cm.viridis):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    val = np.maximum(vmin, np.minimum(vmax, data))
    norm = (val-vmin)/(vmax-vmin)
    cm = cmap(norm)
    if isinstance(cm, tuple):
        return cm[:3]
    if len(cm.shape) > 2:
        cm = cm[:,:,:3]
    return cm


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def make_grid(data, padsize=2, padval=255, nCols=10, dsize=None, fx=None, fy=None, normalize=False):
    # if not isinstance(data, np.ndarray):
    data = np.array(data)
    if data.shape[0] == 0:
        return
    if data.shape[1] == 3:
        data = data.transpose((0,2,3,1))
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if normalize:
        data -= data.min()
        data /= data.max()
    else:
        data[data < 0] = 0
    #     data[data > 1] = 1

    # force the number of filters to be square
    # n = int(np.ceil(np.sqrt(data.shape[0])))
    c = nCols
    r = int(np.ceil(data.shape[0]/float(c)))

    padding = ((0, r*c - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((r, c) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((r * data.shape[1], c * data.shape[3]) + data.shape[4:])

    if dsize is not None or fx is not None or fy is not None:
        # data = cv2.resize(data, dsize=dsize, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
        data = cv2.resize(data, dsize=dsize, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

    return data


def vis_square(data, padsize=1, padval=0, wait=0, nCols=10, title='results', dsize=None, fx=None, fy=None, normalize=False):
    img = make_grid(data, padsize=padsize, padval=padval, nCols=nCols, dsize=dsize, fx=fx, fy=fy, normalize=normalize)
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)


def cvt32FtoU8(img):
    return (img * 255.0).astype(np.uint8)


def to_single_channel_heatmap(heatmaps):
    assert heatmaps.ndim in [3, 4]
    axis = 1 if heatmaps.ndim == 4 else 0
    return to_numpy(heatmaps).max(axis=axis)


def to_disp_image(img, denorm=True, output_dtype=np.uint8):
    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float32).copy()
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    if denorm and img.min() < 0:
        img = denormalized(img)
    if img.max() > 2.00:
        if isinstance(img, np.ndarray):
            img /= 255.0
        else:
            raise ValueError("Image data in wrong value range (min/max={:.2f}/{:.2f}).".format(img.min(), img.max()))
    img = np.clip(img, a_min=0, a_max=1)
    if output_dtype == np.uint8:
        img = cvt32FtoU8(img)
    if len(img.shape) == 3 and img.shape[0] == 1:
        img = img[0]
    return img

def to_disp_images(images, denorm=True, output_dtype=np.uint8):
    return [to_disp_image(i, denorm, output_dtype) for i in images]

def prepare_overlay(img, m, cmap):
    m_new = m.copy()
    if m_new.shape != img.shape:
        m_new = cv2.resize(m_new, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    m_colored = color_map(m_new**1, vmin=0, vmax=1.0, cmap=cmap)
    if len(m_colored.shape) < len(img.shape):
        m_colored = m_colored[..., np.newaxis]
    return m_colored, m_new > 0.05


def overlay_heatmap(img, mat, opacity=0.5, cmap=plt.cm.inferno):
    if img is None:
        h, w = mat.shape[-2:]
        img = np.zeros((h, w, 3), dtype=np.uint8)
    img_dtype = img.dtype
    img_new = img.copy()
    if img_new.dtype == np.uint8:
        img_new = img_new.astype(np.float32) / 255.0

    overlay, mask = prepare_overlay(img, mat, cmap)
    img_new = img_new + overlay * opacity

    img_new = img_new.clip(0, 1)
    if img_dtype == np.uint8:
        img_new = cvt32FtoU8(img_new)
    assert img_new.dtype == img.dtype
    assert img_new.shape == img.shape
    return img_new
def add_frames_to_images(images, labels, label_colors, gt_labels=None):
    import collections
    if not isinstance(labels, (collections.Sequence, np.ndarray)):
        labels = [labels] * len(images)
    new_images = to_disp_images(images)
    for idx, (disp, label) in enumerate(zip(new_images, labels)):
        frame_width = 3
        bgr = label_colors[label]
        cv2.rectangle(disp,
                      (frame_width // 2, frame_width // 2),
                      (disp.shape[1] - frame_width // 2, disp.shape[0] - frame_width // 2),
                      color=bgr,
                      thickness=frame_width)

        if gt_labels is not None:
            radius = 8
            color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
            cv2.circle(disp, (disp.shape[1] - 2*radius, 2*radius), radius, color, -1)
    return new_images


def add_cirle_to_images(images, intensities, cmap=plt.cm.viridis, radius=10):
    new_images = to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, intensities)):
        # color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
        # color = plt_colors.to_rgb(val)
        if isinstance(val, float):
            color = np.array(cmap(val)) * 255
        else:
            color = val
        cv2.circle(disp, (2*radius, 2*radius), radius, color, -1, lineType=cv2.LINE_AA)
        # new_images.append(disp)
    return new_images


def get_pos_in_image(loc, text_size, image_shape):
    bottom_offset = int(6*text_size)
    right_offset = int(95*text_size)
    line_height = int(35*text_size)
    mid_offset = right_offset
    top_offset = line_height + int(0.05*line_height)
    if loc == 'tl':
        pos = (2, top_offset)
    elif loc == 'tr':
        pos = (image_shape[1]-right_offset, top_offset)
    elif loc == 'tr+1':
        pos = (image_shape[1]-right_offset, top_offset + line_height)
    elif loc == 'tr+2':
        pos = (image_shape[1]-right_offset, top_offset + line_height*2)
    elif loc == 'bl':
        pos = (2, image_shape[0]-bottom_offset)
    elif loc == 'bl-1':
        pos = (2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bl-2':
        pos = (2, image_shape[0]-bottom_offset-2*line_height)
    # elif loc == 'bm':
    #     pos = (mid_offset, image_shape[0]-bottom_offset)
    # elif loc == 'bm-1':
    #     pos = (mid_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset)
    elif loc == 'br-1':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br-2':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-2*line_height)
    elif loc == 'bm':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset)
    elif loc == 'bm-1':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bm-2':
        pos = (image_shape[1]-right_offset*2, image_shape[0]-bottom_offset-2*line_height)
    else:
        raise ValueError("Unknown location {}".format(loc))
    return pos


def add_label_to_images(images, labels, gt_labels=None, loc='tl', color=(255,255,255), size=0.7, thickness=1):
    new_images = to_disp_images(images)
    _labels = to_numpy(labels)
    _gt_labels = to_numpy(gt_labels)
    for idx, (disp, val) in enumerate(zip(new_images, _labels)):
        if _gt_labels is not None:
            color = (0,255,0) if _labels[idx] == _gt_labels[idx] else (255,0,0)
        # if val != 0:
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, str(val), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_error_to_images(images, errors, loc='bl', size=0.65, vmin=0., vmax=30.0, thickness=1,
                        format_string='{:.2f}', cl=None, colors=None, cmap=plt.cm.jet):
    assert cl is None or colors is None
    new_images = to_disp_images(images)
    errors = to_numpy(errors)
    if cl is not None:
        colors = [cl for i in range(len(new_images))]
    if colors is None:
        colors = color_map(to_numpy(errors), cmap=cmap, vmin=vmin, vmax=vmax)
        if images[0].dtype == np.uint8:
            colors *= 255
    for disp, err, color in zip(new_images, errors, colors):
        pos = get_pos_in_image(loc, size, disp.shape)
        err_str = np.array2string(err, precision=3, separator=' ', suppress_small=True)
        # err_str = format_string.format(err)
        cv2.putText(disp, err_str, pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


# FIXME: clean up this mess!
def add_landmarks_to_images(images, landmarks, color=None, radius=2, gt_landmarks=None, keypoints_visible=None,
                            lm_errs=None, lm_confs=None, lm_rec_errs=None,
                            draw_dots=True, draw_wireframe=False, draw_gt_offsets=False, landmarks_to_draw=None,
                            offset_line_color=None, offset_line_thickness=1):

    def draw_wireframe_lines(img, lms):
        pts = lms.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img, [pts[:17]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # head outline
        cv2.polylines(img, [pts[17:22]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # left eyebrow
        cv2.polylines(img, [pts[22:27]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # right eyebrow
        cv2.polylines(img, [pts[27:31]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose vert
        cv2.polylines(img, [pts[31:36]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose hor
        cv2.polylines(img, [pts[36:42]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # left eye
        cv2.polylines(img, [pts[42:48]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # right eye
        cv2.polylines(img, [pts[48:60]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # outer mouth
        cv2.polylines(img, [pts[60:68]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth

    def draw_wireframe_lines_98(img, lms):
        pts = lms.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img, [pts[:33]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # head outline
        cv2.polylines(img, [pts[33:42]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # left eyebrow
        # cv2.polylines(img, [pts[38:42]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # right eyebrow
        cv2.polylines(img, [pts[42:51]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # nose vert
        cv2.polylines(img, [pts[51:55]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose hor
        cv2.polylines(img, [pts[55:60]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # left eye
        cv2.polylines(img, [pts[60:68]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # right eye
        cv2.polylines(img, [pts[68:76]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # outer mouth
        cv2.polylines(img, [pts[76:88]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth
        cv2.polylines(img, [pts[88:96]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth

    def draw_offset_lines(img, lms, gt_lms, errs):
        if gt_lms.sum() == 0:
            return
        if lm_errs is None:
            # if offset_line_color is None:
            offset_line_color = (255, 255, 255)
            colors = [offset_line_color] * len(lms)
        else:
            colors = color_map(errs, cmap=plt.cm.jet, vmin=0, vmax=15.0)
        if img.dtype == np.uint8:
            colors *= 255
        for i, (p1, p2) in enumerate(zip(lms, gt_lms)):
            if landmarks_to_draw is None or i in landmarks_to_draw:
                visible = errs is None or not np.isnan(errs[i])
                valid_pair = not np.isnan(p1).any() and not np.isnan(p2).any() and p1.min() > 0 and visible
                if valid_pair:
                    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), colors[i], thickness=offset_line_thickness, lineType=cv2.LINE_AA)

    if landmarks is None:
        return images

    new_images = to_disp_images(images)
    landmarks = to_numpy(landmarks)
    gt_landmarks = to_numpy(gt_landmarks)
    lm_errs = to_numpy(lm_errs)
    img_size = new_images[0].shape[0]
    default_color = (255,255,0)

    if len(landmarks.shape) == 2:
        landmarks = landmarks[np.newaxis]

    if gt_landmarks is not None:
        if len(gt_landmarks.shape) == 2:
            gt_landmarks = gt_landmarks[np.newaxis]

        if draw_gt_offsets:
            for img_id  in range(len(new_images)):
                if gt_landmarks[img_id].sum() == 0:
                    continue
                dists = None
                if lm_errs is not None:
                    dists = lm_errs[img_id]
                draw_offset_lines(new_images[img_id], landmarks[img_id], gt_landmarks[img_id], dists)

    for img_id, (disp, lm)  in enumerate(zip(new_images, landmarks)):
        if len(lm) in [68, 21, 19, 98, 8, 5]:
            if draw_dots:
                for lm_id in range(0,len(lm)):
                    if landmarks_to_draw is None or lm_id in landmarks_to_draw or len(lm) != 68:
                        lm_color = color
                        if lm_color is None:
                            if lm_errs is not None:
                                lm_color = color_map(lm_errs[img_id, lm_id], cmap=plt.cm.jet, vmin=0, vmax=1.0)
                            else:
                                lm_color = default_color
                        # if lm_errs is not None and lm_errs[img_id, lm_id] > 40.0:
                        #     lm_color = (1,0,0)
                        cv2.circle(disp, tuple(lm[lm_id].astype(int).clip(0, disp.shape[0]-1)), radius=radius, color=lm_color, thickness=-1, lineType=cv2.LINE_AA)
                        if lm_confs is not None:
                            max_radius = img_size * 0.05
                            try:
                                conf_radius = max(2, int((1-lm_confs[img_id, lm_id]) * max_radius))
                            except ValueError:
                                conf_radius = 2
                            # if lm_confs[img_id, lm_id] > 0.4:
                            cirle_color = (0,0,255)
                            # if lm_confs[img_id, lm_id] < is_good_landmark(lm_confs, lm_rec_errs):
                            # if not is_good_landmark(lm_confs[img_id, lm_id], lm_rec_errs[img_id, lm_id]):
                            if lm_errs[img_id, lm_id] > 10.0:
                                cirle_color = (255,0,0)
                            cv2.circle(disp, tuple(lm[lm_id].astype(int)), conf_radius, cirle_color, 1, lineType=cv2.LINE_AA)

            # Draw outline if we actually have 68 valid landmarks.
            # Landmarks can be zeros for UMD landmark format (21 points).
            if draw_wireframe:
                nlms = (np.count_nonzero(lm.sum(axis=1)))
                if nlms == 68:
                    draw_wireframe_lines(disp, lm)
                elif nlms == 98:
                    draw_wireframe_lines_98(disp, lm)
        else:
            # colors = ['tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:red', 'tab:blue']
            # colors_rgb = list(map(plt_colors.to_rgb, colors))

            # colors = sns.color_palette("Set1", n_colors=14)
            if draw_dots:
                if color is None:
                    cl = default_color
                else:
                    cl = color
                for i in range(0,len(lm)):
                    visible = keypoints_visible is None or keypoints_visible[img_id, i]
                    if not visible:
                        continue
                    x, y = lm[i].astype(int)
                    if x >=0 and y >= 0:
                        cv2.circle(disp, (x, y), radius=radius, color=cl, thickness=1, lineType=cv2.LINE_AA)
    return new_images



def visualize_batch_pose(
        images,
        heatmaps=None,
        keypoints_pred=None,
        keypoints_gt=None,
        keypoints_visible: np.ndarray = None,
        keypoints_visible_pred: np.ndarray = None,
        f=1.0,
        show_images=True,
        show_heatmaps=True,
        gt_color=(0, 255, 0),
        pred_color=(0, 0, 255),
        draw_skeleton=True,
        skeleton_links=None,
        skeleton_link_colors=None,
        nimgs=5
):
    assert show_images or show_heatmaps, "Parameters 'show_images' and 'show_heatmaps' cannot both be False!"

    B, C, H, W = images.shape
    assert H in [128, 256]

    nimgs = min(nimgs, len(images))
    images = nn.atleast4d(images)[:nimgs]

    keypoints_pred = to_numpy(keypoints_pred)

    def resize_image(im, dsize=None, f=None):
        return cv2.resize(im, dsize, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)

    # if landmarks_to_draw is None:
    #     landmarks_to_draw = range(num_landmarks)

    disp_images = to_disp_images(images[:nimgs], denorm=True)
    disp_images = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in disp_images]

    # show heatmaps
    if heatmaps is not None and show_heatmaps:
        merged_heatmaps = to_single_channel_heatmap(to_numpy(heatmaps[:nimgs]))

        if show_images:
            # overlay keypoints on input images
            disp_images = [
                overlay_heatmap(disp_images[i], merged_heatmaps[i]) for i in range(len(merged_heatmaps))
            ]
        else:
            # convert heatmaps to color-mapped RGB images
            def to_rgb(hm):
                return (color_map(hm, vmin=0, vmax=1, cmap=plt.cm.jet)*255).astype(np.uint8)
            disp_images = [to_rgb(hm) for hm in merged_heatmaps]
            disp_images = [resize_image(im, dsize=(W, H)) for im in disp_images]

    # resize images for display and scale landmarks accordingly
    disp_images = [resize_image(im, f=f) for im in disp_images]

    #
    # Annotate with predicted and GT keypoints
    #
    if keypoints_gt is not None:
        keypoints_gt *= f
        disp_images = add_landmarks_to_images(disp_images, keypoints_gt, keypoints_visible=keypoints_visible,
                                                  color=gt_color)
        # disp_images = add_skeletons_to_images(disp_images, keypoints_gt, keypoints_visible=keypoints_visible,
        #                                           skeleton_links=skeleton_links,
        #                                           skeleton_link_colors=skeleton_link_colors)
    if keypoints_pred is not None:
        keypoints_pred = keypoints_pred[:nimgs] * f
        disp_images = add_landmarks_to_images(disp_images, keypoints_pred, keypoints_visible=keypoints_visible,
                                                  color=gt_color)

        if keypoints_visible_pred is not None:
            disp_images = add_landmarks_to_images(disp_images, keypoints_pred, keypoints_visible=keypoints_visible_pred,
                                                      color=(255, 0, 255), radius=3)

    if keypoints_pred is not None and keypoints_gt is not None:
        if keypoints_gt is None:
            keypoints_gt = np.zeros((nimgs, keypoints_pred.shape[1], 2))
        else:
            keypoints_gt = nn.atleast3d(to_numpy(keypoints_gt))[:nimgs]

        # nme_per_lm = calc_landmark_nme(keypoints_gt, keypoints_pred[:nimgs], keypoints_visible[:nimgs])

        disp_images = add_landmarks_to_images(disp_images, keypoints_pred, keypoints_visible=keypoints_visible,
            # lm_errs=nme_per_lm,
                                              gt_landmarks=keypoints_gt, color=pred_color, draw_gt_offsets=True)

        # lm_errs = calc_landmark_nme_per_img(keypoints_gt, keypoints_pred, None, keypoints_visible)
        # disp_images = vis.add_error_to_images(disp_images, lm_errs, loc='br', format_string='{:>5.2f}', vmax=15)

    return make_grid(disp_images, nCols=len(disp_images))


def create_keypoint_result_figure(inputs: Tensor, outputs: Tensor, data_samples: SampleList):

    keypoints_gt = np.concatenate([s.gt_instances[0].transformed_keypoints for s in data_samples])
    keypoints_visible = np.concatenate([s.gt_instances[0].keypoints_visible for s in data_samples])
    heatmaps_gt = np.stack([to_numpy(s.gt_fields.heatmaps) for s in data_samples])
    pred, _ = get_heatmap_maximum(to_numpy(outputs))
    rows = []

    rows.append(visualize_batch_pose(inputs,
                                     keypoints_pred=pred * 4.0,
                                     keypoints_gt=keypoints_gt,
                                     keypoints_visible=keypoints_visible,
                                     show_heatmaps=False))
    # rows.append(visualize_batch_pose(b1.images,
    #                                  keypoints_gt=b1.keypoints_pred,
    #                                  keypoints_visible=b1.keypoints_visible,
    #                                  draw_skeleton=True))

    rows.append(visualize_batch_pose(inputs, heatmaps=heatmaps_gt, show_images=False))
    rows.append(visualize_batch_pose(inputs, heatmaps=outputs, show_images=False))
    return make_grid(rows, nCols=1)


def semfeats_to_images(feats):
    """ convert 3 channel inputs to RGB images, 1 channel inputs to color-mapped heatmaps"""

    def _cmap(tensor, vmin=0, vmax=1.0, cmap=plt.cm.viridis):
        cmapped = [color_map(img, vmin=vmin, vmax=vmax, cmap=cmap) for img in to_numpy(tensor)]
        return [(img * 255).astype(np.uint8) for img in cmapped]

    if feats.min() < 0:
        feats_norm = (feats + 1) / 2 # features are in range [-1,1] -> convert o [0,1]
    else:
        feats_norm = feats
    imgs = to_disp_images(feats_norm, denorm=False)
    if len(imgs[0].shape) == 2:
        if imgs[0].max() > 2.0:
            imgs = [i/255 for i in imgs]
        imgs = _cmap(imgs, cmap=plt.cm.magma, vmin=-1.0, vmax=1.0)
    return imgs


def resize_array(a: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    assert a.ndim == 4
    return to_numpy(F.interpolate(torch.tensor(a), size=size))


class PCAVis():
    def __init__(self, segment=True, bg_thresh=0.5):
        self.pca = None
        self.pca_fg = None
        self.segment = segment
        self.bg_thresh = bg_thresh

    def load(self, filepath):
        print(f"Loading PCA models from {filepath}")
        self.pca = joblib.load(filepath+'.pca.bg.joblib')
        if self.segment:
            self.pca_fg = joblib.load(filepath+'.pca.fg.joblib')

    def save(self, filepath):
        print(f"Saving PCA models to {filepath}")
        joblib.dump(self.pca, filepath+'.pca.bg.joblib')
        if self.pca_fg is not None:
            joblib.dump(self.pca_fg, filepath+'.pca.fg.joblib')

    @staticmethod
    def _plot_components(pca_features):
        """ visualize PCA components for finding a proper threshold 3 histograms for 3 components"""
        plt.subplot(2, 2, 1)
        plt.hist(pca_features[:, 0])
        plt.subplot(2, 2, 2)
        plt.hist(pca_features[:, 1])
        plt.subplot(2, 2, 3)
        plt.hist(pca_features[:, 2])
        plt.show()
        plt.close()

    @staticmethod
    def _show_component(pca_features, comp_id,  H, W):
        for i in range(len(pca_features)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features[i * H * W: (i + 1) * H * W, comp_id].reshape(H, W))
        plt.show()
        plt.close()

    def fit(self, features: np.ndarray | torch.Tensor):
        features = to_numpy(features)

        N, D, H, W = features.shape
        features = features.transpose(0, 2, 3, 1).reshape(-1, D)

        # print(f"Fitting PCA for background segmentation...")
        self.pca = PCA(n_components=3)
        self.pca.fit(features)

        if self.segment:
            pca_features = self.pca.transform(features)
            pca_features = self._minmax_scale(pca_features)

            # self._plot_components(pca_features)
            # self._show_component(pca_features[:4], 0, H, W)

            # segment/seperate the backgound and foreground using the first component
            pca_features_bg = pca_features[:, 0] < self.bg_thresh  # from first histogram
            pca_features_fg = ~pca_features_bg

            if False:
                # plot the pca_features_bg
                for i in range(4):
                    plt.subplot(2, 2, i + 1)
                    plt.imshow(pca_features_bg[i * H * W: (i + 1) * H * W].reshape(H, W))
                plt.show()
                plt.close()

            # print(f"Fitting PCA for only foreground patches...")
            self.pca_fg = PCA(n_components=3)
            self.pca_fg.fit(features[pca_features_fg])

    @staticmethod
    def _minmax_scale(features):
        features = sklearn.preprocessing.minmax_scale(features)
        return features

    def transform(self, features: np.ndarray | torch.Tensor):
        features = to_numpy(features)

        input_ndim = len(features.shape)
        if input_ndim == 3:
            features = features[np.newaxis]

        if self.pca is None:
            self.fit(features)

        N, D, H, W = features.shape

        if D == 9:  # or D == 32:
            features = features[:, :3]
            D = 3

        if D == 3:
            pca_features = features.transpose(0, 2, 3, 1)

            f = pca_features
            for i in range(3):
                f[..., i] = (f[...,i] - f[...,i].min()) / (f[..., i].max() - f[...,i].min())
            return (255 * f).astype(np.uint8)

        features = features.transpose(0, 2, 3, 1).reshape(-1, D)

        pca_features = self.pca.transform(features)
        pca_features = self._minmax_scale(pca_features)

        if self.segment:
            # segment/seperate the backgound and foreground using the first component
            pca_features_bg = pca_features[:, 0] < self.bg_thresh
            pca_features_fg = ~pca_features_bg

            pca_features_left = self.pca_fg.transform(features[pca_features_fg])
            pca_features_left = self._minmax_scale(pca_features_left)

            pca_features_rgb = pca_features.copy()
            pca_features_rgb[pca_features_bg] = 0
            pca_features_rgb[pca_features_fg] = pca_features_left
        else:
            pca_features_rgb = pca_features

        # reshaping to numpy image format
        pca_features_rgb = pca_features_rgb.reshape(N, H, W, 3)

        if False:
            nimgs = min(10, N)
            for i in range(nimgs):
                plt.subplot(min(2, nimgs), nimgs // min(2, nimgs), i + 1)
                plt.imshow(pca_features_rgb[i])
            plt.show()
            plt.close()

        if input_ndim == 3:
            pca_features_rgb = pca_features_rgb[0]

        return (pca_features_rgb * 255).astype(np.uint8)


class Visualization(object):
    def __init__(self, image_size=(256, 256)):
        self.pcavis = PCAVis(segment=False)
        self.dinovis = PCAVis(segment=False)
        # self.dinovis.load("../data/dino/dino_train-split1_vits14_14.h5")
        # self.dinovis.load("/media/browatbn/data/dino/dino_train-split1_vits14_14.h5")
        self.dinovis.load("/home/browatbn/dev/data/dino/dino_train-split1_vits14_14.h5")
        self.H = image_size[0]
        self.W = image_size[1]

    def fit_pca(self, features):
        self.pcavis.fit(features)

    def draw_attentions(self, attentions):
        features_rgb = self.dinovis.transform(attentions) # returns [N, H, W, 3]
        features_rgb = resize_array(features_rgb.transpose(0, 3, 1, 2), size=(self.H, self.W)).transpose(0, 2, 3, 1)
        return features_rgb

    def draw_features_pca(self, semfeats):
        features_rgb = self.pcavis.transform(semfeats) # returns [N, H, W, 3]
        features_rgb = resize_array(features_rgb.transpose(0, 3, 1, 2), size=(self.H, self.W)).transpose(0, 2, 3, 1)
        return features_rgb

    def visualize_batch(self,
                        images,
                        attentions,
                        horizontal=True,
                        nimgs=4):

        nimgs = min(nimgs, len(images))
        ncols = 1 if horizontal else nimgs

        rows = []

        N, C, H, W = images.shape
        f = 1.0 * (256 / H)

        disp_X1 = to_disp_images(images[:nimgs])
        disp_X1 = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in disp_X1]
        rows.append(make_grid(disp_X1, nCols=ncols))

        # X1_feats = attentions
        # if X1_feats is not None:
        #     X1_feats = F.interpolate(X1_feats[:nimgs], size=(H, W))
        #     X1_feats = torch.nn.functional.normalize(X1_feats, dim=1)

        #
        # show attentions (all channels averaged per image)
        #

        # dino1
        # def draw_attentions(attentions):
        #     resized_atts = F.interpolate(attentions, size=(H, W))
        #     mean_atts = to_numpy(resized_atts.mean(dim=1))
        #     return cmap(mean_atts, cmap=plt.cm.viridis, vmin=mean_atts.min(), vmax=mean_atts.max())

        if attentions is not None:
            rows.append(make_grid(self.draw_attentions(attentions[:nimgs]), nCols=ncols))
        # if attentions_recon is not None:
        #     rows.append(make_grid(self.draw_attentions(batch1.attentions_recon[:nimgs]), nCols=ncols))
        # if attentions_emb is not None:
        #     rows.append(make_grid(self.draw_features_pca(batch1.attentions_emb[:nimgs]), nCols=ncols))

        ncols_panel = len(rows) if horizontal else 1
        return make_grid(rows, nCols=ncols_panel, normalize=False, fx=f, fy=f)