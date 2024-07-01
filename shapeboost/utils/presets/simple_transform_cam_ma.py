import math
import random

import cv2
cv2.setNumThreads(0)
from matplotlib.pyplot import get
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d,
                          get_affine_transform, get_affine_transform_rectangle, im_to_torch,
                          get_composed_trans)
from shapeboost.beta_decompose.beta_process2 import part_names
from shapeboost.beta_decompose.beta_process_finer2 import part_names_finer

skeleton_coco = np.array([(-1, -1)] * 28).astype(int)
skeleton_coco[ [6, 7, 17, 18, 19, 20] ] = np.array([
        (13, 15), (14, 16), (5, 7), (6, 8), (7, 9), (8, 10)
]).astype(int)
# print('skeleton_coco', skeleton_coco)

mma_attributes ={
    'female': [
        'Big', 
        'Broad Shoulders', 
        'Large Breasts', 
        'Long Legs', 
        'Long Neck', 
        'Long Torso', 
        'Muscular', 
        'Pear Shaped', 
        'Petite', 
        'Short', 
        'Short Arms', 
        'Skinny Legs', 
        'Slim Waist', 
        'Tall',
        'Feminine', 
    ],
    'male': [
        'Average', 
        'Big', 
        'Broad Shoulders', 
        'Delicate Build', 
        'Long Legs', 
        'Long Neck', 
        'Long Torso', 
        'Masculine', 
        'Muscular', 
        'Rectangular', 
        'Short', 
        'Short Arms', 
        'Skinny Arms', 
        'Soft Body', 
        'Tall'
    ],
}

model_output_attributes ={
    'female': [
        'Big', 
        'Broad Shoulders', 
        'Feminine', 
        'Large Breasts', 
        'Long Legs', 
        'Long Neck', 
        'Long Torso', 
        'Muscular', 
        'Pear Shaped', 
        'Petite', 
        'Short', 
        'Short Arms', 
        'Skinny Legs', 
        'Slim Waist', 
        'Tall'
    ],
    'male': [
        'Average', 
        'Big', 
        'Broad Shoulders', 
        'Delicate Build', 
        'Long Legs', 
        'Long Neck', 
        'Long Torso', 
        'Masculine', 
        'Muscular', 
        'Rectangular', 
        'Short', 
        'Short Arms', 
        'Skinny Arms', 
        'Soft Body', 
        'Tall'
    ],
}


class SimpleTransformCamMA(object):
    """Generation of cropped input person and pose heatmaps from SimplePose.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, color_factor, occlusion, add_dpg,
                 input_size, output_size, rot, sigma,
                 train, loss_type='MSELoss', dict_output=False, change_widths=True, bbox_3d_shape=0,
                 use_finer=False, depth_dim=64, focal_length=1000.0, scale_mult=1.1, use_mask=False):
        self._joint_pairs = dataset.joint_pairs
        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1
        self.dict_output = dict_output

        self.change_widths = change_widths
        self.scale_mult = scale_mult

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids
        
        if use_finer:
            self.part_names = part_names_finer
        else:
            self.part_names = part_names

        self.attributes_names = ['mass', 'height', 'chest', 'waist', 'hips']
        self.use_mask = use_mask

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        return img, bbox

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        if label['bbox'] is not None:
            bbox = list(label['bbox'])
        else:
            bbox = None
        # gt_joints = label['joint_img_29']
        joint_img = label['joint_img_29'].copy()
        joints_vis = label['joint_vis_29'].copy()
        self.num_joints = joint_img.shape[0]

        gt_joints = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
        gt_joints[:, :, 0] = joint_img
        gt_joints[:, :, 1] = joints_vis

        imgwidth = src.shape[1] 
        imght = src.shape[0]
        # imgwidth, imght = label['width'], label['height']
        # assert imgwidth == src.shape[1] and imght == src.shape[0]
        self.num_joints = gt_joints.shape[0]

        joints_vis = np.zeros((self.num_joints, 1), dtype=np.float32)
        joints_vis[:, 0] = gt_joints[:, 0, 1]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            if xmax < 0 or ymax < 0:
                xmin, ymin, xmax, ymax = 0, 0, imgwidth, imght
            
            center, scale = _box_to_center_scale(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self.scale_mult)
        else:
            xmin, ymin, xmax, ymax = 0, 0, imgwidth, imght
            center, scale = _box_to_center_scale(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self.scale_mult)

        if self._train and random.random() < 0.5:
            rand_norm = np.array([random.gauss(mu=0, sigma=1), random.gauss(mu=0, sigma=1)])
            rand_shift = 0.03 * scale * rand_norm
            center = center + rand_shift

        xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

        # half body transform
        # if self._train and (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
        if False:
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints[:, :, 0], joints_vis
            )

            if c_half_body is not None and s_half_body is not None:
                center, scale = c_half_body, s_half_body

        # rescale
        width_changed = False # if width_changed, do not supervise xyz29 and camera
        if self._train:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn()*0.5 * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.5 else 0
        else:
            r = 0
        
        r2 = 0

        cnt = 0
        if self._train and self._occlusion and bbox is not None and random.random() < 0.5:
            while True:
                cnt += 1
                area_min = 0.0
                area_max = 0.3
                synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                ratio_min = 0.3
                ratio_max = 1 / 0.3
                synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                synth_h = math.sqrt(synth_area * synth_ratio)
                synth_w = math.sqrt(synth_area / synth_ratio)
                synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                    synth_xmin = int(synth_xmin)
                    synth_ymin = int(synth_ymin)
                    synth_w = int(synth_w)
                    synth_h = int(synth_h)
                    src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                    break
                # elif random.random() < 0.001:
                #     synth_xmin = max(0, int(synth_xmin))
                #     synth_ymin = max(0, int(synth_ymin))
                #     synth_w = min(int(synth_w), imgwidth-synth_xmin)
                #     synth_h = min(int(synth_h), imght-synth_ymin)
                #     src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255

                if cnt > 10000:
                    print('bug')
                    assert False, (bbox, label['img_path'], src.shape)

        joints = gt_joints
        flipped = False
        if random.random() > 0.5 and self._train:
            flipped = True
            # src, fliped = random_flip_image(src, px=0.5, py=0)
            # if fliped[0]:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_3d(joints, imgwidth, self._joint_pairs)
            center[0] = imgwidth - center[0] - 1

        center_flipped = center.copy()
        center_flipped[0] = imgwidth - center[0] - 1

        inp_h, inp_w = input_size
        trans1 = get_affine_transform_rectangle(center, scale, r, [inp_w, inp_h])
        trans2 = get_affine_transform(
            center=np.array([128, 128]), scale=np.array([inp_w, inp_h]), rot=r2, output_size=[inp_w, inp_h])
        trans = get_composed_trans(trans1, trans2)
        trans_inv = get_composed_trans(trans1, trans2, inv=True)

        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        intrinsic_param = np.zeros((3, 3)).astype(np.float32)
        joint_root = np.zeros((3)).astype(np.float32)
        depth_factor = np.array([2200]).astype(np.float32)

        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # generate training targets
        target, target_weight = self._integral_target_generator(joints, self.num_joints, inp_h, inp_w)

        bbox = _center_scale_to_box(center, scale)
        # bbox_flipped = _center_scale_to_box(center_flipped, scale)

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        # cv2.imwrite(f'exp/visualize/shapeboost_coco/{bbox[0]}.png', img)
        cam_scale, cam_trans = 1, np.zeros(2)
        img_center = np.array( [float(imgwidth)*0.5, float(imght)*0.5] )

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        attributes_raw = label['attribute']

        attributes = [attributes_raw[n]  for n in self.attributes_names]
        attributes_w = [attributes_raw[f'{n}_w']  for n in self.attributes_names]
        gender = attributes_raw['gender']
        gender_w = attributes_raw['gender_w']
        # print(attributes_raw['attributes'])
        if not isinstance(attributes_raw['attributes'], list):
            assert attributes_raw['attributes'] == -1
            attributes_raw['attributes'] = [-1] * 15

        if gender == 'male':
            # male only account for first 15 attributes
            attributes += attributes_raw['attributes']
            attributes_w += ([attributes_raw['attributes_w'] * gender_w] * 15 )
            attributes += [0] * 15
            attributes_w += [0] * 15
        else:
            # reorder
            attributes += [0] * 15
            attributes_w += [0] * 15
            attributes += attributes_raw['attributes']
            attributes_w += ([attributes_raw['attributes_w'] * gender_w] * 15 )

        output = {
            'type': '2d_data',
            'image': img,
            'target': torch.from_numpy(target).float(),
            'target_weight': torch.from_numpy(target_weight).float()*0.0,
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
            'joint_root': torch.from_numpy(joint_root).float(),
            'depth_factor': torch.from_numpy(depth_factor).float(),
            'bbox': torch.Tensor(bbox),
            # 'bbox_flipped': torch.Tensor(bbox_flipped),
            'camera_scale': torch.from_numpy(np.array([cam_scale])).float(),
            'camera_trans': torch.from_numpy(cam_trans).float(),
            'camera_valid': 0.0,
            'camera_error': 0.0,
            'img_center': torch.from_numpy(img_center).float(),
            'part_widths': torch.zeros(len(self.part_names)).float(),
            'width_ratio': torch.ones(len(self.part_names)).float()*0.5,
            'width_changed': width_changed*1.0,
            'width_changed_ratio': scale[1] / scale[0],
            'attributes': torch.tensor(attributes).float(),
            'attributes_w': torch.tensor(attributes_w).float(),
        }

        if 'mask_path' in label and self.use_mask:
            mask_path = label['mask_path']
            mask_img = cv2.imread(mask_path)

            if flipped:
                mask_img = mask_img[:, ::-1, :]

            height, width = mask_img.shape[0], mask_img.shape[1]
            # mask_img_part = np.zeros((height, width, 22), dtype=np.uint8)
            # mask_img_part[:, :, 0] = (mask_img[:, :, 0] < 0.1) * 1.0

            mask_img_part = np.zeros((64, 64, 22), dtype=np.uint8)

            # mask_img_part = cv2.warpAffine(mask_img_part, trans, (256, 256), flags=cv2.INTER_LINEAR)
            # mask_img_part = cv2.resize(mask_img_part, (64, 64), interpolation=cv2.INTER_LINEAR)

            mask_img_0 = cv2.warpAffine(mask_img[:, :, [0]], trans, (256, 256), flags=cv2.INTER_LINEAR)
            mask_img_0 = cv2.resize(mask_img_0, (64, 64), interpolation=cv2.INTER_LINEAR)
            mask_img_part[:, :, 0] = (mask_img_0 < 0.1) * 1.0

            mask_img_part = torch.from_numpy(mask_img_part).float()
            mask_img_part = mask_img_part.permute(2, 0, 1)
            mask_part_weight = torch.zeros(22)
            mask_part_weight[0] = 1
            output.update({
                'mask_img_part': mask_img_part,
                'mask_part_weight': mask_part_weight,
            })
            
        return output


    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale
