import math
import random

import cv2
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, get_affine_transform_rectangle, im_to_torch, batch_rodrigues_numpy, flip_twist,
                          rotmat_to_quat_numpy, rotate_xyz_jts, rot_aa, flip_cam_xyz_joints_3d, get_composed_trans)
from ..pose_utils import get_intrinsic_metrix
from shapeboost.beta_decompose.beta_process2 import vertice2capsule_fast, part_names, spine_joints
from shapeboost.beta_decompose.beta_process_finer2 import vertice2capsule_finer, part_names_finer, spine_joints_finer
from shapeboost.models.layers.smpl.SMPL import SMPL_layer

s_coco_2_smpl_jt = [
    -1, 11, 12,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]


left_bones_idx = [
    (0, 1), (1, 4), (4, 7), (12, 13),
    (13, 16), (16, 18), (18, 20)
]

right_bones_idx = [
    (0, 2), (2, 5), (5, 8), (12, 14),
    (14, 17), (17, 19), (19, 21)
]

skeleton_29 = [ 
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), # 5
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), # 11
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), # 17
    (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (15, 24), # 23
    (22, 25), (23, 26), (10, 27), (11, 28) # 27
]

skeleton_3dhp = np.array([(-1, -1)] * 28).astype(int)
skeleton_3dhp[ [6, 7, 17, 18, 19, 20] ] = np.array([
        (19, 20), (24, 25), (9, 10), (14, 15), (10, 11), (15, 16)
    ]).astype(int)


class SimpleTransform3DCamEFT(object):
    """Generation of cropped input person, pose coords, smpl parameters.

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
                 input_size, output_size, depth_dim, bbox_3d_shape,
                 rot, sigma, train, loss_type='MSELoss', scale_mult=1.25, focal_length=1000, two_d=False,
                 root_idx=0, change_widths=True, use_finer=False):
        
        self._joint_pairs_24 = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self._joint_pairs_29 = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
        self._joint_pairs_17 = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

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

        self.bbox_3d_shape = dataset.bbox_3d_shape
        self._scale_mult = scale_mult
        # self.kinematic = dataset.kinematic
        self.two_d = two_d

        # convert to unit: meter
        self.depth_factor2meter = self.bbox_3d_shape[2] if self.bbox_3d_shape[2] < 50 else self.bbox_3d_shape[2]*1e-3

        self.focal_length = focal_length
        self.root_idx = root_idx
        self.change_widths = change_widths

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids
        
        if use_finer:
            self.vertice2capsule_fast, self.part_names, self.spine_joints = \
                vertice2capsule_finer, part_names_finer, spine_joints_finer
        else:
            self.vertice2capsule_fast, self.part_names, self.spine_joints = \
                vertice2capsule_fast, part_names, spine_joints

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
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

        img_center = np.array([float(src.shape[1]) * 0.5, float(src.shape[0]) * 0.5])

        return img, bbox, img_center

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[0]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_uvd_target_generator(self, joints_3d, num_joints, patch_height, patch_width):

        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 1, 1]
        target_weight[:, 2] = joints_3d[:, 2, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_xyz_target_generator(self, joints_3d, joints_3d_vis, num_joints):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d_vis[:, 0]
        target_weight[:, 1] = joints_3d_vis[:, 1]
        target_weight[:, 2] = joints_3d_vis[:, 2]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0] / self.bbox_3d_shape[0]
        target[:, 1] = joints_3d[:, 1] / self.bbox_3d_shape[1]
        target[:, 2] = joints_3d[:, 2] / self.bbox_3d_shape[2]

        # if self.bbox_3d_shape[0] < 1000:
        #     print(self.bbox_3d_shape, target)
        
        # assert (target[0] == 0).all(), f'{target}, {self.bbox_3d_shape}'

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        
        bbox = list(label['bbox'])
        joint_img_17 = label['joint_img_17'].copy()
        joint_relative_17 = label['joint_relative_17'].copy()
        joint_cam_17 = label['joint_cam_17'].copy()
        joints_vis_17 = label['joint_vis_17'].copy()
        joint_img_29 = label['joint_img_29'].copy()
        joint_cam_29 = label['joint_cam_29'].copy()
        joints_vis_29 = label['joint_vis_29'].copy()
        joints_vis_xyz_29 = label['joint_vis_xyz_29'].copy()
        smpl_weight = label['smpl_weight'].copy()
        # root_cam = label['root_cam'].copy()
        # root_depth = root_cam[2] / self.bbox_3d_shape[2]
        self.num_joints = joint_img_29.shape[0]

        beta = label['beta'].copy()
        theta = label['theta'].copy()

        beta_kid = label['beta_kid'].copy() if 'beta_kid' in label else np.zeros(1)

        if 'twist_phi' in label.keys():
            twist_phi = label['twist_phi'].copy()
            twist_weight = label['twist_weight'].copy()
        else:
            twist_phi = np.zeros((23, 2))
            twist_weight = np.zeros((23, 2))

        gt_joints_17 = np.zeros((17, 3, 2), dtype=np.float32)
        gt_joints_17[:, :, 0] = joint_img_17.copy()
        gt_joints_17[:, :, 1] = joints_vis_17.copy()
        gt_joints_29 = np.zeros((29, 3, 2), dtype=np.float32)
        gt_joints_29[:, :, 0] = joint_img_29.copy()
        gt_joints_29[:, :, 1] = joints_vis_29.copy()

        imgwidth, imght = src.shape[1], src.shape[0]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)

        xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

        # half body transform
        self.num_joints = 24
        half_body_flag = False
        # if self._train and (np.sum(joints_vis_17[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
        if False:
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints_17[:, :, 0], joints_vis_17
            )

            if c_half_body is not None and s_half_body is not None:
                center, scale = c_half_body, s_half_body
                half_body_flag = True

        # rescale
        width_changed = False
        if self.change_widths and self._train and random.random() < 0.67:
        # if self.change_widths and self._train:
            sf = self._scale_factor
            width_changed = True
            # scale_diff = np.array([1.3, 1.4]) + (np.random.rand(2) - 0.5) * 1.2 # 0.8 ~ 2.0
            scale_aspect = (random.random()*0.6 + 0.4) # 0.25 - 1.0
            scale_ratio = np.array([np.sqrt(scale_aspect), 1/np.sqrt(scale_aspect)])
            if random.random() > 0.67:
                scale_aspect = 1 / scale_aspect# / 3 + 2/3 # (1 - 4) / 3
                scale_ratio = np.array([scale_aspect*0.95, 0.95])
            
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf) * scale_ratio
        elif self._train:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        # rotation
        if self._train:
            rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        else:
            r = 0

        if self._train and self._occlusion and not half_body_flag and random.random() < 0.5:
            while True:
                area_min = 0.0
                area_max = 0.3
                synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                ratio_min = 0.5
                ratio_max = 1 / 0.5
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

        joints_17_uvd = gt_joints_17
        joints_29_uvd = gt_joints_29

        joint_cam_17_xyz = joint_cam_17
        joints_cam_29_xyz = joint_cam_29

        if random.random() > 0.75 and self._train:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints_17_uvd = flip_joints_3d(joints_17_uvd, imgwidth, self._joint_pairs_17)
            joints_29_uvd = flip_joints_3d(joints_29_uvd, imgwidth, self._joint_pairs_29)
            joint_cam_17_xyz = flip_cam_xyz_joints_3d(joint_cam_17_xyz, self._joint_pairs_17)
            joints_cam_29_xyz = flip_cam_xyz_joints_3d(joints_cam_29_xyz, self._joint_pairs_29)
            theta = flip_thetas(theta, self._joint_pairs_24)
            twist_phi, twist_weight = flip_twist(twist_phi, twist_weight, self._joint_pairs_24)
            center[0] = imgwidth - center[0] - 1
        
        # width_process
        part_widths, width_ratio_2d, width_ratio, part_spines = self.width_process(
                        beta.copy(), joints_29_uvd[:, :2, 0].copy(), joints_cam_29_xyz.reshape(-1, 3).copy(), r, scale=scale)

        joints_cam_24_xyz = joints_cam_29_xyz[:24]
        # rotate global theta
        theta[0, :3] = rot_aa(theta[0, :3], r)

        theta_rot_mat = batch_rodrigues_numpy(theta).reshape(24 * 9)

        # rotate xyz joints
        joint_cam_17_xyz = rotate_xyz_jts(joint_cam_17_xyz, r)
        joints_17_xyz = joint_cam_17_xyz - joint_cam_17_xyz[:1].copy()
        joints_cam_24_xyz = rotate_xyz_jts(joints_cam_24_xyz, r)
        joints_24_xyz = joints_cam_24_xyz - joints_cam_24_xyz[:1].copy()

        inp_h, inp_w = input_size
        trans = get_affine_transform_rectangle(center, scale, r, [inp_w, inp_h])
        trans_inv = get_affine_transform_rectangle(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
        intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
        joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
        depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        # affine transform
        for i in range(17):
            if joints_17_uvd[i, 0, 1] > 0.0:
                joints_17_uvd[i, 0:2, 0] = affine_transform(joints_17_uvd[i, 0:2, 0], trans)

        for i in range(29):
            if joints_29_uvd[i, 0, 1] > 0.0:
                joints_29_uvd[i, 0:2, 0] = affine_transform(joints_29_uvd[i, 0:2, 0], trans)
        
        target_smpl_weight = torch.ones(1).float() * smpl_weight
        theta_24_weights = np.ones((24, 4)) * smpl_weight

        theta_24_weights = theta_24_weights.reshape(24 * 4)

        # generate training targets
        target_uvd_29, target_weight_29 = self._integral_uvd_target_generator(joints_29_uvd, 29, inp_h, inp_w)
        target_xyz_17, target_weight_17 = self._integral_xyz_target_generator(joints_17_xyz, joints_vis_17, 17)
        target_xyz_24, target_weight_24 = self._integral_xyz_target_generator(joints_24_xyz, joints_vis_29[:24, :], 24)

        target_weight_29 *= joints_vis_29.reshape(-1)
        target_weight_24 *= joints_vis_xyz_29[:24, :].reshape(-1)
        target_weight_17 *= joints_vis_17.reshape(-1)
        bbox = _center_scale_to_box(center, scale)

        tmp_uvd_24 = target_uvd_29.reshape(-1, 3)[:24]
        tmp_uvd_24_weight = target_weight_29.reshape(-1, 3)[:24] * target_weight_24.reshape(-1, 3)

        if self.focal_length > 0:
            cam_scale, cam_trans, cam_valid, cam_error, new_uvd = self.calc_cam_scale_trans2(
                                                            target_xyz_24.reshape(-1, 3).copy(), 
                                                            tmp_uvd_24.copy(), 
                                                            tmp_uvd_24_weight.copy())
        
            target_uvd_29 = (target_uvd_29 * target_weight_29).reshape(-1, 3)
        else:
            cam_scale = 1
            cam_trans = np.zeros(2)
            cam_valid = 0
            cam_error = 0

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        img_center = np.array([float(imgwidth) * 0.5, float(imght) * 0.5])
        target_weight_29 = target_weight_29.reshape(29, 3)
        target_weight_29[:, 2] = 0.0
        target_weight_29 = target_weight_29.reshape(-1)

        target_xyz_29 = np.zeros((29, 3))
        target_xyz_29[:24] = target_xyz_24.reshape(24, 3) * smpl_weight

        target_xyz_weight_29 = np.zeros((29, 3))
        target_xyz_weight_29[:24] = target_weight_24.reshape(24, 3) * smpl_weight
    
        output = {
            'type': '3d_data_w_smpl',
            'image': img,
            'target_theta': torch.from_numpy(theta_rot_mat).float(),
            'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
            'target_beta': torch.from_numpy(beta).float(),
            'target_smpl_weight': target_smpl_weight,
            'target_uvd_29': torch.from_numpy(target_uvd_29.reshape(-1)).float(),
            'target_xyz_29': torch.from_numpy(target_xyz_29.reshape(-1)).float(),
            'target_weight_29': torch.from_numpy(target_weight_29).float(),
            'target_xyz_17': torch.from_numpy(target_xyz_17).float(),
            'target_weight_17': torch.from_numpy(target_weight_17).float(),
            'target_xyz_weight_29': torch.from_numpy(target_xyz_weight_29.reshape(-1)).float(),
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
            'joint_root': torch.from_numpy(joint_root).float(),
            'depth_factor': torch.from_numpy(depth_factor).float(),
            'bbox': torch.Tensor(bbox),
            'target_twist': torch.from_numpy(twist_phi).float(),
            'target_twist_weight': torch.from_numpy(twist_weight).float(),
            'camera_scale': torch.from_numpy(np.array([cam_scale])).float(),
            'camera_trans': torch.from_numpy(cam_trans).float(),
            'camera_valid': cam_valid,
            'camera_error': cam_error,
            'img_center': torch.from_numpy(img_center).float(),
            'part_widths': torch.from_numpy(part_widths).float(),
            'width_ratio': torch.from_numpy(width_ratio).float(),
            'width_changed': width_changed*1.0,
            'width_changed_ratio': scale[1] / scale[0],
            'is_eft': target_smpl_weight
        }

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

    def calc_cam_scale_trans2(self, xyz_29, uvd_29, uvd_weight):

        f = self.focal_length

        # unit: meter
        # the equation to be solved: 
        # u * 256 / f * (z + f/256 * 1/scale) = x + tx
        # v * 256 / f * (z + f/256 * 1/scale) = y + ty

        weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0 # 24 x 1
        # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

        if weight.sum() < 2:
            # print('bad data')
            return 0, np.zeros(2), 0.0, -1, uvd_29

        xyz_29 = xyz_29 * self.depth_factor2meter  # convert to meter
        new_uvd = uvd_29.copy()

        num_joints = len(uvd_29)

        Ax = np.zeros((num_joints, 3))
        Ax[:, 1] = -1
        Ax[:, 0] = uvd_29[:, 0]

        Ay = np.zeros((num_joints, 3))
        Ay[:, 2] = -1
        Ay[:, 0] = uvd_29[:, 1]

        Ax = Ax * weight
        Ay = Ay * weight

        A = np.concatenate([Ax, Ay], axis=0)

        bx = (xyz_29[:, 0] - 256 * uvd_29[:, 0] / f * xyz_29[:, 2]) * weight[:, 0]
        by = (xyz_29[:, 1] - 256 * uvd_29[:, 1] / f * xyz_29[:, 2]) * weight[:, 0]
        b = np.concatenate([bx, by], axis=0)

        A_s = np.dot(A.T, A)
        b_s = np.dot(A.T, b)

        cam_para = np.linalg.solve(A_s, b_s)

        trans = cam_para[1:]
        scale = 1.0 / cam_para[0]

        target_camera = np.zeros(3)
        target_camera[0] = scale
        target_camera[1:] = trans

        backed_projected_xyz = self.back_projection(uvd_29, target_camera, f)
        backed_projected_xyz[:, 2] = backed_projected_xyz[:, 2] * self.depth_factor2meter
        diff = np.sum((backed_projected_xyz-xyz_29)**2, axis=-1) * weight[:, 0]
        diff = np.sqrt(diff).sum() / (weight.sum()+1e-6) * 1000 # roughly mpjpe > 70
        # print(scale, trans, diff)
        if diff < 70:
            new_uvd = self.projection(xyz_29, target_camera, f)
            return scale, trans, 1.0, diff, new_uvd * uvd_weight
        else:
            return scale, trans, 0.0, diff, new_uvd

    def projection(self, xyz, camera, f):
        # xyz: unit: meter, u = f/256 * (x+dx) / (z+dz)
        transl = camera[1:3]
        scale = camera[0]
        z_cam = xyz[:, 2:] + f / (256.0 * scale) # J x 1
        uvd = np.zeros_like(xyz)
        uvd[:, 2] = xyz[:, 2] / self.bbox_3d_shape[2]
        uvd[:, :2] = f / 256.0 * (xyz[:, :2] + transl) / z_cam
        return uvd
    
    def back_projection(self, uvd, pred_camera, focal_length=5000.):
        camScale = pred_camera[:1].reshape(1, -1)
        camTrans = pred_camera[1:].reshape(1, -1)

        camDepth = focal_length / (256 * camScale)

        pred_xyz = np.zeros_like(uvd)
        pred_xyz[:, 2] = uvd[:, 2].copy()
        pred_xyz[:, :2] = (uvd[:, :2] * 256 / focal_length) * (pred_xyz[:, 2:]*self.depth_factor2meter + camDepth) - camTrans

        return pred_xyz

    def width_process(self, beta, uv_2d_24, xyz_3d, rot, scale=None):
        beta_torch = torch.from_numpy(beta).float().reshape(1, 10)
        rot_rad = np.pi * rot / 180.0

        with torch.no_grad():
            smpl_out = self.smpl_layer.get_rest_pose(beta_torch)
            part_widths = self.vertice2capsule_fast(smpl_out.vertices, smpl_out.joints)[0]
        
        # refine part widths in different scale
        part_num = len(self.part_names)
        part_spines = np.zeros((part_num, 2))
        part_spines_3d = np.zeros((part_num, 3))
        for i, k in enumerate(self.part_names):
            part_spine_idx = self.spine_joints[k]
            if part_spine_idx is not None:
                base_joints = uv_2d_24[part_spine_idx[0]], uv_2d_24[part_spine_idx[1]] # batch x 3
                base_joints_3d = xyz_3d[part_spine_idx[0]], xyz_3d[part_spine_idx[1]] # batch x 3
            elif k == 'hips':
                base_joints = uv_2d_24[0], uv_2d_24[[1, 2]].mean(axis=0)
                base_joints_3d = xyz_3d[0], xyz_3d[[1, 2]].mean(axis=0)
            else:
                print(k)
                assert False
        
            part_spines[i] = base_joints[1] - base_joints[0]
            part_spines_3d[i] = base_joints_3d[1] - base_joints_3d[0]

        # change spine to new axis
        rot_mat = np.array(
            [[np.cos(rot_rad), np.sin(rot_rad)],
            [-np.sin(rot_rad), np.cos(rot_rad)]]
        ) # rot back

        part_spines_new = np.einsum('ij,bj->bi', rot_mat, part_spines)
        part_spines_new_norm = np.linalg.norm(part_spines_new, axis=-1) + 1e-5
        cos_a = part_spines_new[:, 0] / part_spines_new_norm
        sin_a = part_spines_new[:, 1] / part_spines_new_norm # L

        # a = (-sx * sin a, sy * cos a), b = (sx * cos a, sy * sin a)
        # width ratio = || a x b || / (|| b ||^2)
        a = np.stack(
            [1/scale[0] * (-2) * sin_a, 1/scale[1]*2*cos_a], axis=-1
        )
        b = np.stack(
            [1/scale[0] * 2 * cos_a, 1/scale[1]*2*sin_a], axis=-1
        )
        a_cross_b_norm = np.abs(np.cross(a, b))
        b_norm = np.linalg.norm(b, axis=-1)
        width_ratio = a_cross_b_norm / (b_norm**2 + 1e-5)

        part_widths_new = width_ratio * part_widths.numpy()

        part_spines_3d = part_spines_3d / self.bbox_3d_shape * 2.2 # unit: meter

        part_widths_ratio = part_widths_new / (np.linalg.norm(part_spines_3d, axis=-1) + 1e-5) # widths / bone_len
        # print('!!!!', part_widths, np.linalg.norm(part_spines_3d, axis=-1), width_ratio)

        return part_widths_new, width_ratio, part_widths_ratio, part_spines_new
         

def _box_to_center_scale_nosquare(x, y, w, h, aspect_ratio=1.0, scale_mult=1.5):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale



'''.eggs/output = {
                # 'target_theta': torch.from_numpy(theta_quat).float(),
                'target_theta': torch.from_numpy(theta_rot_mat).float(),
                'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
                'target_beta': torch.from_numpy(beta).float(),
                'target_uvd_29': torch.from_numpy(target_uvd_29.reshape(-1)).float(),
                'target_xyz_24': torch.from_numpy(target_xyz_24).float(),
                'target_weight_29': torch.from_numpy(target_weight_29).float(),
                'target_weight_24': torch.from_numpy(target_weight_24).float(),
                'target_xyz_17': torch.from_numpy(target_xyz_17).float(),
                'target_weight_17': torch.from_numpy(target_weight_17).float(),
                'target_xyz_weight_24': torch.from_numpy(target_weight_24).float(),
         
                'target_twist': torch.from_numpy(twist_phi).float(),
                'target_twist_weight': torch.from_numpy(twist_weight).float(),
}
'''