import math
import random
import os

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, get_affine_transform_rectangle, im_to_torch, batch_rodrigues_numpy, flip_twist,
                          rotmat_to_quat_numpy, rotate_xyz_jts, rot_aa, flip_cam_xyz_joints_3d, get_composed_trans)
from ..pose_utils import get_intrinsic_metrix
from shapeboost.models.layers.smpl.SMPL import SMPL_layer
from shapeboost.beta_decompose.beta_process2 import vertice2capsule_fast, part_names, spine_joints, used_part_seg, part_names_ids
from shapeboost.beta_decompose.beta_process_finer2 import vertice2capsule_finer, part_names_finer, spine_joints_finer, part_seg_finer
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle
from shapeboost.beta_decompose.beta_process_multiview import get_tv_widths_ratio2d, mean_part_width_ratio

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

s_coco_2_h36m_jt = [
    -1,
    -1, 13, 15,
    -1, 14, 16,
    -1, -1,
    0, -1,
    5, 7, 9,
    6, 8, 10
]

s_coco_2_smpl_jt_2d = [
    -1, -1, -1,
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

skeleton_29 = [ 
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), # 5
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), # 11
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), # 17
    (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (15, 24), # 23
    (22, 25), (23, 26), (10, 27), (11, 28) # 27
]


class SimpleTransform3DSMPLMask(object):
    def __init__(self, dataset, scale_factor, color_factor, occlusion, add_dpg,
                 input_size, output_size, depth_dim, bbox_3d_shape,
                 rot, sigma, train, loss_type='MSELoss', scale_mult=1.25, focal_length=1000, two_d=False,
                 root_idx=0, change_widths=True, use_finer=False, use_2dv=False):
        if two_d:
            self._joint_pairs = dataset.joint_pairs
        else:
            self._joint_pairs_17 = dataset.joint_pairs_17
            self._joint_pairs_24 = dataset.joint_pairs_24
            self._joint_pairs_29 = dataset.joint_pairs_29

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
        self.two_d = two_d

        self.depth_factor2meter = self.bbox_3d_shape[2] if self.bbox_3d_shape[2] < 500 else self.bbox_3d_shape[2]*1e-3

        self.focal_length = focal_length
        self.root_idx = root_idx

        self.change_widths = change_widths
        self.use_2dv = use_2dv

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
            self.vertice2capsule_fast, self.part_names, self.spine_joints, self.used_part_seg = \
                vertice2capsule_finer, part_names_finer, spine_joints_finer, part_seg_finer
        else:
            self.vertice2capsule_fast, self.part_names, self.spine_joints, self.used_part_seg = \
                vertice2capsule_fast, part_names, spine_joints, used_part_seg

    def test_transform(self, src, label):
        # print(label['mask_img_path'])
        # # print(label['mask_img_path'], label['is_valid'])
        # assert os.path.exists(label['mask_img_path']), label['mask_img_path']
        mask_img = cv2.cvtColor(cv2.imread(label['mask_img_path']), cv2.COLOR_BGR2RGB)

        bbox = list(label['bbox'])
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        trans_inv = get_affine_transform(center, scale, 0, [inp_w, inp_h], inv=True)
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        # trans: 2x3
        trans_high_res = trans.copy()
        if mask_img.shape[0] > 1500:
            trans_high_res[:, :2] = trans_high_res[:, :2] / 3

        mask_img = cv2.warpAffine(mask_img, trans_high_res, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        mask_img = im_to_torch(mask_img)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        beta = label['beta'].copy()
        theta = label['theta'].copy()

        output = {
            'type': '2d_data',
            'img_path': label['img_path'],
            'image': img,
            'mask_img': mask_img,
            'trans': torch.from_numpy(trans).float(),
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'bbox': torch.Tensor(bbox),
            'width_changed_ratio': scale[1] / scale[0],

            'target_theta': torch.from_numpy(theta).float(),
            'target_beta': torch.from_numpy(beta).float(),
            'cam_params': torch.from_numpy(label['cam_params']).float(),
        }

        return output

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
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        # target_weight[target[:, 0] > 0.5] = 0
        # target_weight[target[:, 0] < -0.5] = 0
        # target_weight[target[:, 1] > 0.5] = 0
        # target_weight[target[:, 1] < -0.5] = 0
        # target_weight[target[:, 2] > 0.5] = 0
        # target_weight[target[:, 2] < -0.5] = 0

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

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        if self.two_d:
            return self._call_2d(src, label)
        else:
            return self._call_3d(src, label)
        
    def _call_2d(self, src, label):
        bbox = list(label['bbox'])
        joint_img = label['joint_img'].copy()
        joints_vis = label['joint_vis'].copy()
        joint_cam = label['joint_cam'].copy()
        self.num_joints = joint_img.shape[0]

        gt_joints = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
        gt_joints[:, :, 0] = joint_img
        gt_joints[:, :, 1] = joints_vis

        imgwidth, imght = label['width'], label['height']
        assert imgwidth == src.shape[1] and imght == src.shape[0]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
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
        if self.change_widths and self._train and random.random() < 0.67:
            sf = self._scale_factor * 0.5
            width_changed = True
            # scale_diff = np.array([1.3, 1.4]) + (np.random.rand(2) - 0.5) * 1.2 # 0.8 ~ 2.0
            scale_aspect = random.random()*0.6 + 0.4 # 0.4 - 1.0
            scale_ratio = np.array([0.95, 0.95/scale_aspect])
            if random.random() > 0.65:
                scale_aspect = 1 / scale_aspect
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
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.5 else 0
        else:
            r = 0
        
        # if self._train:
        #     rf = self._rot
        #     r2 = np.clip(np.random.randn() * rf, -rf * 2-r, rf * 2-r) if random.random() <= 0.3 else 0
        # else:
        #     r2 = 0
        r2 = 0

        if self._train and self._occlusion:
            while True:
                area_min = 0.0
                area_max = 0.7
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
        
        joint_cam = joint_cam.reshape(-1 , 3)
        joints_xyz = joint_cam - joint_cam[[self.root_idx]].copy() # the root index of mpii_3d is 4 !!!
        # print(joints_xyz)

        joints = gt_joints
        if random.random() > 0.7 and self._train:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_3d(joints, imgwidth, self._joint_pairs)
            joints_xyz = flip_xyz_joints_3d(joints_xyz, self._joint_pairs)
            center[0] = imgwidth - center[0] - 1
        
        joints_xyz = rotate_xyz_jts(joints_xyz, r+r2)

        inp_h, inp_w = input_size
        trans1 = get_affine_transform_rectangle(center, scale, r, [inp_w, inp_h])
        trans2 = get_affine_transform(
            center=np.array([128, 128]), scale=np.array([inp_w, inp_h]), rot=r2, output_size=[inp_w, inp_h])
        trans = get_composed_trans(trans1, trans2)
        trans_inv = get_composed_trans(trans1, trans2, inv=True)

        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # deal with joints visibility
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
        joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
        depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

        # generate training targets
        target, target_weight = self._integral_target_generator(joints, self.num_joints, inp_h, inp_w)
        target_xyz, target_xyz_weight = self._integral_xyz_target_generator(joints_xyz, joints_vis, len(joints_vis))

        target_weight *= joints_vis.reshape(-1)
        # target_xyz_weight *= joints_vis.reshape(-1)
        bbox = _center_scale_to_box(center, scale)

        img_center = np.array( [float(imgwidth)*0.5, float(imght)*0.5] )

        cam_scale, cam_trans, cam_valid, cam_error, _ = self.calc_cam_scale_trans2(
                                    target_xyz.reshape(-1, 3).copy(), 
                                    target.reshape(-1, 3).copy(), 
                                    target_weight.reshape(-1, 3).copy())
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
        
        output = {
            'type': '2d_data',
            'image': img,
            'target': torch.from_numpy(target.reshape(-1)).float(),
            'target_weight': torch.from_numpy(target_weight).float(),
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
            'joint_root': torch.from_numpy(joint_root).float(),
            'depth_factor': torch.from_numpy(depth_factor).float(),
            'bbox': torch.Tensor(bbox),
            'camera_scale': torch.from_numpy(np.array([cam_scale])).float(),
            'camera_trans': torch.from_numpy(cam_trans).float(),
            'camera_valid': cam_valid,
            'target_xyz': torch.from_numpy(target_xyz).float(),
            'target_xyz_weight': torch.from_numpy(target_xyz_weight).float(),
            'camera_error': cam_error,
            'img_center': torch.from_numpy(img_center).float(),
            'part_widths': torch.zeros(len(self.part_names)).float(),
            'width_ratio': torch.ones(len(self.part_names)).float()*0.5,
            'width_changed': width_changed*1.0,
            'width_changed_ratio': scale[1] / scale[0]
        }

        return output

    def _call_3d(self, src, label):
        bbox = list(label['bbox'])
        joint_img_17 = label['joint_img_17'].copy()
        joint_relative_17 = label['joint_relative_17'].copy()
        joint_cam_17 = label['joint_cam_17'].copy()
        joints_vis_17 = label['joint_vis_17'].copy()
        joint_img_29 = label['joint_img_29'].copy()
        joint_cam_29 = label['joint_cam_29'].copy()
        joints_vis_29 = label['joint_vis_29'].copy()
        # root_cam = label['root_cam'].copy()
        # root_depth = root_cam[2] / self.bbox_3d_shape[2]
        fx, fy = label['f'].copy()

        beta = label['beta'].copy()
        theta = label['theta'].copy()

        assert not (theta<1e-3).all(), label

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

        # imgwidth, imght = label['width'], label['height']
        # assert imgwidth == src.shape[1] and imght == src.shape[0]
        imgwidth, imght = src.shape[1], src.shape[0]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)

        # half body transform
        self.num_joints = 24
        half_body_flag = False
        if self._train and (np.sum(joints_vis_17[:, 0]) > self.num_joints_half_body and random.random() < self.prob_half_body):
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints_29[:, :, 0], joints_vis_29
            )

            if c_half_body is not None and s_half_body is not None:
                # print(c_half_body, s_half_body, center, scale)
                center, scale = c_half_body, s_half_body
                half_body_flag = True
        
        if (not half_body_flag) and self._train and random.random() < 0.5:
            rand_norm = np.array([random.gauss(mu=0, sigma=1), random.gauss(mu=0, sigma=1)])
            rand_scale_norm = random.gauss(mu=0, sigma=1)

            rand_shift = 0.03 * scale * rand_norm
            rand_scale_shift = 0.05 * scale * rand_scale_norm

            center = center + rand_shift
            scale = scale + rand_scale_shift

        xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

        # rescale
        width_changed = False # if width_changed, do not supervise xyz29 and camera
        if self.change_widths and self._train and random.random() < 0.67:
        # if self.change_widths and self._train:
            sf = self._scale_factor * 0.5
            width_changed = True
            # scale_diff = np.array([1.3, 1.4]) + (np.random.rand(2) - 0.5) * 1.2 # 0.8 ~ 2.0
            scale_aspect = 0.4 + random.random() * 0.6 # 0.25 - 1.0
            scale_ratio = np.array([0.9*np.sqrt(scale_aspect), 0.9/np.sqrt(scale_aspect)])
            if random.random() > 0.67:
                scale_aspect = 1 / scale_aspect # (1 - 4) / 3
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
            # rf = 0 # no rotation when 3d data
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.5 else 0
        else:
            r = 0

        joints_17_uvd = gt_joints_17
        joints_29_uvd = gt_joints_29
        joint_cam_17_xyz = joint_cam_17
        joints_cam_29_xyz = joint_cam_29
        flipped = False
        if random.random() > 0.75 and self._train:
            flipped = True
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

        if self._train and self._occlusion and random.random() > 0.5 and not half_body_flag:
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

        if self._train:
            rf = self._rot
            r2 = np.clip(np.random.randn() * rf * 0.3, -rf * 2-r, rf * 2-r) if random.random() <= 0.3 else 0
        else:
            r2 = 0
        # r2 = 0

        # rotate global theta
        theta[0, :3] = rot_aa(theta[0, :3], r+r2)

        theta_rot_mat = batch_rodrigues_numpy(theta).reshape(24 * 9)
        # theta_quat = rotmat_to_quat_numpy(theta_rot_mat).reshape(24 * 4)

        # rotate xyz joints
        joint_cam_17_xyz = rotate_xyz_jts(joint_cam_17_xyz, r+r2)
        joints_17_xyz = joint_cam_17_xyz - joint_cam_17_xyz[:1].copy()
        joints_cam_29_xyz = rotate_xyz_jts(joints_cam_29_xyz, r+r2)
        joints_29_xyz = joints_cam_29_xyz - joints_cam_29_xyz[:1].copy()

        inp_h, inp_w = input_size
        trans1 = get_affine_transform_rectangle(center, scale, r, [inp_w, inp_h])

        trans2 = get_affine_transform(
            center=np.array([128, 128]), scale=np.array([inp_w, inp_h]), rot=r2, output_size=[inp_w, inp_h])
        
        trans = get_composed_trans(trans1, trans2)
        trans_inv = get_composed_trans(trans1, trans2, inv=True)

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
        
        target_smpl_weight = torch.ones(1).float()
        theta_24_weights = np.ones((24, 4))

        theta_24_weights = theta_24_weights.reshape(24 * 4)

        # generate training targets
        target_uvd_29, target_weight_29 = self._integral_uvd_target_generator(joints_29_uvd, 29, inp_h, inp_w)
        target_xyz_17, target_weight_17 = self._integral_xyz_target_generator(joints_17_xyz, joints_vis_17, 17)
        target_xyz_29, target_xyz_weight_29 = self._integral_xyz_target_generator(joints_29_xyz, joints_vis_29, 29)

        target_weight_29 *= joints_vis_29.reshape(-1)
        target_xyz_weight_29 *= joints_vis_29.reshape(-1)
        target_xyz_weight_29 = target_xyz_weight_29.reshape(-1, 3)
        target_xyz_weight_29[24:] = 0
        target_xyz_weight_29 = target_xyz_weight_29.reshape(-1)

        target_weight_17 *= joints_vis_17.reshape(-1)
        bbox = _center_scale_to_box(center, scale)

        tmp_uvd_24 = target_uvd_29.reshape(-1, 3)[:24]
        tmp_uvd_24_weight = target_weight_29.reshape(-1, 3)[:24]

        if self.focal_length > 0:
            img_center = np.array( [float(imgwidth)*0.5, float(imght)*0.5] )

            cam_scale, cam_trans, cam_valid, cam_error, _ = self.calc_cam_scale_trans2(
                                                            target_xyz_29.reshape(-1, 3).copy()[:24], 
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

        if self.use_2dv and 'is_valid' in label:
            # AGORA
            pred_2dv_weight = 1
            cam_para = label['cam_param']
            smpl_out = self.smpl_layer(
                pose_axis_angle=torch.from_numpy(label['theta'].copy()).unsqueeze(0).float(),
                betas=torch.from_numpy(beta).unsqueeze(0).float(),
                global_orient=None,
                return_29_jts=True
            )
            pred_vertices = smpl_out.vertices.numpy()[0]
            pred_xyz = smpl_out.joints.numpy()[0]

            pred_xyz_homo = np.ones((29, 4))
            pred_xyz_homo[:, :3] = pred_xyz
            pred_vertices_homo = np.ones((6890, 4))
            pred_vertices_homo[:, :3] = pred_vertices

            pred_uv_homo = np.einsum('ij,bj->bi', cam_para, pred_xyz_homo)
            pred_uv = pred_uv_homo[:, :2] / pred_uv_homo[:, [2]]
            assert np.absolute(pred_uv - joint_img_29[:, :2]).mean() < 0.1, (pred_uv, joint_img_29[:, :2])

            pred_2dv_homo = np.einsum('ij,bj->bi', cam_para, pred_vertices_homo)
            pred_2dv_homo[:, :2] = pred_2dv_homo[:, :2] / pred_2dv_homo[:, [2]]
            pred_2dv_homo[:, [2]] = 1 # normalize

            if flipped:
                # TODO 
                pred_2dv_weight = 0
            
            pred_2dv = np.einsum('ij,bj->bi', trans, pred_2dv_homo)
            pred_2dv = pred_2dv / 256.0 - 0.5
        elif self.use_2dv:
            # use approximated camera to calculate 2dv
            # assert not self.use_2dv # tmp
            pred_2dv_weight = 1
            smpl_out = self.smpl_layer(
                pose_axis_angle=torch.from_numpy(label['theta'].copy()).unsqueeze(0).float(),
                betas=torch.from_numpy(beta).unsqueeze(0).float(),
                global_orient=None,
                return_29_jts=True
            )
            pred_vertices = smpl_out.vertices.numpy()[0]
            # cam_scale_rawimg, cam_trans_rawimg
            v_abs_d = pred_vertices[:, [2]] + 1000 / (256.0 * cam_scale_rawimg)
            pred_2dv = (pred_vertices[:, :2] + cam_trans_rawimg.reshape(1, 2)) / v_abs_d / (256.0 / 1000.0)
            # pred_2dv = (pred_2dv + 0.5) * np.array([imgwidth, imght])
            pred_2dv = (pred_2dv + np.array([0.5, 0.5*imght/imgwidth])) * np.array([imgwidth, imgwidth])

            if flipped:
                # TODO 
                pred_2dv_weight = 0
            
            pred_2dv_homo = np.ones((6890, 3))
            pred_2dv_homo[:, :2] = pred_2dv
            pred_2dv = np.einsum('ij,bj->bi', trans, pred_2dv_homo)
            pred_2dv = pred_2dv / 256.0 - 0.5

        if 'mask_path' in label:
            mask_path = label['mask_path']
            mask_img = cv2.imread(mask_path)

            height, width = mask_img.shape[0], mask_img.shape[1]
            x_part = np.zeros((height, width, 22), dtype=np.uint8)

            # clothes
            x_part[:, :, 21] = (mask_img[:, :, 0] > 0.1) * 1.0 - (mask_img[:, :, 2] > 0.1) * 1.0
            x_part[:, :, 21] = (x_part[:, :, 21] > 0.1) * 1.0
            
            for i, name_id in enumerate(part_names_ids):
                part_seg = (mask_img[:, :, 1] == (name_id+1) * 10)
                x_part[:, :, i+1] = part_seg

            trans_mask = label['mask_trans_inv']
            # trans_mask = get_composed_trans(trans_mask, trans)

            # mask_img_part = cv2.warpAffine(x_part, trans_mask, (64, 64), flags=cv2.INTER_LINEAR)
            mask_img_part = cv2.warpAffine(x_part, trans_mask, (imgwidth, imght), flags=cv2.INTER_LINEAR)
            if flipped:
                mask_img_part = mask_img_part[:, ::-1, :]
            mask_img_part = cv2.warpAffine(mask_img_part, trans, (256, 256), flags=cv2.INTER_LINEAR)
            mask_img_part = cv2.resize(mask_img_part, (64, 64), interpolation=cv2.INTER_LINEAR)
            
            # background
            mask_img_0 = cv2.warpAffine(mask_img, trans_mask, (imgwidth, imght), flags=cv2.INTER_LINEAR)
            if flipped:
                mask_img_0 = mask_img_0[:, ::-1, :]

            mask_img_0 = cv2.warpAffine(mask_img_0, trans, (256, 256), flags=cv2.INTER_LINEAR)
            mask_img_0 = cv2.resize(mask_img_0, (64, 64), interpolation=cv2.INTER_LINEAR)
            mask_img_part[:, :, 0] = (mask_img_0[:, :, 0] < 0.1) * 1.0
            # x_part[:, :, 0] = (mask_img[:, :, 0] < 0.1) * 1.0
            
            mask_img_part = torch.from_numpy(mask_img_part).float()
            mask_img_part = mask_img_part.permute(2, 0, 1)
            mask_part_weight = torch.ones(22)
            
        output = {
            'type': '3d_data_w_smpl',
            'image': img,
            'target_theta': torch.from_numpy(theta_rot_mat).float(),
            'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
            'target_beta': torch.from_numpy(beta).float(),
            'target_smpl_weight': target_smpl_weight,
            'target_uvd_29': torch.from_numpy(target_uvd_29.reshape(-1)).float(),
            'target_xyz_29': torch.from_numpy(target_xyz_29).float(),
            'target_weight_29': torch.from_numpy(target_weight_29).float(),
            'target_xyz_weight_29': torch.from_numpy(target_xyz_weight_29).float(),
            'target_xyz_17': torch.from_numpy(target_xyz_17).float(),
            'target_weight_17': torch.from_numpy(target_weight_17).float()*(1-width_changed*1.0),
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
            'mask_img_part': mask_img_part,
            'mask_part_weight': mask_part_weight,
        }  

        if self.use_2dv:
            output['pred_2dv'] = torch.from_numpy(pred_2dv).float()
            output['pred_2dv_weight'] = pred_2dv_weight

        if 'clothes_label' in label:
            # output['clothes_label'] = torch.from_numpy(label['clothes_label']).float()
            # output['clothes_weight'] = label['clothes_weight']
            
            upper_body_parts = [
                'pelvis', 'spine1', 'spine2', 'spine3', 'left_shoulder', 'right_shoulder'
            ]
            upper_body_indices = [part_names.index(n) for n in upper_body_parts]
            lower_body_parts = [
                'left_hip', 'right_hip', 'left_knee', 'right_knee'
            ]
            lower_body_indices = [part_names.index(n) for n in lower_body_parts]

            upper_thin, upper_thick = 1.25, 2.5
            lower_thin, lower_thick = 0.9, 2.5

            clothes_widths = label['clothes_label']
            upper_width = clothes_widths[upper_body_indices].mean()
            lower_width = clothes_widths[lower_body_indices].mean()

            label_onehot = np.zeros((2, 3))
            if upper_width > upper_thick:
                label_onehot[0, 2] = 1
            elif upper_width < upper_thin:
                label_onehot[0, 0] = 1
            else:
                label_onehot[0, 1] = 1
            
            if lower_width > lower_thick:
                label_onehot[1, 2] = 1
            elif lower_width < lower_thin:
                label_onehot[1, 0] = 1
            else:
                label_onehot[1, 1] = 1

            output['clothes_label'] = torch.from_numpy(label_onehot).float()
            output['clothes_widths'] = torch.from_numpy(clothes_widths).float()
            output['clothes_weight'] = float(label['clothes_weight'])
            
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

        # do not take into account 2.2
        # the equation to be solved: 
        # u * 256 / f * (z + f/256 * 1/scale) = x + tx
        # v * 256 / f * (z + f/256 * 1/scale) = y + ty

        weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0 # 24 x 1
        # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

        if weight.sum() < 2:
            # print('bad data')
            return 1, np.zeros(2), 0.0, -1, uvd_29

        xyz_29 = xyz_29 * 2.2 # convert to meter
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

        backed_projected_xyz = back_projection(uvd_29, target_camera, f)
        backed_projected_xyz[:, 2] = backed_projected_xyz[:, 2] * 2.2
        diff = np.sum((backed_projected_xyz-xyz_29)**2, axis=-1) * weight[:, 0]
        diff = np.sqrt(diff).sum() / (weight.sum()+1e-6) * 1000 # roughly mpjpe > 70
        # print(scale, trans, diff)
        if diff < 70:
            new_uvd = self.projection(xyz_29, target_camera, f)
            return scale, trans, 1.0, diff, new_uvd * uvd_weight
        else:
            return scale, trans, 0.0, diff, new_uvd
    
    def calc_cam_scale_trans_refined(self, xyz_29, uv_29, uvd_weight, img_center):

        # the equation to be solved: 
        # u_256 / f * (1-cx/u) * (z + tz) = x + tx 
        #   -> (u - cx) * (z * 1/f + tz/f) = x + tx
        #   
        # v_256 / f * (1-cy/v) * (z + tz) = y + ty

        # calculate: tz/f, tx, ty
        # return scale, [tx, ty], is_valid, error, None

        weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0 # 24 x 1
        # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

        xyz_29 = xyz_29 * 2.2
        uv_29_fullsize = uv_29[:, :2] * 256.0
        uv_c_diff = uv_29_fullsize - img_center

        if weight.sum() <= 2:
            # print('bad data')
            return 1, np.zeros(2), 0.0, -1, None

        num_joints = len(uv_29)

        Ax = np.zeros((num_joints, 3))
        Ax[:, 0] = uv_c_diff[:, 0]
        Ax[:, 1] = -1

        Ay = np.zeros((num_joints, 3))
        Ay[:, 0] = uv_c_diff[:, 1]
        Ay[:, 2] = -1

        Ax = Ax * weight
        Ay = Ay * weight

        A = np.concatenate([Ax, Ay], axis=0)

        bx = (xyz_29[:, 0] - uv_c_diff[:, 0] * xyz_29[:, 2] / 1000.0) * weight[:, 0]
        by = (xyz_29[:, 1] - uv_c_diff[:, 1] * xyz_29[:, 2] / 1000.0) * weight[:, 0]
        b = np.concatenate([bx, by], axis=0)

        A_s = np.dot(A.T, A)
        b_s = np.dot(A.T, b)

        cam_para = np.linalg.solve(A_s, b_s)

        # f_estimated = 1.0 / cam_para[0]
        f_estimated = 1000.0
        tz = cam_para[0] * f_estimated
        tx, ty = cam_para[1:]

        target_camera = np.zeros(4)
        target_camera[0] = f_estimated
        target_camera[1:] = np.array([tx, ty, tz])

        backed_projected_xyz = back_projection_matrix(uv_29_fullsize, xyz_29, target_camera, img_center)
        diff = np.sum((backed_projected_xyz-xyz_29)**2, axis=-1) * weight[:, 0]
        diff = np.sqrt(diff).sum() / (weight.sum()+1e-6) * 1000 # roughly mpjpe > 70

        out = np.zeros(3)
        out[1:] = cam_para[1:]
        out[0] = 1000.0 / 256.0 / tz

        if diff < 60:
            return out[0], out[1:], 1.0, diff, None
        else:
            return out[0], out[1:], 0.0, diff, None

    def projection(self, xyz, camera, f):
        # xyz: unit: meter, u = f/256 * (x+dx) / (z+dz)
        transl = camera[1:3]
        scale = camera[0]
        z_cam = xyz[:, 2:] + f / (256.0 * scale) # J x 1
        uvd = np.zeros_like(xyz)
        uvd[:, 2] = xyz[:, 2] / self.bbox_3d_shape[2]
        uvd[:, :2] = f / 256.0 * (xyz[:, :2] + transl) / z_cam
        return uvd

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
        part_spines_new_norm = np.linalg.norm(part_spines_new, axis=-1)
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
        width_ratio = a_cross_b_norm / (b_norm**2)

        part_widths_new = width_ratio * part_widths.numpy()
        part_spines_3d = part_spines_3d / self.bbox_3d_shape * 2.2 # unit: meter
        part_widths_ratio = part_widths_new / np.linalg.norm(part_spines_3d, axis=-1) # widths / bone_len

        return part_widths_new, width_ratio, part_widths_ratio, part_spines_new
            
    def get_spine_widths_ratio(self, widths_ratio_raw, uv_raw, uv_new):
        part_num = len(self.part_names)
        part_spines = np.zeros((part_num, 2))
        part_spines_new = np.zeros((part_num, 2))
        for i, k in enumerate(self.part_names):
            part_spine_idx = self.spine_joints[k]
            if part_spine_idx is not None:
                base_joints = uv_raw[part_spine_idx[0]], uv_raw[part_spine_idx[1]] # batch x 3
                base_joints_new = uv_new[part_spine_idx[0]], uv_new[part_spine_idx[1]] # batch x 3
                
            elif k == 'hips':
                base_joints = uv_raw[0], uv_raw[[1, 2]].mean(axis=0)
                base_joints_new = uv_new[0], uv_new[[1, 2]].mean(axis=0)
                
            else:
                print(k)
                assert False
            
            part_spines_new[i] = base_joints_new[1] - base_joints_new[0]
            part_spines[i] = base_joints[1] - base_joints[0]
        
        # print('part_spines_new', part_spines_new/part_spines_raw)
        # spine_2d_ratio = np.linalg.norm(part_spines_new, axis=-1) / np.linalg.norm(part_spines_raw, axis=-1)
        spine_2d_ratio = np.linalg.norm(part_spines_new, axis=-1) / np.linalg.norm(part_spines, axis=-1)
        print('spine_2d_ratio', spine_2d_ratio)
    
    def width_process_from_vertices(self, beta, pose, uv_2d_24, xyz_3d, rot, scale, proj_func):
        beta_torch = torch.from_numpy(beta).float().reshape(1, 10)
        rot_rad = np.pi * rot / 180.0
        scale_tmp = np.array([scale[1], scale[0]]) / 256.0

        with torch.no_grad():
            smpl_out_rest = self.smpl_layer.get_rest_pose(beta_torch)
            part_widths = self.vertice2capsule_fast(smpl_out_rest.vertices, smpl_out_rest.joints)[0]
            smpl_out = self.smpl_layer(
                pose_axis_angle=torch.from_numpy(pose).float().reshape(1, 24, 3),
                betas=beta_torch,
                global_orient=None
            )

            smpl_vertices = smpl_out.vertices

            proj_uv = proj_func(xyz_3d)

            proj_uv_normed = proj_uv[:24]
            proj_uv_normed = proj_uv_normed - proj_uv_normed.mean(axis=0)
            proj_uv_normed = proj_uv_normed / np.std(proj_uv_normed, axis=0)

            uv_2d_24_normed = uv_2d_24[:24]
            uv_2d_24_normed = uv_2d_24_normed - uv_2d_24_normed.mean(axis=0)
            uv_2d_24_normed = uv_2d_24_normed / np.std(uv_2d_24_normed, axis=0)

            assert np.absolute(proj_uv_normed[:24] - uv_2d_24_normed[:24]).max() < 1e-3, f'{proj_uv_normed}, {uv_2d_24_normed}'

            smpl_vertices_uv = proj_func(smpl_vertices[0].numpy())
        
        # change spine to new axis
        rot_mat = np.array(
            [[np.cos(rot_rad), np.sin(rot_rad)],
            [-np.sin(rot_rad), np.cos(rot_rad)]]
        ) # rot back

        # refine part widths in different scale
        part_num = len(self.part_names)
        part_spines = np.zeros((part_num, 2))
        part_spines_3d = np.zeros((part_num, 3))
        width_ratio_list = np.zeros((part_num))
        for i, k in enumerate(self.part_names):
            vid = self.used_part_seg[k]
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

            root_joint = part_root_joints[k]
            part_lbs_weight = self.smpl_layer.lbs_weights[vid, root_joint].numpy()
            # print('part_lbs_weight', k, part_lbs_weight.mean(), part_lbs_weight.max(), part_lbs_weight.min())
            part_lbs_weight_sum = part_lbs_weight.sum()
        
            part_spines[i] = base_joints[1] - base_joints[0]
            part_spines_3d[i] = base_joints_3d[1] - base_joints_3d[0]

            v_uv = smpl_vertices_uv[vid]

            part_spine_norm = np.linalg.norm(part_spines[i])
            v_uv_rel = v_uv - base_joints[0]
            width_2d_old = v_uv_rel[:, 0] * part_spines[i, 1] - v_uv_rel[:, 1] * part_spines[i, 0]
            width_2d_old = np.absolute(width_2d_old) / part_spine_norm
            widths_ratio_old = (width_2d_old / part_spine_norm) * part_lbs_weight
            widths_ratio_old = widths_ratio_old.sum() / part_lbs_weight_sum

            part_spines_new = np.matmul(rot_mat, part_spines[i])
            v_uv_rel_new = np.einsum('ij,nj->ni', rot_mat, v_uv_rel)

            part_spines_new = part_spines_new * scale_tmp
            v_uv_rel_new = v_uv_rel_new * scale_tmp
            
            part_spine_new_norm = np.linalg.norm(part_spines_new)
            
            # print(v_uv_new_rel.shape, part_spines_new.shape)
            width_2d_new = v_uv_rel_new[:, 0] * part_spines_new[1] - v_uv_rel_new[:, 1] * part_spines_new[0]
            width_2d_new = np.absolute(width_2d_new) / part_spine_new_norm
            widths_ratio_new = (width_2d_new / part_spine_new_norm) * part_lbs_weight 
            widths_ratio_new = widths_ratio_new.sum() / part_lbs_weight_sum

            width_ratio = widths_ratio_new / widths_ratio_old
            # width_ratio = widths_ratio_new
            width_ratio_list[i] = width_ratio

            # print('???', rot, width_2d_old.mean(), part_spine_norm, width_2d_new.mean(), part_spine_new_norm)

        part_widths_new = width_ratio_list * part_widths.numpy()
        part_spines_3d = part_spines_3d / self.bbox_3d_shape * 2.2 # unit: meter

        part_widths_ratio = part_widths_new / np.linalg.norm(part_spines_3d, axis=-1) # widths / bone_len

        return part_widths_new, width_ratio_list, part_widths_ratio

    def get_width_ratio_from2d(self, beta_out, transl_glob, transforms):
        rest_out = self.smpl_layer.get_rest_pose(beta_out)
        template_out = self.smpl_layer.get_rest_pose(torch.zeros_like(beta_out))

        rest_v, rest_j = rest_out['vertices'], rest_out['joints']
        temp_v, temp_j = template_out['vertices'], template_out['joints']
        batch_size = rest_v.shape[0]
        lbs_weights = self.smpl_layer.lbs_weights

        inp_v = torch.cat([rest_v, temp_v], dim=0)
        inp_j = torch.cat([rest_j, temp_j], dim=0)

        transl_glob = torch.cat([transl_glob, transl_glob], dim=0)
        transforms = torch.cat([transforms, transforms], dim=0)

        tv_wr_2d = get_tv_widths_ratio2d(
            inp_v.reshape(batch_size*2, -1, 3), 
            inp_j.reshape(batch_size*2, -1, 3), 
            lbs_weights, 
            transl_glob.reshape(batch_size*2, 3),
            transforms.reshape(batch_size*2, 24, 4, 4)
        ).reshape(batch_size*2, -1)
        
        ratio_2d = tv_wr_2d[:batch_size] / tv_wr_2d[batch_size:]

        width_ratio_from2d = ratio_2d * torch.tensor(mean_part_width_ratio)
        return width_ratio_from2d


def back_projection(uvd, pred_camera, focal_length=5000.):
    camScale = pred_camera[:1].reshape(1, -1)
    camTrans = pred_camera[1:].reshape(1, -1)

    camDepth = focal_length / (256 * camScale)

    pred_xyz = np.zeros_like(uvd)
    pred_xyz[:, 2] = uvd[:, 2].copy()
    pred_xyz[:, :2] = (uvd[:, :2] * 256 / focal_length) * (pred_xyz[:, 2:]*2.2 + camDepth) - camTrans

    return pred_xyz


def back_projection_matrix(uv, xyz, pred_camera, img_center):
    # pred_camera: f, tx, ty, tz
    f, tx, ty, tz = pred_camera
    cx, cy = img_center
    intrinsic_inv = np.array(
        [[1/f, 0, -cx / f],
        [0, 1/f, -cy/f],
        [0, 0, 1]]
    )

    uv_homo = np.ones((len(uv), 3))
    uv_homo[:, :2] = uv

    xyz_cam = np.matmul(uv_homo, intrinsic_inv.T) # 29 x 3
    abs_z = xyz[:, [2]] + tz # 29 x 1
    xyz_cam = xyz_cam * abs_z

    pred_xyz = xyz_cam - pred_camera[1:]

    return pred_xyz