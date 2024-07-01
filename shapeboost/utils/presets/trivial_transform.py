import math
import random
import bisect

import cv2
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy, flip_twist,
                          rotmat_to_quat_numpy, rotate_xyz_jts, rot_aa, flip_cam_xyz_joints_3d)
from ..pose_utils import get_intrinsic_metrix


class TrivialTransform3D(object):
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
                 root_idx=0, ambiguous=False, occ_jts_list=None, is_vibe=False, occlusion_synthesize=False):
        # if two_d:
        #     self._joint_pairs = dataset.joint_pairs
        # else:
        #     self._joint_pairs_17 = dataset.joint_pairs_17
        #     self._joint_pairs_24 = dataset.joint_pairs_24
        #     self._joint_pairs_29 = dataset.joint_pairs_29

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self.occlusion_synthesize = occlusion_synthesize
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

        self.focal_length = focal_length
        self.root_idx = root_idx

        self._ambiguous = ambiguous
        self.occ_jts_list = occ_jts_list
        self.occ_interval = 8

        self.is_vibe = is_vibe

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

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

        return img, bbox

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
        if True:
            bbox = list(label['bbox'])

            if self._ambiguous:
                img_idx = label['img_id']
                frame_idx = label['frame_idx']

            imgwidth = src.shape[1] 
            imght = src.shape[0]

            if (np.array(bbox) <= 0).all():
                bbox = 0, 0, imgwidth, imght

            input_size = self._input_size

            xmin, ymin, xmax, ymax = bbox

            ####
            # valid = 1.0
            # xmax_old = (xmax - xmin) / 1.3 + xmin
            # ymax_old = (ymax - ymin) / 1.3 + ymin
            # # if xmax_old > imgwidth * 1.01 and ymax_old > imght * 1.01:
            # #     valid = 0.0
            # # else:
            # #     assert src.shape[0] > 210 or src.shape[1] > 210, label['img_path']
            # if 'the-models' in label['img_path'] and src.shape[0] < 201 or src.shape[1] < 201:
            #     valid = 0.0

            ####
            center, scale = _box_to_center_scale(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)

            xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)
            bbox_new = xmin, ymin, xmax, ymax

            if self._ambiguous:
                uv_29 = label['uv_14']
                # bbox_gt = label['bbox_gt']
                # c_amb, s_amb = self.rand_img_clip_transforms(
                #     img_idx, frame_idx, uv_29, np.ones_like(uv_29), (center, scale)
                # )
                c_amb, s_amb = label['amb_center_scale'][:2], label['amb_center_scale'][2:]

                # print(center, scale)
                if c_amb is not None and s_amb is not None:
                    center, scale = c_amb, s_amb
                    bbox = _center_scale_to_box(center, scale)
            
            # rescale
            if self._train:
                sf = self._scale_factor
                scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            else:
                scale = scale * 1.0

            # rotation
            if self._train:
                rf = self._rot
                # rf = 0 # no rotation when 3d data
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
                # r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
            else:
                r = 0

            if self._ambiguous or (self._train and self._occlusion):
                # synth_xmin, synth_ymin, synth_w, synth_h = self.get_synth_sizes(img_idx, bbox, imgwidth, imght)
                synth_xmin, synth_ymin, synth_xmax, synth_ymax = label['amb_synth_size']
                synth_w = synth_xmax - synth_xmin
                synth_h = synth_ymax - synth_ymin

                synth_xmin, synth_ymin = max(0, int(synth_xmin)), max(0, int(synth_ymin))
                synth_ymax = min(int(synth_ymin + synth_h), imght)
                synth_xmax = min(int(synth_xmin + synth_w), imgwidth)
                src[synth_ymin:synth_ymax, synth_xmin:synth_xmax, :] = np.random.rand(synth_ymax-synth_ymin, synth_xmax-synth_xmin, 3) * 255

            if random.random() > 0.75 and self._train:
            # if False:
            # if True:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]
                center[0] = imgwidth - center[0] - 1

            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
            intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
            joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
            depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

            if self.occlusion_synthesize:
                synth_xmin, synth_ymin, synth_xmax, synth_ymax = label['amb_synth_size']
                synth_w = synth_xmax - synth_xmin
                synth_h = synth_ymax - synth_ymin

                synth_xmin, synth_ymin = max(0, int(synth_xmin)), max(0, int(synth_ymin))
                synth_ymax = min(int(synth_ymin + synth_h), int(inp_h))
                synth_xmax = min(int(synth_xmin + synth_w), int(inp_w))
                img[synth_ymin:synth_ymax, synth_xmin:synth_xmax, :] = np.random.rand(synth_ymax-synth_ymin, synth_xmax-synth_xmin, 3) * 255

            img_center = np.array( [float(imgwidth)*0.5, float(imght)*0.5] )
            # trans2 = get_affine_transform(center, scale, r, [1024, 1024])
            # cropped_img = cv2.warpAffine(src, trans2, (1024, 1024), flags=cv2.INTER_LINEAR)
            # cropped_img = im_to_torch(cropped_img)
            
            target_smpl_weight = torch.ones(1).float()
            theta_24_weights = np.ones((24, 4))

            theta_24_weights = theta_24_weights.reshape(24 * 4)
            
            bbox = _center_scale_to_box(center, scale)

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        img = im_to_torch(img)

        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # mean
        if self.is_vibe:
            img[0].add_(-0.485)
            img[1].add_(-0.456)
            img[2].add_(-0.406)

            # std
            img[0].div_(0.229)
            img[1].div_(0.224)
            img[2].div_(0.225)
        else:
            img[0].add_(-0.406)
            img[1].add_(-0.457)
            img[2].add_(-0.480)

            # std
            img[0].div_(0.225)
            img[1].div_(0.224)
            img[2].div_(0.229)
            
        output = {
            'type': '3d_data_w_smpl',
            # 'cropped_img': cropped_img,
            'image': img,
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
            'joint_root': torch.from_numpy(joint_root).float(),
            'depth_factor': torch.from_numpy(depth_factor).float(),
            'bbox': torch.Tensor(bbox).float(),
            'img_center': torch.from_numpy(img_center).float(),
            'img_height': imght,
            'img_width': imgwidth,
            # 'valid': valid,
            'img_path': label['img_path']
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

    def calc_cam_scale_trans(self, xyz_29, uvd_29, uvd_weight):

        assert np.absolute(xyz_29[:, 2] - uvd_29[:, 2]).sum() < 0.01, f'{xyz_29[:, 2]}, {uvd_29[:, 2]}'

        xy_29 = xyz_29[:, :2].copy()
        uv_29 = uvd_29[:, :2].copy()
        # uv_weight = uvd_weight[:, :2].copy()
        uv_weight = np.ones_like(uvd_weight[:, :2])

        xy_29 = xy_29 * uv_weight
        uv_29 = uv_29 * uv_weight

        xy_29_mean = np.sum(xy_29, axis=0) / uv_weight.sum(axis=0)
        uv_29_mean = np.sum(uv_29, axis=0) / uv_weight.sum(axis=0)

        assert uv_weight.sum() > 2

        if uv_weight.sum() > 2:
            x_29_zero_center = ((xy_29 - xy_29_mean)*uv_weight)[:, 0] 
            u_29_zero_center = ((uv_29 - uv_29_mean)*uv_weight)[:, 0]

            y_29_zero_center = ((xy_29 - xy_29_mean)*uv_weight)[:, 1] 
            v_29_zero_center = ((uv_29 - uv_29_mean)*uv_weight)[:, 1]


            x_var = (x_29_zero_center*x_29_zero_center).sum() / uv_weight[:, 0].sum()
            u_var = (u_29_zero_center*u_29_zero_center).sum() / uv_weight[:, 0].sum()

            y_var = (y_29_zero_center*y_29_zero_center).sum() / uv_weight[:, 1].sum()
            v_var = (v_29_zero_center*v_29_zero_center).sum() / uv_weight[:, 1].sum()

            scale = (np.sqrt(x_var) / np.sqrt(u_var) + np.sqrt(y_var) / np.sqrt(v_var))*0.5
        else:
            print('bug, bug')
            scale = 1
        
        trans = xy_29_mean - uv_29_mean*scale
        # trans = xy_29[0] - uv_29[0]*scale
        
        # assert np.absolute(uv_29 * scale + trans - xy_29).sum() < 0.01, f'{uv_29 * scale + trans}, {xy_29}, {(uv_29 * scale + trans - xy_29)}'

        return scale, trans
    
    def check_uvd_xyz_rel(self, uvd, xyz):
        uvd2 = np.array(uvd[:, :, 0]).reshape(-1, 3).copy()
        xyz2 = np.array(xyz).reshape(-1, 3).copy()
        xyz2 = xyz2 - xyz2[:1].copy()
        # print(xyz2[:, 2] == uvd2[:, 2])

        xy = xyz2[:, :2]
        uv = uvd2[:, :2]
        uv = uv - uv[:1].copy()
        
        # ratio = (xy[1:] / uv[1:]).mean()
        ratio = np.std(xy) / np.std(uv)
        xy_pred = uv * ratio
        diff = np.sum((xy_pred - xy)**2, axis=1)

        print('error', np.sqrt(diff).mean())
        # assert (ratio == ratio[0]).all()

    def projection(self, xyz, camera, f):
        # xyz: unit: meter, u = f/256 * (x+dx) / (z+dz)
        transl = camera[1:3]
        scale = camera[0]
        z_cam = xyz[:, 2:] + f / (256.0 * scale) # J x 1
        uvd = np.zeros_like(xyz)
        uvd[:, 2] = xyz[:, 2] / self.bbox_3d_shape[2]
        uvd[:, :2] = f / 256.0 * (xyz[:, :2] + transl) / z_cam
        return uvd

    def rand_img_clip_transforms(self, img_idx, frame_idx, joints, joints_vis, bbox_origin):
        local_random = random.Random(0)
        if frame_idx % self.occ_interval == 0:
            local_random.seed(img_idx)
            return self.rand_img_clip_transform(joints, joints_vis, bbox_origin, local_random)
        
        interv = frame_idx % self.occ_interval
        last_img_idx = img_idx - interv
        next_img_idx = last_img_idx + self.occ_interval

        local_random.seed(last_img_idx)
        center0, scale0 = self.rand_img_clip_transform(joints, joints_vis, bbox_origin, local_random) 
        # assume the joints location and joints vis does not change much in these frames

        local_random.seed(next_img_idx)
        center1, scale1 = self.rand_img_clip_transform(joints, joints_vis, bbox_origin, local_random)

        interp_coef = (img_idx - last_img_idx)*1.0 / self.occ_interval
        interp_f = lambda x,y: interp_coef*y + (1-interp_coef)*x

        center = interp_f(center0[0], center1[0]), interp_f(center0[1], center1[1])
        scale = interp_f(scale0[0], scale1[0]), interp_f(scale0[1], scale1[1])

        # print(frame_idx, center, frame_idx-interv, center0, occ_idx0, frame_idx-interv+self.occ_interval, center1, occ_idx1)

        return np.array(center), np.array(scale)

    def rand_img_clip_transform(self, joints, joints_vis, bbox_origin, local_random):
        # self.occ_jts_list: list of list of jts indices
        part_joints_vis = [np.array([joints_vis[idx][0] for idx in occ_jts]) for occ_jts in self.occ_jts_list]
        part_jts_is_visible = [item.sum()*1.0 / len(item) > 0.5 for item in part_joints_vis]

        valid_occ_jts_list = []
        for i in range(len(self.occ_jts_list)):
            if part_jts_is_visible[i]:
                valid_occ_jts_list.append(self.occ_jts_list[i])
            
        if len(valid_occ_jts_list) == 0:
            # return None, None
            return bbox_origin

        len_occ = len(valid_occ_jts_list)
        prob_intervals = [0.667/len_occ*(i+1) for i in range(len_occ)]
        p = local_random.random()
        occ_idx = bisect.bisect_right(prob_intervals, p)
        if occ_idx == len_occ:
            # return None, None
            return bbox_origin
        
        selected_joints = []
        for joint_id in range(14):
            if joints_vis[joint_id][0] > 0.5 and (not joint_id in valid_occ_jts_list[occ_idx]):
                selected_joints.append(joints[joint_id])
        
        selected_joints = np.array(selected_joints, dtype=np.float32)
        # print(selected_joints)

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)
        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        center = (right_bottom[0] + left_top[0]) * 0.5, (right_bottom[1] + left_top[1]) * 0.5
        center = np.array(center)
        rand_center_shift = np.array([local_random.random() * 10 - 5, local_random.random() * 10 - 5])
        # rand_center_shift = 0
        center = center + rand_center_shift

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

        scale = scale * (1.1 + local_random.random()*0.2-0.1)
        # scale = scale * (1.1 + random.random()*0.2-0.1)

        return center, scale
    
    def get_synth_sizes(self, img_idx, bbox, imgwidth, imght):
        # interpolation in some frames to get a more continuous occlusion
        if img_idx % self.occ_interval == 0:
            return self.get_synth_size(img_idx, bbox, imgwidth, imght)
        
        last_img_idx = (img_idx // self.occ_interval)*self.occ_interval
        next_img_idx = last_img_idx + self.occ_interval

        synth_xmin0, synth_ymin0, synth_w0, synth_h0 = self.get_synth_size(last_img_idx, bbox, imgwidth, imght)
        synth_xmin1, synth_ymin1, synth_w1, synth_h1 = self.get_synth_size(next_img_idx, bbox, imgwidth, imght)

        interp_coef = (img_idx - last_img_idx) *1.0 / self.occ_interval

        interp_f = lambda x,y: interp_coef*y + (1-interp_coef)*x
        
        return interp_f(synth_xmin0, synth_xmin1), interp_f(synth_ymin0, synth_ymin1), interp_f(synth_w0, synth_w1), interp_f(synth_h0, synth_h1)

    def get_synth_size(self, img_idx, bbox, imgwidth, imght):
        local_random = random.Random(img_idx)
        xmin, ymin, xmax, ymax = bbox
        # print(bbox, imgwidth, imght)
        while True:
            area_min = 0.0
            area_max = 0.3
            synth_area = (local_random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

            ratio_min = 0.5
            ratio_max = 1 / 0.5
            synth_ratio = (local_random.random() * (ratio_max - ratio_min) + ratio_min)

            synth_h = math.sqrt(synth_area * synth_ratio)
            synth_w = math.sqrt(synth_area / synth_ratio)
            synth_xmin = local_random.random() * ((xmax - xmin) - synth_w - 1) + xmin
            synth_ymin = local_random.random() * ((ymax - ymin) - synth_h - 1) + ymin

            if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                synth_xmin = int(synth_xmin)
                synth_ymin = int(synth_ymin)
                synth_w = int(synth_w)
                synth_h = int(synth_h)

                return synth_xmin, synth_ymin, synth_w, synth_h