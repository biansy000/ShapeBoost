import json
import pickle as pk
import os
import random
import torch
import numpy as np
import torch.nn as nn
import cv2
import bisect
import copy

from shapeboost.models.layers.smpl.SMPL import SMPL_layer, to_np, to_tensor

import torch.utils.data as data
from tqdm import tqdm
from shapeboost.beta_decompose.beta_process2 import vertice2capsule_fast, part_names, spine_joints, part_names_ids, \
    used_part_seg, part_root_joints, mean_part_width_ratio
from shapeboost.beta_decompose.beta_process_finer2 import vertice2capsule_finer, part_names_finer, spine_joints_finer, part_names_ids_finer, \
    part_seg_finer, part_root_joints_finer, mean_part_width_ratio_finer
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle


class AmassSimulated(data.Dataset):
    dataset_names = ['ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU', 
        'DFaust_67', 'DanceDB', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D', 'HumanEva', 'KIT', 'MPI_mosh',
        'MPI_HDM05', 'MPI_Limits', 'SFU', 'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap' ]
    
    parents = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11], dtype=torch.long)

    skeleton_29jts = [ 
        [0, 1], [0, 2], [0, 3], # 2
        [1, 4], [2, 5], [3, 6], # 5
        [4, 7], [5, 8], [6, 9], # 8
        [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], # 13
        [12, 15], [13, 16], [14, 17], # 16
        [16, 18], [17, 19], # 18
        [18, 20], [19, 21], # 20
        [20, 22], [21, 23], # 22
        [15, 24], [22, 25], [23, 26], [10, 27], [11, 28] # 27
    ]

    beta_mean = np.array([0.18907155 ,
        0.17565838 ,
        0.23318224 ,
        -0.31987512 ,
        0.093891405 ,
        -0.053046446 ,
        0.3311212 ,
        -0.909195 ,
        0.1416569 ,
        -0.19439596 ,
        -0.14454094 ,
        -0.06638198 ,
        0.04704309 ,
        -0.33996928 ,
        0.61636394 ,
        -0.47443476]) * 0.1

    beta_std = np.array([1.5] * 16)

    def __init__(self, seq_len=1, label_list=None, use_rendering=False, d_len=None, use_finer=False):
        self.pkl_dir = 'data/amass/processed_pose_25fps'
        self.seq_len = seq_len
        
        self.kpt29_error = np.zeros(29)
        self.kpt29_error[[1, 2, 3]] = 15
        self.kpt29_error[[4, 5, 6, 9]] = 40
        self.kpt29_error[[7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]] = 65
        self.kpt29_error[[20, 21, 22, 23, 24, 27, 28]] = 80
        self.kpt29_error[[25, 26]] = 120
        self.kpt29_error = self.kpt29_error.reshape(-1, 1) * 1e-3

        self.xyz_ratio = np.array([0.2, 0.2, 0.4])

        self.cam_mean = np.array([0.63473389, -0.00110506,  0.01689055])
        self.cam_std = np.array([0.07, 0.06, 0.08]) * 2
        self.cam_std[1:] = self.cam_std[1:] * 0.25
        self.scale_min = 0.4

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        if label_list is not None:
            self.label_list = {
                    'beta': label_list['beta'],
                    'theta': label_list['theta'],
                    'twist_phi': label_list['twist_phi'],
                    'twist_weight': label_list['twist_weight']
                } 
            self.label_list_len = len(label_list['beta'])
        else:
            self.files = [os.path.join(self.pkl_dir, f) for f in os.listdir(self.pkl_dir) if f.split('.')[-1] == 'pkl']
            self.files.sort()

            with open("data/amass/amass_betas_whole.pkl", "rb") as f:
                beta_list = pk.load(f)

            self.beta_list = np.array(beta_list)

        self.use_rendering = True
        self._get_render_config()
        self.d_len = d_len

        if use_finer:
            self.vertice2capsule_fast, self.part_names, self.spine_joints, self.part_names_ids = \
                vertice2capsule_finer, part_names_finer, spine_joints_finer, part_names_ids_finer
            self.used_part_seg, self.part_root_joints, self.mean_part_width_ratio = \
                part_seg_finer, part_root_joints_finer, mean_part_width_ratio_finer
        else:
            self.vertice2capsule_fast, self.part_names, self.spine_joints, self.part_names_ids = \
                vertice2capsule_fast, part_names, spine_joints, part_names_ids
            self.used_part_seg, self.part_root_joints, self.mean_part_width_ratio = \
                used_part_seg, part_root_joints, mean_part_width_ratio

    def __getitem__(self, idx):
        target = self._get_item_xyz(idx)

        rendering_inp = self._get_rendering_inp(idx)

        target.update(rendering_inp)
        
        return target
    
    def __len__(self):
        if self.d_len is None:
            return 9999999
        else:
            return self.d_len
    
    def _get_item_xyz(self, idx):
        amass_flag = True
        # if self.label_list is not None and random.random() < 0.5:
        if self.label_list is not None:
            amass_flag = False
            random_indices = [random.randint(0, self.label_list_len-1) for _ in range(self.seq_len)]
            theta = []
            phi = []
            phi_weight = []

            for i in random_indices:
                # i = 0
                label = {
                    k: v[i] for k, v in self.label_list.items()
                }

                theta_raw = label['theta'].reshape(24, 3).copy()
                theta.append(theta_raw)
                phi.append(label['twist_phi'].reshape(23, 2).copy())
                phi_weight.append(label['twist_weight'].reshape(23, 2).copy())
            
            theta = np.stack(theta, axis=0)
            theta = theta * (1 + np.random.randn(self.seq_len, 24, 3) * 0.01)
            phi = np.stack(phi, axis=0)
            phi_weight = np.stack(phi_weight, axis=0)
            theta_weight = np.ones_like(theta)
        else:
            idx = random.randint(0, len(self.files)-1)
            with open(self.files[idx], 'rb') as f:
                file_db = pk.load(f)
            
            file_db_len = len(file_db)

            choosed_indices = [random.randint(0, file_db_len-1) for _ in range(self.seq_len)]
            targets = [file_db[i] for i in choosed_indices]
            
            pose = [target['pose'].reshape(1, 24, 3) for target in targets]
            phi_angle = [target['twist_angle'].reshape(1, 23) for target in targets]

            new_pose = []
            for single_pose in pose:
                new_pose.append(rectify_pose(single_pose.reshape(72)).reshape(1, 24, 3))
            
            pose = np.concatenate(new_pose, axis=0)
            theta = pose
            theta_weight = np.zeros_like(theta)
            theta_weight[:, :20] = 1

            phi_angle = np.concatenate(phi_angle, axis=0)

            phi = np.zeros((self.seq_len, 23, 2))
            phi_weight = np.zeros((self.seq_len, 23, 2))
            phi[:, :, 0] = np.cos(phi_angle)
            phi[:, :, 1] = np.sin(phi_angle)

            phi_weight[:, :, 0] = (phi_angle > -10) * 1.0
            phi_weight[:, :, 1] = (phi_angle > -10) * 1.0

            if np.isnan(phi).any():
                phi[:, :, 0] = 1
                phi[:, :, 1] = 0
                phi_weight[:] = 0

        go_old = axis_angle_to_matrix(torch.from_numpy(theta[:, 0])).reshape(self.seq_len, 3, 3).float()
        # euler: z: rot, y: orientartion, x: camera angle
        delta_go_np = np.random.randn(self.seq_len, 3) * np.array([np.pi/12, np.pi/9, np.pi/9])

        go_mask = (np.random.rand(self.seq_len, 3) > 0.5) * 1.0
        delta_go_np = delta_go_np * go_mask

        delta_go = euler_angles_to_matrix(torch.from_numpy(delta_go_np).float(), convention='XYZ').reshape(self.seq_len, 3, 3)
        go_new = torch.bmm(delta_go, go_old)
        go_new_aa = matrix_to_axis_angle(go_new).reshape(self.seq_len, 3)
        theta[:, 0] = go_new_aa.numpy()

        beta_single = np.random.randn(self.seq_len, 16) * self.beta_std + self.beta_mean
        beta = np.zeros((self.seq_len, 10))
        beta[:, :10] = beta_single[:, :10]
        # beta[:, :10] = beta_single[0, :10]

        pose_torch = torch.from_numpy(theta).float()
        beta_torch = torch.from_numpy(beta).float()
        with torch.no_grad():
            smpl_out = self.smpl_layer(
                pose_axis_angle=pose_torch,
                betas=beta_torch,
                global_orient=None,
                return_29_jts=True
            )

            gt_xyz_29 = smpl_out.joints.numpy()
            gt_xyz_17 = smpl_out.joints_from_verts.numpy()

            transforms = self.smpl_layer.get_global_transform(
                pose=pose_torch,
                betas=beta_torch
            ).numpy()

        beta_weight = np.ones_like(beta)

        gt_xyz_29 = gt_xyz_29 - gt_xyz_29[:, [0], :]
        gt_xyz_17 = gt_xyz_17 - gt_xyz_17[:, [0], :]

        xyz_29_weight = np.ones_like(gt_xyz_29)
        xyz_17_weight = np.ones_like(gt_xyz_17)
        if amass_flag:
            xyz_29_weight[:, [25, 26] ] = 0

        elem_rand_scale = np.random.randn(self.seq_len, gt_xyz_29.shape[1], 3) * self.kpt29_error * self.xyz_ratio
        rand_xyz_29 = gt_xyz_29 + elem_rand_scale

        rand_scale = np.random.rand(self.seq_len).reshape(self.seq_len, 1, 1)
        rand_xyz_29 = rand_xyz_29 * (1+ 0.04*rand_scale - 0.02)

        with torch.no_grad():
            smpl_out_rest = self.smpl_layer.get_rest_pose(beta_torch)
            part_widths = self.vertice2capsule_fast(smpl_out_rest.vertices, smpl_out_rest.joints)
            part_spine, _ = self.get_part_spine(smpl_out_rest.joints)

            error_ratio = torch.clamp(torch.randn(self.seq_len, part_widths.shape[1]), min=-3, max=3) * 0.05
            rand_part_widths = part_widths * (1 + error_ratio)

            part_widths_ratio = part_widths / torch.norm(part_spine, dim=-1)
            pred_part_widths_ratio = rand_part_widths / torch.norm(part_spine, dim=-1)

        theta_rotmat = axis_angle_to_matrix(torch.from_numpy(theta).reshape(-1, 24, 3))
        target = {
                'a_betas': torch.from_numpy(beta).float()[0],
                'a_betas_weight': torch.from_numpy(beta_weight).float()[0],
                'a_theta': theta_rotmat.float().reshape(-1),
                'a_theta_weight': torch.from_numpy(theta_weight.reshape(self.seq_len, 72)).float()[0],
                'a_phi': torch.from_numpy(phi).float()[0],
                'a_phi_weight': torch.from_numpy(phi_weight).float()[0],
                'a_gt_xyz_29': torch.from_numpy(gt_xyz_29).float()[0] / 2.2,
                'a_xyz_29_weight': torch.from_numpy(xyz_29_weight).float()[0],
                'a_gt_xyz_17': torch.from_numpy(gt_xyz_17).float()[0] / 2.2,
                'a_xyz_17_weight': torch.from_numpy(xyz_17_weight).float()[0],
                'a_pred_xyz_29': torch.from_numpy(rand_xyz_29).float()[0] / 2.2,

                'a_pred_part_widths': rand_part_widths.float()[0],
                'a_gt_part_widths': part_widths.float()[0],
                'a_gt_part_spine':part_spine.float()[0],
                'a_part_widths_ratio': part_widths_ratio.float()[0],
            }
        
        target['a_vertices'] = smpl_out.vertices.reshape(-1, 3).float()

        rand_scale_trans = np.random.randn(self.seq_len, 3) * self.cam_std + self.cam_mean
        rand_scale_trans[:, 0] = np.maximum(rand_scale_trans[:, 0], self.scale_min)

        target_trans = np.zeros((self.seq_len, 3))
        target_trans[:, :2] = rand_scale_trans[:, 1:]
        target_trans[:, 2] = 1000.0 / (256.0 * rand_scale_trans[:, 0])

        target['a_scale_trans'] = torch.from_numpy(rand_scale_trans).float()[0]
        target['a_trans_l'] = torch.from_numpy(target_trans).float()[0]

        joint_pos_glob = target_trans.reshape(-1, 1, 3) + transforms[:, :, :3, 3] 
        joint_pos_glob = joint_pos_glob - transforms[:, :1, :3, 3] # if transforms[:, [0], :3, 3], shape=1,4,3?
        joint_view_angle = np.einsum('blji,blj->bli', transforms[:, :, :3, :3], joint_pos_glob)
        joint_view_angle = joint_view_angle / np.linalg.norm(joint_view_angle, axis=-1, keepdims=True)

        target['a_transforms'] = torch.from_numpy(transforms).float()[0]
        target['a_view_angle'] = torch.from_numpy(joint_view_angle).float()[0, self.part_names_ids]

        target_uv = self.project(
            torch.from_numpy(gt_xyz_29).float(), torch.from_numpy(rand_scale_trans)).reshape(29, 2)
        
        pred_rand_scale_trans = rand_scale_trans * (np.random.randn(3) * 0.01 + 1)
        pred_uv = self.project(
            torch.from_numpy(rand_xyz_29).float(), torch.from_numpy(pred_rand_scale_trans)).reshape(29, 2)
        pred_d = torch.from_numpy(rand_xyz_29).float().reshape(29, 3)[:, [2]] / 2.2
        pred_uvd = torch.cat([pred_uv, pred_d], dim=-1)
        target['a_pred_uvd'] = pred_uvd.reshape(29*3)

        target_uv = target_uv.reshape(29, 2)
        target_d = torch.from_numpy(gt_xyz_29).float().reshape(29, 3)[:, [2]] / 2.2
        target_uvd = torch.cat([target_uv, target_d], dim=-1)
        target['a_target_uvd'] = target_uvd.reshape(29*3)

        return target
    
    def _get_render_config(self):
        backgrounds_dir_path = 'data/lsun_backgrounds/train'
        textures_path = 'data/rendering_texture/smpl_train_textures.npz'

        textures = np.load(textures_path)
        self.grey_textures = textures['grey']
        self.nongrey_textures = textures['nongrey']
        self.grey_tex_prob = 0.05

        self.backgrounds_paths = sorted([os.path.join(backgrounds_dir_path, f)
                                         for f in os.listdir(backgrounds_dir_path)
                                         if f.endswith('.webp')])
        self.img_wh = 256
        return
    
    def _get_rendering_inp(self, x):
        sample = {}
        num_samples = self.seq_len
        texture_samples = []
        for _ in range(num_samples):
            if torch.rand(1).item() < self.grey_tex_prob:
                tex_idx = torch.randint(low=0, high=len(self.grey_textures), size=(1,)).item()
                texture = self.grey_textures[tex_idx]
            else:
                tex_idx = torch.randint(low=0, high=len(self.nongrey_textures), size=(1,)).item()
                texture = self.nongrey_textures[tex_idx]
            texture_samples.append(texture)
        
        texture_samples = np.stack(texture_samples, axis=0).squeeze()
        assert texture_samples.shape[-3:] == (1200, 800, 3), "Texture shape is wrong: {}".format(texture_samples.shape)
        sample['a_texture'] = torch.from_numpy(texture_samples / 255.).float()  # (1200, 800, 3) or (num samples, 1200, 800, 3)

        num_samples = self.seq_len
        bg_samples = []
        for _ in range(num_samples):
            bg_idx = torch.randint(low=0, high=len(self.backgrounds_paths), size=(1,)).item()
            bg_path = self.backgrounds_paths[bg_idx]
            background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
            background = cv2.resize(background, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
            background = background.transpose(2, 0, 1)
            bg_samples.append(background)
            
        bg_samples = np.stack(bg_samples, axis=0).squeeze()
        assert bg_samples.shape[-3:] == (3, self.img_wh, self.img_wh), "BG shape is wrong: {}".format(sample['background'].shape)
        sample['a_background'] = torch.from_numpy(bg_samples / 255.).float()

        return sample
    
    def get_part_spine(self, pred_xyz, pred_weight=None):
        part_num = len(self.part_names)
        batch_size = pred_xyz.shape[0]
        if pred_weight is None:
            pred_weight = torch.ones_like(pred_xyz)

        part_spines_3d = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
        part_spine_weight = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
        for i, k in enumerate(self.part_names):
            part_spine_idx = self.spine_joints[k]
            if part_spine_idx is not None:
                base_joints_3d = pred_xyz[:, part_spine_idx[0]], pred_xyz[:, part_spine_idx[1]] # batch x 3
                weight_joints = pred_weight[:, part_spine_idx[0]], pred_weight[:, part_spine_idx[1]]
            else:
                base_joints_3d = pred_xyz[:, 0], pred_xyz[:, [1, 2]].mean(dim=1)
                weight_joints = pred_weight[:, 0], pred_weight[:, 1] * pred_weight[:, 2]

            part_spines_3d[:, i] = base_joints_3d[1] - base_joints_3d[0]
            part_spine_weight[:, i] = weight_joints[1] * weight_joints[0]
        
        return part_spines_3d, part_spine_weight

    def project(self, xyz, scale_trans):
        focal_length = 1000.0
        camDepth = focal_length / (256.0 * scale_trans[:, [0]] + 1e-9)  # batch x 1
        transl = torch.cat([scale_trans[:, 1:3], camDepth], dim=1)
        pred_joints_cam = xyz + transl.reshape(xyz.shape[0], 1, 3)

        pred_keypoints_2d = pred_joints_cam[:, :, :2] / pred_joints_cam[:, :, [2]] * focal_length / 256.0
        return pred_keypoints_2d
    
    def perspective_project(self, xyz, scale_trans, img_center):
        focal_length = 1000.0
        camDepth = focal_length / (256.0 * scale_trans[:, [0]] + 1e-9)  # batch x 1
        transl = torch.cat([scale_trans[:, 1:3], camDepth], dim=1)

        delta_trans_xy = img_center.reshape(-1, 2) / (scale_trans[:, [0]] + 1e-6)
        transl[:, :2] += delta_trans_xy

        # print(delta_trans_xy)

        pred_joints_cam = xyz + transl.reshape(xyz.shape[0], 1, 3)

        pred_keypoints_2d = pred_joints_cam[:, :, :2] / pred_joints_cam[:, :, [2]] * focal_length / 256.0
        return pred_keypoints_2d - img_center.reshape(-1, 2), transl.reshape(-1, 3)
    

def rectify_pose(pose):
    """
    Rectify "upside down" people in global coord
 
    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi*0.5, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    # new_root = R_root.dot(R_mod)
    new_root = R_mod.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose