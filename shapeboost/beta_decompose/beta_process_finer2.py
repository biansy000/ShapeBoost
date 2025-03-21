import os
import json

import numpy as np
import torch
import copy
import bisect

from shapeboost.beta_decompose.beta_process2 import *
from shapeboost.beta_decompose.beta_process_finer2_import import *
from shapeboost.models.layers.smpl.SMPL import SMPL_layer
import pickle as pk
from easydict import EasyDict as edict


def split_vertices_on_spine(vertices, joints24, split_num=2, part_names_used=part_names):

    finer_part_names_list = [ 
        [f'{part_name}_{k}' for k in range(split_num)] for part_name in part_names_used
    ]

    finer_part_names = []
    for item in finer_part_names_list:
        finer_part_names += item
    
    finer_part_seg = {
        finer_part_name: [] for finer_part_name in finer_part_names
    }

    for k in part_names_used:
        v_ids = part_seg[k]
        part_spine_idx = spine_joints[k]
        if part_spine_idx is not None:
            part_spine = - joints24[:, part_spine_idx[0]] + joints24[:, part_spine_idx[1]] # batch x 3
            base_joint = joints24[:, part_spine_idx[0]]
        elif k == 'hips':
            part_spine = - joints24[:, 0] + joints24[:, [1, 2]].mean(dim=1)
            base_joint = joints24[:, 0]
        else:
            print(k)
            assert False
        
        part_vertices = vertices[:, v_ids]
        part_spine_len = torch.norm(part_spine, dim=-1, keepdim=True) # batch x 1
        part_spine_normed = part_spine / part_spine_len # batch x 3
        part_spine_normed = part_spine_normed.unsqueeze(dim=1)
        vec = part_vertices - base_joint.unsqueeze(1)

        inner_prod = (part_spine_normed * vec).sum(dim=-1)

        distance_ratio = inner_prod / part_spine_len

        distance_ratio = distance_ratio[0]
        dis_sorted, _ = torch.sort(distance_ratio)
        len_part = len(distance_ratio)

        bound_inds = [(len_part * i) // split_num for i in range(1, split_num)]
        boundaries = torch.tensor([dis_sorted[idx] for idx in bound_inds])
        bin_idx = torch.bucketize(distance_ratio, boundaries)

        for i in range(split_num):
            indices = (bin_idx == i).nonzero(as_tuple=True)[0]
            finer_part_seg[f'{k}_{i}'] = [v_ids[i] for i in indices]
            print(f'{k}_{i}', len(indices))

    return finer_part_names, finer_part_seg


def vertice2capsule_finer(vertices, joints24, part_names_finer, part_seg_finer, lbs_weights=None):
    # vertices: batch x 6980 x 3, joints24: batch x 24 x 3
    # print(vertices.shape, joints24.shape)
    capsule_dict = []
    if lbs_weights is None:
        lbs_weights = default_lbs_weights.to(vertices.device).clone()

    for part_name_finer in part_names_finer:
        v = part_seg_finer[part_name_finer]
        k = part_name_finer.split('_')[:-1]
        k = '_'.join(k)
        # print(k, part_root_joints[k])

        root_joint = part_root_joints[k]
        part_lbs_weights = lbs_weights[v, root_joint] # l
        part_lbs_weights[part_lbs_weights<0.5] = 0

        part_spine_idx = spine_joints[k]
        if part_spine_idx is not None:
            part_spine = joints24[:, part_spine_idx[0]] - joints24[:, part_spine_idx[1]] # batch x 3
            base_joint = joints24[:, part_spine_idx[0]]
        elif k == 'hips':
            part_spine = joints24[:, 0] - joints24[:, [1, 2]].mean(dim=1)
            base_joint = joints24[:, 0]
        else:
            print(k)
            assert False

        part_vertices = vertices[:, v] # batch x L x 3
        part_spine_normed = part_spine / torch.norm(part_spine, dim=-1, keepdim=True) # batch x 3
        part_spine_normed = part_spine_normed.unsqueeze(dim=1)

        vec = part_vertices - base_joint.unsqueeze(1)
        dis2spine = torch.cross(part_spine_normed.expand(-1, vec.shape[1], -1), vec, dim=-1)
        assert dis2spine.shape[-1] == 3

        dis2spine = torch.norm(dis2spine, dim=-1)
        mean_dis2spine = (dis2spine * part_lbs_weights).sum(dim=-1, keepdims=True) / part_lbs_weights.sum()
        capsule_dict.append(mean_dis2spine)
    
    capsule_dict = torch.cat(capsule_dict, dim=1)
    return capsule_dict


def get_part_spine_finer(pred_xyz, pred_weight, part_names_finer):
    part_num = len(part_names_finer)
    batch_size = pred_xyz.shape[0]
    if pred_weight is None:
        pred_weight = torch.ones_like(pred_xyz)

    part_spines_3d = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
    part_spine_weight = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
    for i, part_name_finer in enumerate(part_names_finer):

        k = part_name_finer.split('_')[:-1]
        k = '_'.join(k)
        part_spine_idx = spine_joints[k]

        if part_spine_idx is not None:
            base_joints_3d = pred_xyz[:, part_spine_idx[0]], pred_xyz[:, part_spine_idx[1]] # batch x 3
            weight_joints = pred_weight[:, part_spine_idx[0]], pred_weight[:, part_spine_idx[1]]
        else:
            base_joints_3d = pred_xyz[:, 0], pred_xyz[:, [1, 2]].mean(dim=1)
            weight_joints = pred_weight[:, 0], pred_weight[:, 1] * pred_weight[:, 2]

        part_spines_3d[:, i] = base_joints_3d[1] - base_joints_3d[0]
        part_spine_weight[:, i] = weight_joints[1] * weight_joints[0]
    
    return part_spines_3d, part_spine_weight


def get_all_info(part_seg_finer, part_names_finer, smpl_layer):

    default_beta = torch.zeros(1, 10)
    default_smpl_out = smpl_layer.get_rest_pose(default_beta)
    default_capsule = vertice2capsule_finer(
        default_smpl_out.vertices, default_smpl_out.joints, lbs_weights=smpl_layer.lbs_weights,
        part_names_finer=part_names_finer, part_seg_finer=part_seg_finer)
    
    # print('default_capsule', default_capsule)
    part_spines, _ = get_part_spine_finer(default_smpl_out.joints, pred_weight=None, part_names_finer=part_names_finer)

    part_root_joints_finer = {}
    spine_joints_finer = {}
    mean_part_width_ratio_dict_finer = {}
    mean_part_width_dict_finer = {}
    part_seg_lens_finer = []
    part_pairs_finer = []
    part_names_ids_finer = []
    for i, part_name_finer in enumerate(part_names_finer):
        v = part_seg_finer[part_name_finer]

        k = part_name_finer.split('_')[:-1]
        k = '_'.join(k)

        part_root_joints_finer[part_name_finer] = part_root_joints[k]
        spine_joints_finer[part_name_finer] = spine_joints[k]

        part_seg_lens_finer.append(len(v))
        part_names_ids_finer.append(joints_name_24.index(k))
        if 'left' in part_name_finer:
            part_pairs_finer.append(
                (part_name_finer, part_name_finer.replace('left', 'right'))
            )

        mean_width = float(default_capsule[0, i].numpy())
        mean_ratio = float(mean_width / part_spines[0, i].norm(dim=-1).numpy())

        print(
            f"'{part_name_finer}': {mean_ratio}"
        )

        mean_part_width_ratio_dict_finer[part_name_finer] = mean_ratio
        mean_part_width_dict_finer[part_name_finer] = mean_width
    
    return part_root_joints_finer, spine_joints_finer, mean_part_width_ratio_dict_finer, part_seg_lens_finer, part_pairs_finer, part_names_ids_finer, mean_part_width_dict_finer


if __name__ == "__main__":
    h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
    smpl_layer = SMPL_layer(
        './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
        h36m_jregressor=h36m_jregressor,
        dtype=torch.float32
    )

    output = smpl_layer.get_rest_pose(torch.zeros(1, 10).float())

    split_num = 3
    finer_part_names, finer_part_seg = split_vertices_on_spine(output.vertices, output.joints, part_names_used=joints_name_24, split_num=split_num)

    part_root_joints_finer, spine_joints_finer, mean_part_width_ratio_dict_finer, part_seg_lens_finer, part_pairs_finer, part_names_ids_finer, mean_part_width_dict_finer = \
        get_all_info(finer_part_seg, finer_part_names, smpl_layer)

    all_info = edict(
        finer_part_names=finer_part_names, 
        finer_part_seg=finer_part_seg, 
        part_root_joints_finer=part_root_joints_finer, 
        spine_joints_finer=spine_joints_finer, 
        mean_part_width_ratio_dict_finer=mean_part_width_ratio_dict_finer, 
        part_seg_lens_finer=part_seg_lens_finer, 
        part_pairs_finer=part_pairs_finer, 
        part_names_ids_finer=part_names_ids_finer,
        mean_part_width_dict_finer=mean_part_width_dict_finer
    )
    
    with open(f'shapeboost/beta_decompose/beta_process_finer2_split{split_num}.pkl', 'wb') as f:
        pk.dump(all_info, f)
    
    # print(all_info)