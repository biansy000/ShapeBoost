import os
import json

import numpy as np
import torch
import copy
import pickle as pk


joints_name_24 = (
    'pelvis', 'left_hip', 'right_hip',      # 2
    'spine1', 'left_knee', 'right_knee',    # 5
    'spine2', 'left_ankle', 'right_ankle',  # 8
    'spine3', 'left_foot', 'right_foot',    # 11
    'neck', 'left_collar', 'right_collar',  # 14
    'jaw',                                  # 15
    'left_shoulder', 'right_shoulder',      # 17
    'left_elbow', 'right_elbow',            # 19
    'left_wrist', 'right_wrist',            # 21
    'left_thumb', 'right_thumb',            # 23
)

joints_name_29 = (
    'pelvis', 'left_hip', 'right_hip',      # 2
    'spine1', 'left_knee', 'right_knee',    # 5
    'spine2', 'left_ankle', 'right_ankle',  # 8
    'spine3', 'left_foot', 'right_foot',    # 11
    'neck', 'left_collar', 'right_collar',  # 14
    'jaw',                                  # 15
    'left_shoulder', 'right_shoulder',      # 17
    'left_elbow', 'right_elbow',            # 19
    'left_wrist', 'right_wrist',            # 21
    'left_thumb', 'right_thumb',            # 23
    'head', 'left_middle', 'right_middle',  # 26
    'left_bigtoe', 'right_bigtoe'           # 28
)

part_names = (
    'pelvis', 'left_hip', 'right_hip',      # 2
    'spine1', 'left_knee', 'right_knee',    # 5
    'spine2', 'left_ankle', 'right_ankle',  # 8
    'spine3',     # 11
    'neck', 'left_collar', 'right_collar',  # 14
    'jaw',                                  # 15
    'left_shoulder', 'right_shoulder',      # 17
    'left_elbow', 'right_elbow',            # 19
    'left_wrist', 'right_wrist',            # 21           # 23
)

part_pairs_24 = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
part_pairs_29 = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
part_pairs = [
    (joints_name_24[i], joints_name_24[j]) for i, j in part_pairs_24 if joints_name_24[i] in part_names
]

part_names_ids = [joints_name_24.index(item) for item in part_names]
part_root_joints = {name: i for i, name in enumerate(joints_name_24)}

######### MAY NEED CHANGE ###########
spine_joints = {
    'pelvis': [0, 1],
    'left_hip': [1, 4],
    'right_hip': [2, 5],
    'spine1': [3, 6],
    'left_knee': [4, 7], # 4
    'right_knee': [5, 8],
    'spine2': [6, 9],
    'left_ankle': [7, 10],
    'right_ankle': [8, 11], 
    'spine3': [9, 12], # 9
    'left_foot': [10, 27],
    'right_foot': [11, 28],
    'neck': [12, 15],
    'left_collar': [13, 16],
    'right_collar': [14, 17], # 14
    'jaw': [15, 24],
    'left_shoulder': [16, 18],
    'right_shoulder': [17, 19],
    'left_elbow': [18, 20],
    'right_elbow': [19, 21],
    'left_wrist': [20, 22],
    'right_wrist': [21, 23],
    'left_thumb': [22, 25],
    'right_thumb': [23, 26]
}

part_width_dict = {
    'pelvis': {'mean': 0.13929885625839233, 'std': 0.013549627736210823},
    'left_hip': {'mean': 0.0847155973315239, 'std': 0.008838041685521603},
    'right_hip': {'mean': 0.08412877470254898, 'std': 0.008959203958511353},
    'spine1': {'mean': 0.13803082704544067, 'std': 0.018005041405558586},
    'left_knee': {'mean': 0.053455255925655365, 'std': 0.004312389530241489},
    'right_knee': {'mean': 0.054265670478343964, 'std': 0.0040938532911241055},
    'spine2': {'mean': 0.1382027566432953, 'std': 0.01569226384162903},
    'left_ankle': {'mean': 0.04846811667084694, 'std': 0.0033021627459675074},
    'right_ankle': {'mean': 0.04458404332399368, 'std': 0.00285510066896677},
    'spine3': {'mean': 0.1316613405942917, 'std': 0.011307621374726295},
    'left_foot': {'mean': 0.027405019849538803, 'std': 0.0014423956163227558},
    'right_foot': {'mean': 0.02842632122337818, 'std': 0.0016088547417894006},
    'neck': {'mean': 0.05528124421834946, 'std': 0.0053253816440701485},
    'left_collar': {'mean': 0.07151363790035248, 'std': 0.006135474890470505},
    'right_collar': {'mean': 0.07085411250591278, 'std': 0.005884986370801926},
    'jaw': {'mean': 0.08187920600175858, 'std': 0.003583488054573536},
    'left_shoulder': {'mean': 0.05078605189919472, 'std': 0.00545124989002943},
    'right_shoulder': {'mean': 0.05125593766570091, 'std': 0.00513016851618886},
    'left_elbow': {'mean': 0.03471076861023903, 'std': 0.0029087967704981565},
    'right_elbow': {'mean': 0.03544062376022339, 'std': 0.0031154637690633535},
    'left_wrist': {'mean': 0.04814157262444496, 'std': 0.0037109225522726774},
    'right_wrist': {'mean': 0.048644401133060455, 'std': 0.004005746450275183},
    'left_thumb': {'mean': 0.028104497119784355, 'std': 0.00169226317666471},
    'right_thumb': {'mean': 0.029078522697091103, 'std': 0.0019420929020270705},
}

mean_part_width_ratio_dict = {
    'pelvis': 1.2189841270446777,
    'left_hip': 0.2245626151561737,
    'right_hip': 0.21838949620723724,
    'spine1': 1.023822546005249,
    'left_knee': 0.13488133251667023,
    'right_knee': 0.13670524954795837,
    'spine2': 2.373756170272827,
    'left_ankle': 0.38240596652030945,
    'right_ankle': 0.3475398123264313,
    'spine3': 0.5910796523094177,
    'neck': 0.6615042090415955,
    'left_collar': 0.7297677397727966,
    'right_collar': 0.6790704131126404,
    'jaw': 0.40063807368278503,
    'left_shoulder': 0.1908128708600998,
    'right_shoulder': 0.19680874049663544,
    'left_elbow': 0.13956378400325775,
    'right_elbow': 0.13914592564105988,
    'left_wrist': 0.5958896279335022,
    'right_wrist': 0.6032969355583191,
    
    'left_thumb': 0.26233047246932983,
    'right_thumb': 0.28363528847694397,
    'left_foot': 0.3328140079975128,
    'right_foot': 0.34842967987060547,
}

mean_bone_len_dict = {
    'pelvis': 0.11504146456718445,
    'left_hip': 0.3767879009246826,
    'right_hip': 0.3845822513103485,
    'spine1': 0.13529616594314575,
    'left_knee': 0.4005826413631439,
    'right_knee': 0.40096548199653625,
    'spine2': 0.05873069167137146,
    'left_ankle': 0.13430221378803253,
    'right_ankle': 0.13481949269771576,
    'spine3': 0.21813981235027313,
    'neck': 0.0829717218875885,
    'left_collar': 0.0963524878025055,
    'right_collar': 0.10179169476032257,
    'jaw': 0.20524826645851135,
    'left_shoulder': 0.26137229800224304,
    'right_shoulder': 0.25499147176742554,
    'left_elbow': 0.24939829111099243,
    'right_elbow': 0.25547686219215393,
    'left_wrist': 0.08575001358985901,
    'right_wrist': 0.08546747267246246,
}

mean_bone_len_full_dict = {
    'left_hip': 0.11504146456718445,
    'right_hip': 0.11310231685638428,
    'spine1': 0.11221449077129364,
    'left_knee': 0.3767879009246826,
    'right_knee': 0.3845822513103485,
    'spine2': 0.13529616594314575,
    'left_ankle': 0.4005826413631439,
    'right_ankle': 0.40096548199653625,
    'spine3': 0.05873069167137146,
    'left_foot': 0.13430221378803253,
    'right_foot': 0.13481949269771576,
    'neck': 0.21813981235027313,
    'left_collar': 0.1490015834569931,
    'right_collar': 0.14932164549827576,
    'jaw': 0.0829717218875885,
    'left_shoulder': 0.0963524878025055,
    'right_shoulder': 0.10179169476032257,
    'left_elbow': 0.26137229800224304,
    'right_elbow': 0.25499147176742554,
    'left_wrist': 0.24939829111099243,
    'right_wrist': 0.25547686219215393,
    'left_thumb': 0.08575001358985901,
    'right_thumb': 0.08546747267246246
}

mean_part_width_ratio = [mean_part_width_ratio_dict[n] for n in part_names]
mean_bone_len = [mean_bone_len_dict[n] for n in part_names]
mean_part_width_ratio_all = [mean_part_width_ratio_dict[n] for n in joints_name_24]

with open('shapeboost/models/layers/smpl/lbs_weights.pkl', 'rb') as f:
    default_lbs_weights = pk.load(f)

with open('shapeboost/models/layers/smpl/part_seg2.json', 'r') as fid:
    part_seg = json.load(fid)

used_part_seg = {
    k: part_seg[k] for k in part_names
}
part_seg_lens = [len(used_part_seg[name]) for name in part_names]


def vertice2capsule(vertices, joints24, lbs_weights=None):
    # vertices: batch x 6980 x 3, joints24: batch x 24 x 3
    # print(vertices.shape, joints24.shape)
    if lbs_weights is None:
        lbs_weights = default_lbs_weights.clone().to(vertices.device)

    capsule_dict = {}
    for k, v in used_part_seg.items():
        part_spine_idx = spine_joints[k]
        root_joint = part_root_joints[k]

        part_lbs_weights = lbs_weights[v, root_joint] # l
        part_lbs_weights[part_lbs_weights<0.5] = 0

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
        part_spine_normed = part_spine_normed.unsqueeze(dim=1) # batch x 1 x 3

        vec = part_vertices - base_joint.unsqueeze(1)
        dis2spine = torch.cross(part_spine_normed.expand(-1, vec.shape[1], -1), vec, dim=-1)
        assert dis2spine.shape[-1] == 3

        dis2spine = torch.norm(dis2spine, dim=-1)
        mean_dis2spine = (dis2spine * part_lbs_weights).sum(dim=-1) / part_lbs_weights.sum()
        std_dis2spine = dis2spine.std(dim=1)

        capsule_dict[k] = {'part_spine': part_spine, 'mean_dis2spine': mean_dis2spine, 
            'base_joint': base_joint, 'std_dis2spine': std_dis2spine
            }
    
    return capsule_dict


def vertice2capsule_fast(vertices, joints24, lbs_weights=None):
    # vertices: batch x 6980 x 3, joints24: batch x 24 x 3
    # print(vertices.shape, joints24.shape)
    capsule_dict = []
    if lbs_weights is None:
        lbs_weights = default_lbs_weights.clone().detach().to(vertices.device)
    
    for k in part_names:
        v = used_part_seg[k]
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
        # print(dis2spine.mean(dim=1, keepdim=True))

        capsule_dict.append(mean_dis2spine)
    
    capsule_dict = torch.cat(capsule_dict, dim=1)
    return capsule_dict


def vertice2capsule_fast_24(vertices, joints24, lbs_weights=None):
    # vertices: batch x 6980 x 3, joints24: batch x 24 x 3
    # print(vertices.shape, joints24.shape)
    capsule_dict = []
    if lbs_weights is None:
        lbs_weights = default_lbs_weights.clone().detach().to(vertices.device)
    
    for k in joints_name_24:
        v = part_seg[k]
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
        # print(dis2spine.mean(dim=1, keepdim=True))

        capsule_dict.append(mean_dis2spine)
    
    capsule_dict = torch.cat(capsule_dict, dim=1)
    return capsule_dict

    
def width2vertex(widths, v_templates, t_pose):
    # v_templates: batch x 6890 x 3, t_pose: batch x 29(24) x 3, widths: batch x 20
    batch_size = widths.shape[0]
    new_vertices = []
    for i, name in enumerate(part_names):
        part_v = v_templates[:, used_part_seg[name]]

        mean_dis = part_width_dict[name]['mean']
        new_mean_dis = widths[:, i]

        adjust_ratio = new_mean_dis / mean_dis # batch_size
        adjust_ratio = adjust_ratio.reshape(batch_size, 1, 1)

        part_spine_idx = spine_joints[name]
        if name != 'hips':
            part_spine = [t_pose[:, part_spine_idx[0]], t_pose[:, part_spine_idx[1]]] # batch x 3
        else:
            part_spine = [t_pose[:, 0], t_pose[:, [1, 2]].mean(dim=1)]

        # The plane that passes (vx, vy, vz) and perpendicular to spine: 
        #       rx * x + ry * y + rz * z - (rx * vx + ry * vy + rz * vz) = 0
        # The intersection point: ( x1 + k * rx, y1 + k * ry, z1 + k * z1 )
        # if rx^2 + ry^2 + rz^2 = 0, then k = rx * (vx - x1) + ry * (vy - y1) + rz * (vz - z1)
        spine_dir = part_spine[1] - part_spine[0]
        spine_dir = spine_dir / torch.norm(spine_dir, dim=1, keepdim=True) # batch x 3

        dv = part_v - part_spine[0].unsqueeze(1)
        inner_prod = (dv * spine_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)
        intersect_p = spine_dir.unsqueeze(1) * inner_prod + part_spine[0].unsqueeze(1)

        perp_vec = part_v - intersect_p
        perp_flag = torch.abs( (perp_vec * spine_dir.unsqueeze(1)).sum(dim=-1) )
        assert (perp_flag < 1e-3).all(), perp_flag.max()

        adjust_perp_vec = perp_vec * adjust_ratio
        new_part_v = adjust_perp_vec + intersect_p

        new_vertices.append(new_part_v)

    new_vertices = torch.cat(new_vertices, dim=1)
    return new_vertices


def bonelen2vertex(target_t_pose, v_templates, t_pose):

    batch_size = target_t_pose.shape[0]
    new_vertices = []
    for i, name in enumerate(part_names):
        part_v = v_templates[:, used_part_seg[name]]

        part_spine_idx = spine_joints[name]
        if name != 'hips':
            part_spine = [t_pose[:, part_spine_idx[0]], t_pose[:, part_spine_idx[1]]] # batch x 3
            new_part_spine = [target_t_pose[:, part_spine_idx[0]], target_t_pose[:, part_spine_idx[1]]]
        else:
            part_spine = [t_pose[:, 0], t_pose[:, [1, 2]].mean(dim=1)]
            new_part_spine = [target_t_pose[:, 0], target_t_pose[:, [1, 2]].mean(dim=1)]

        spine_len = torch.norm(part_spine[1] - part_spine[0], dim=1)
        new_spine_len = torch.norm(new_part_spine[1] - new_part_spine[0], dim=1)

        adjust_ratio = new_spine_len / spine_len # batch_size
        adjust_ratio = adjust_ratio.reshape(batch_size, 1, 1)

        # The plane that passes (vx, vy, vz) and perpendicular to spine: 
        #       rx * x + ry * y + rz * z - (rx * vx + ry * vy + rz * vz) = 0
        # The intersection point: ( x1 + k * rx, y1 + k * ry, z1 + k * z1 )
        # if rx^2 + ry^2 + rz^2 = 0, then k = rx * (vx - x1) + ry * (vy - y1) + rz * (vz - z1)
        spine_dir = part_spine[1] - part_spine[0]
        spine_dir = spine_dir / torch.norm(spine_dir, dim=1, keepdim=True) # batch x 3

        new_spine_dir = new_part_spine[1] - new_part_spine[0]
        new_spine_dir = new_spine_dir / torch.norm(new_spine_dir, dim=1, keepdim=True) # batch x 3
        # if name != 'hips':
        #     assert (torch.abs(new_spine_dir - spine_dir) < 1e-3).all(), torch.abs(new_spine_dir - spine_dir).max()

        p1 = (part_spine[0] + part_spine[1]) * 0.5
        p1_new = (new_part_spine[0] + new_part_spine[1]) * 0.5

        dv = part_v - p1.unsqueeze(1)
        inner_prod = (dv * spine_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)
        intersect_p = spine_dir.unsqueeze(1) * inner_prod + p1.unsqueeze(1)

        perp_vec = part_v - intersect_p
        perp_flag = torch.abs( (perp_vec * spine_dir.unsqueeze(1)).sum(dim=-1) )
        assert (perp_flag < 1e-3).all(), perp_flag.max()

        new_part_v = spine_dir.unsqueeze(1) * inner_prod * adjust_ratio + p1_new.unsqueeze(1) + perp_vec
        # new_part_v = spine_dir.unsqueeze(1) * inner_prod * adjust_ratio + p1.unsqueeze(1) + perp_vec

        new_vertices.append(new_part_v)

    new_vertices = torch.cat(new_vertices, dim=1)
    return new_vertices



def convert2tpose(raw_pose, t_pose):
    # raw_pose: torch.tensor, batch x 29 x 3
    parents = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
        16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11])
    
    batch_size = raw_pose.shape[0]
    new_t_pose = torch.zeros((batch_size, 29, 3), device=raw_pose.device)
    for i in range(1, 29):
        rel_part = raw_pose[:, i] - raw_pose[:, parents[i]]
        rel_length = torch.norm(rel_part, dim=-1, keepdim=True)

        direction = t_pose[:, i] - t_pose[:, parents[i]]
        direction = direction / torch.norm(direction, dim=-1, keepdim=True)

        new_rel_part = rel_length * direction
        new_t_pose[:, i] = new_t_pose[:, parents[i]] + new_rel_part
    
    return new_t_pose