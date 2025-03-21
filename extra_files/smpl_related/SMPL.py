from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn

from .lbs import lbs, hybrik, quat_to_rotmat,get_rest_pose_lbs, get_rest_pose_lbs_from_v, \
    get_global_transform, hybrik_get_global_transform, hybrik_fromwidths, hybrik_fromwidths_finer, \
    ts_decompose_rot

try:
    import cPickle as pk
except ImportError:
    import pickle as pk


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
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
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self,
                 model_path,
                 h36m_jregressor,
                 gender='neutral',
                 dtype=torch.float32,
                 num_joints=29,
                 shape_num=10):
        ''' SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        '''
        super(SMPL_layer, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        self.dtype = dtype

        self.faces = self.smpl_data.f

        ''' Register Buffer '''
        # Faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model, (6890, 3)
        self.register_buffer('v_template',
                             to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        # print(to_tensor(to_np(self.smpl_data.shapedirs)).shape, 'to_tensor(to_np(self.smpl_data.shapedirs)')
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype)[:, :, :shape_num])

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        # Vertices to Human3.6M Joints location (17, 6890)
        self.register_buffer(
            'J_regressor_h36m',
            to_tensor(to_np(h36m_jregressor), dtype=dtype))

        self.num_joints = num_joints

        # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        # extend kinematic tree
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]

        # print(parents)
        self.register_buffer(
            'children_map',
            self._parents_to_children(parents))
        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self,
                pose_axis_angle,
                betas,
                global_orient,
                transl=None,
                return_verts=True,
                is_get_twist=False,
                return_29_jts=False,
                pose2rot=True):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        # batch_size = pose_axis_angle.shape[0]

        # concate root orientation with thetas
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle

        # Translate thetas to rotation matrics
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_h36m, twist_angle = lbs(betas, full_pose, self.v_template,
                                                                 self.shapedirs, self.posedirs,
                                                                 self.J_regressor, self.J_regressor_h36m, self.parents,
                                                                 self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype,
                                                                 is_get_twist=is_get_twist)

        if return_29_jts:
            joints = get_jts_29(vertices, joints)
        
        if transl is not None:
            # apply translations
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = edict(
            vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m, twist_angle=twist_angle)
        return output

    def hybrik(self,
               pose_skeleton,
               betas,
               phis,
               global_orient,
               transl=None,
               return_verts=True,
               leaf_thetas=None,
               return_29_jts=False):
        ''' Inverse pass for the SMPL model

            Parameters
            ----------
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (X, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        vertices, new_joints, rot_mats, joints_from_verts = hybrik(
            betas, global_orient, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype, train=self.training,
            leaf_thetas=leaf_thetas)

        rot_mats = rot_mats.reshape(batch_size, 24, 3, 3)

        if return_29_jts:
            new_joints = get_jts_29(vertices, new_joints)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = edict(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output
    
    def hybrik_fromwidths(self,
               pose_skeleton,
               part_widths,
               phis,
               pred_bone_weight=None,
               init_beta=None,
               global_orient=None,
               transl=None,
               return_verts=True,
               leaf_thetas=None,
               return_29_jts=False,
               finer=False,
               rotate=True):
        ''' Inverse pass for the SMPL model

            Parameters
            ----------
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (X, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        batch_size = pose_skeleton.shape[0]

        vertices, new_joints, rot_mats, joints_from_verts, v_shaped, rest_J, bone_len_pred, bad_bone, target_new_pose = hybrik_fromwidths(
            part_widths, global_orient, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype, train=self.training,
            leaf_thetas=leaf_thetas, finer=finer, rotate=rotate, init_beta=init_beta,
            pred_bone_weight=pred_bone_weight)

        if not rotate:
            v_shaped = v_shaped - rest_J[:, [0]]
            rest_J = rest_J - rest_J[:, [0]]
            output = edict(
                v_shaped=v_shaped, rest_J=rest_J)
            return output

        rot_mats = rot_mats.reshape(batch_size, 24, 3, 3)

        if return_29_jts:
            new_joints = get_jts_29(vertices, new_joints)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

            v_shaped = v_shaped - rest_J[:, [0]]
            rest_J = rest_J - rest_J[:, [0]]

        output = edict(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts, 
            v_shaped=v_shaped, rest_J=rest_J, bone_len_pred=bone_len_pred, bad_bone=bad_bone, target_new_pose=target_new_pose)
        return output
    
    def get_rest_pose(self, beta, align=True):
        rest_J, v_shaped = get_rest_pose_lbs(beta, self.v_template, self.shapedirs, self.J_regressor)

        if align:
            root = rest_J[:, [0]].clone()
            v_shaped = v_shaped - root
            rest_J = rest_J - root
        
        output = edict(
            vertices=v_shaped, joints=rest_J)
        return output
    
    def get_rest_pose_from_v(self, vertices, align=True):
        rest_J, v_shaped = get_rest_pose_lbs_from_v(vertices, self.J_regressor)

        if align:
            root = rest_J[:, [0]].clone()
            v_shaped = v_shaped - root
            rest_J = rest_J - root
        
        output = edict(
            vertices=v_shaped, joints=rest_J)
        return output
    
    def get_bone_len(self, pred_xyz, pred_weight=None):
        batch_size = pred_xyz.shape[0]
        if pred_weight is None:
            pred_weight = torch.ones_like(pred_xyz)

        part_spines_3d = torch.zeros((batch_size, 28, 3), device=pred_xyz.device)
        part_spine_weight = torch.zeros((batch_size, 28, 3), device=pred_xyz.device)
        for i, k in enumerate(self.parents):
            if i == 0:
                continue

            base_joints_3d = pred_xyz[:, i], pred_xyz[:, k] # batch x 3
            weight_joints = pred_weight[:, i], pred_weight[:, k]

            part_spines_3d[:, i-1] = base_joints_3d[1] - base_joints_3d[0]
            part_spine_weight[:, i-1] = weight_joints[1] * weight_joints[0]
        
        return part_spines_3d, part_spine_weight

    def get_global_transform(self, pose, betas,):
        return get_global_transform(
            betas, pose, self.v_template, self.shapedirs, self.J_regressor, self.parents[:24], pose2rot=True)

    def hybrik_get_global_transform(self, pose_skeleton, phis,):
        return hybrik_get_global_transform(
            pose_skeleton, phis, self.v_template, self.J_regressor, self.parents, self.children_map)
    
    def hybrik_fromwidth_finer(self,
               pose_skeleton,
               part_widths,
               phis,
               split_num,
               all_info,
               global_orient=None,
               transl=None,
               return_verts=True,
               leaf_thetas=None,
               return_29_jts=False,
               rotate=True,
               pred_bone_weight=None):
        '''
        different from hybrik_fromwidths(finer=True) in that it supports split_num > 2
        '''
        batch_size = pose_skeleton.shape[0]

        part_names_finer_used = all_info.part_names_finer
        mean_part_width_finer_used = all_info.mean_part_width_finer
        part_seg_finer_used = all_info.part_seg_finer

        vertices, new_joints, rot_mats, joints_from_verts, v_shaped, rest_J = hybrik_fromwidths_finer(
            part_widths, global_orient, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype, train=self.training,
            leaf_thetas=leaf_thetas, rotate=rotate,
            split_num=split_num, part_names_finer_used=part_names_finer_used, 
            mean_part_width_finer_used=mean_part_width_finer_used, part_seg_finer_used=part_seg_finer_used,
            pred_bone_weight=pred_bone_weight)

        if not rotate:
            v_shaped = v_shaped - rest_J[:, [0]]
            rest_J = rest_J - rest_J[:, [0]]
            output = edict(
                v_shaped=v_shaped, rest_J=rest_J)
            return output

        rot_mats = rot_mats.reshape(batch_size, 24, 3, 3)

        if return_29_jts:
            new_joints = get_jts_29(vertices, new_joints)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

            v_shaped = v_shaped - rest_J[:, [0]]
            rest_J = rest_J - rest_J[:, [0]]

        output = edict(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts, v_shaped=v_shaped, rest_J=rest_J)
        return output

    def twist_swing_decompose_rot(
            self,
            rotmats,
            betas):

        batch_size = rotmats.shape[0]

        rotmat_swing, rotmat_twist, angle_twist = ts_decompose_rot(
            betas, rotmats,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype)

        return rotmat_swing.reshape(batch_size, 24, 3, 3), rotmat_twist.reshape(batch_size, 23, 3, 3), angle_twist.reshape(batch_size, 23, 1)

def get_jts_29(vertices, jts):
    leaf_indices = [411, 2445, 5905, 3216, 6617] # head, left hand, right hand, left foot, right foot
    leaf_jts = vertices[:, leaf_indices]
    new_jts = torch.cat([jts, leaf_jts], dim=1)
    return new_jts



class SMPL_layer_kid(SMPL_layer):
    def __init__(self,
                 model_path,
                 kid_template_path,
                 h36m_jregressor,
                 gender='neutral',
                 dtype=torch.float32,
                 num_joints=29):
        
        super(SMPL_layer_kid, self).__init__(
                model_path,
                h36m_jregressor,
                gender=gender,
                dtype=dtype,
                num_joints=num_joints
            )
        num_betas = 10
        v_template_smil = np.load(kid_template_path)
        v_template_smil -= np.mean(v_template_smil, axis=0)
        v_template_diff = np.expand_dims(v_template_smil - self.v_template.numpy(), axis=2)

        shapedirs = self.shapedirs.numpy()
        shapedirs = np.concatenate((shapedirs[:, :, :num_betas], v_template_diff), axis=2)
        num_betas = num_betas + 1

        self._num_betas = num_betas
        shapedirs = shapedirs[:, :, :num_betas]
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))