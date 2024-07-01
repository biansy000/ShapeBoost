from easydict import EasyDict as edict

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.smpl.SMPL import SMPL_layer
from .layers.hrnet.hrnet import get_hrnet
from shapeboost.beta_decompose.beta_process2 import part_names, part_width_dict, mean_part_width_ratio, spine_joints, part_pairs, \
    part_names_ids, joints_name_24, mean_part_width_ratio_all
from shapeboost.beta_decompose.beta_process2 import width2vertex, bonelen2vertex, vertice2capsule_fast, convert2tpose

def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def norm_heatmap(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


@SPPE.register_module
class HRNetSMPLCamSRatioAnalytical(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLCamSRatioAnalytical, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        self.USE_WIDTH_SIGMA = kwargs.get('USE_WIDTH_SIGMA', False)

        self.preact = get_hrnet(kwargs['HRNET_TYPE'], num_joints=self.num_joints,
                                depth_dim=self.depth_dim,
                                is_train=True, generate_feat=True, generate_hm=True,
                                pretrain=kwargs['HR_PRETRAINED'])

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )
        self.smpl_male = SMPL_layer(
            'model_files/smpl_v1.1.0/smpl/SMPL_MALE.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        self.smpl_female = SMPL_layer(
            'model_files/smpl_v1.1.0/smpl/SMPL_FEMALE.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

        self.joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                               (22, 23), (25, 26), (27, 28))

        self.root_idx_smpl = 0

        if isinstance(part_pairs[0][0], str):
            self.part_pairs_20 = [(part_names.index(item1), part_names.index(item2)) for (item1, item2) in part_pairs]
        else:
            self.part_pairs_20 = part_pairs

        # mean shape
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        init_cam = torch.tensor([0.9])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float())
        
        mean_width_ratio = torch.tensor(mean_part_width_ratio_all).float()
        self.register_buffer('mean_width_ratio', mean_width_ratio.float().unsqueeze(0))
        mean_part_widths_list = [part_width_dict[n]['mean'] for n in joints_name_24]
        self.register_buffer('mean_width', torch.tensor(mean_part_widths_list).float().unsqueeze(0))

        part_num = len(part_names)
        self.part_num = part_num
        self.dec_width_ratio = nn.Linear(2048, 20)
        if self.USE_WIDTH_SIGMA:
            self.decwidth_ratio_sigma = nn.Linear(2048, part_num)

        self.decphi = nn.Linear(2048, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(2048, 1)
        self.decsigma = nn.Linear(2048, 29)

        self.focal_length = float(kwargs['FOCAL_LENGTH'])
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2200, 2200, 2200)
        self.register_buffer('bbox_3d_shape', torch.tensor(bbox_3d_shape).float())
        self.depth_factor = float(self.bbox_3d_shape[2].numpy()) * 1e-3
        self.input_size = 256.0

        self._get_beta_net()
        self.get_smpl_templates()
        self.get_laplacian_matrix()

    def _get_beta_net(self):
        self.beta_regressor = nn.Sequential(
            nn.Linear(20*4+10, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10),
        )
  
    def get_smpl_templates(self):
        smpl_template_out = self.smpl_layer.get_rest_pose(beta=torch.zeros(1, 10), align=False)
        v_templates = smpl_template_out.vertices
        self.register_buffer('v_templates', v_templates.float()) 

        shapedirs = self.smpl_layer.shapedirs.reshape(6890, 3, 10)
        shapedirs_new = torch.zeros(6890, 3, 13)
        shapedirs_new[:, :, :10] = shapedirs
        shapedirs_new[:, 0, 10] = 1
        shapedirs_new[:, 1, 11] = 1
        shapedirs_new[:, 2, 12] = 1

        shapedirs_new_T = shapedirs_new.reshape(-1, 13).T

        shapedirs_new_inv = torch.inverse(shapedirs_new_T @ shapedirs_new.reshape(-1, 13))
        self.register_buffer('shapedirs_new_T', shapedirs_new_T.float())
        self.register_buffer('shapedirs_new_inv', shapedirs_new_inv.float())


    def get_laplacian_matrix(self):
        smpl_faces = self.smpl_layer.faces_tensor
        laplacian_matrix = torch.zeros((6890, 6890), dtype=torch.float32)
        for face in smpl_faces:
            laplacian_matrix[ face[0], [face[1], face[2]] ] = 1.0
            laplacian_matrix[ face[1], [face[0], face[2]] ] = 1.0
            laplacian_matrix[ face[2], [face[0], face[1]] ] = 1.0
        
        assert torch.abs(laplacian_matrix - laplacian_matrix.T).max() < 1e-5
        laplacian_matrix = laplacian_matrix / laplacian_matrix.sum(dim=1, keepdim=True)
        laplacian_matrix = laplacian_matrix - torch.eye(6890)

        self.register_buffer('laplacian_matrix', laplacian_matrix.float()) 

        template_laplace = torch.einsum('ij,bjk->bik', self.laplacian_matrix, self.v_templates)
        self.register_buffer('template_laplace', template_laplace.float()) 

    def _initialize(self):
        pass
    
    def forward(self, x, flip_test=False, **kwargs):
        flip_test = False
        batch_size = x.shape[0]

        out, x0 = self.preact(x)
        out = out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
        out = out.reshape((out.shape[0], self.num_joints, -1))
        heatmaps = norm_heatmap(self.norm_type, out)

        assert heatmaps.dim() == 3, heatmaps.shape
        maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))

        hm_x0 = heatmaps.sum((2, 3))  # (B, K, W)
        hm_y0 = heatmaps.sum((2, 4))  # (B, K, H)
        hm_z0 = heatmaps.sum((3, 4))  # (B, K, D)

        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device).unsqueeze(-1)

        coord_x = hm_x0.matmul(range_tensor)
        coord_y = hm_y0.matmul(range_tensor)
        coord_z = hm_z0.matmul(range_tensor)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)

        x0 = x0.view(x0.size(0), -1)
        init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

        xc = x0

        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        sigma = self.decsigma(xc).reshape(batch_size, 29, 1).sigmoid()

        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        pred_width_ratio_raw = self.dec_width_ratio(xc)
        pred_width_ratio = torch.zeros(batch_size, 24, device=pred_width_ratio_raw.device) + self.mean_width_ratio
        pred_width_ratio[:, part_names_ids] += pred_width_ratio_raw

        width_ratio_sigma = torch.ones_like(pred_width_ratio)
        if self.USE_WIDTH_SIGMA:
            width_ratio_sigma = self.decwidth_ratio_sigma(xc).sigmoid() + 1e-3

        camScale = pred_camera[:, :1].unsqueeze(1)
        # camTrans = pred_camera[:, 1:].unsqueeze(1)

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = (bboxes[:, 2] - bboxes[:, 0])
            h = (bboxes[:, 3] - bboxes[:, 1])

            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h

            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)

            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].detach().clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2].detach() + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].detach().clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2].detach() * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)

        transl = camera_root.clone()
        if 'bboxes' in kwargs.keys():
            transl[:, :2] = transl[:, :2] - bbox_center.squeeze() * (256 / 1000.0) * camDepth.reshape(batch_size, 1)

        pred_bone, _ = self.get_bone(pred_xyz_jts_29*2.2)
        pred_bone_len = pred_bone.norm(dim=-1)
        pred_width = pred_width_ratio * pred_bone_len

        out_raw = self.smpl_layer.shapeboost_fromwidth(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) *2.2, 
            part_widths=pred_width,
            phis=pred_phi,
            return_verts=True,
            return_29_jts=True
        )

        pred_v_raw = out_raw.vertices
        pred_v_shaped = out_raw.v_shaped

        if 'bboxes' in kwargs.keys():
            pred_vertices_uv = (pred_v_raw[:, :, :2] + camera_root[:, :2].unsqueeze(1)) / (pred_v_raw[:, :, [2]] + camDepth) / 256.0 * 1000.0 - bbox_center
        else:
            pred_vertices_uv = (pred_v_raw[:, :, :2] + camera_root[:, :2].unsqueeze(1)) / (pred_v_raw[:, :, [2]] + camDepth) / 256.0 * 1000.0

        beta_net_output = self.forward_beta_net(
            pred_width_ratio[:, part_names_ids], pred_bone_len[:, part_names_ids], width_ratio_sigma, pred_v_shaped)
        
        pred_shape = beta_net_output['pred_betas']

        output_smpl = self.smpl_layer.shapeboost(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) *2.2, 
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True,
            return_29_jts=True
        )

        pred_vertices = output_smpl.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_29_struct = output_smpl.joints.float() / 2.2
        pred_xyz_jts_24_struct = pred_xyz_jts_29_struct[:, :24]
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output_smpl.joints_from_verts.float() / 2.2
        pred_theta_mats = output_smpl.rot_mats.float().reshape(batch_size, 24 * 9)
        # pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_29_struct = pred_xyz_jts_29_struct.reshape(batch_size, -1)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        if 'gt_beta' in kwargs:
            gt_beta = kwargs['gt_beta']
            gt_smpl_out = self.smpl_layer.get_rest_pose(gt_beta, align=True)
            gt_vertices = gt_smpl_out.vertices

            gt_smpl_out_rest = {
                'vertices': gt_vertices,
                'joints': gt_smpl_out.joints
            }
        else:
            gt_smpl_out_rest = {}
        
        output = edict(
            pred_phi=pred_phi,
            pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1),
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_29_struct=pred_xyz_jts_29_struct,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct.reshape(batch_size, -1),

            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            pred_theta_mats=pred_theta_mats,
            gt_smpl_out_rest=gt_smpl_out_rest,

            cam_scale=camScale[:, 0],
            cam_root=camera_root,
            transl=transl,
            pred_camera=pred_camera,
            sigma=sigma,
            scores=1 - sigma,
            maxvals=maxvals,
            img_feat=x0,
            pred_skeleton_new=pred_xyz_jts_29,

            pred_width=pred_width,
            pred_width_ratio=pred_width_ratio,
            bone_len=pred_bone.norm(dim=-1),
            pred_shape=beta_net_output['pred_betas'],
            smpl_out_rest=beta_net_output,
            width_ratio_sigma=width_ratio_sigma,
            pred_vertices_2d=pred_vertices_uv,
            smpl_out_raw=out_raw,
            # smpl_out_raw_detached=out_raw_detached
        )
        return output

    def forward_beta_net(self, part_widths_ratio, bone_len, width_sigma, v_shaped_raw):
    
        batch_size = v_shaped_raw.shape[0]
        vshaped = v_shaped_raw - self.smpl_layer.v_template

        result = torch.einsum('ij,bj->bi', self.shapedirs_new_T, vshaped.reshape(batch_size, 6890*3))
        result = torch.einsum('ij,bj->bi', self.shapedirs_new_inv, result)
        pred_beta_raw0 = result[:, :10] # least square projection
        
        pred_beta_raw = torch.clamp(pred_beta_raw0, min=-10, max=10)

        beta_raw_out = self.smpl_layer.get_rest_pose(pred_beta_raw, align=True)
        bone_len_raw, _ = self.get_part_spine(beta_raw_out.joints)
        bone_len_raw = torch.norm(bone_len_raw, dim=-1)
        part_widths_raw = vertice2capsule_fast(beta_raw_out.vertices, beta_raw_out.joints)
        part_widths_ratio_raw = part_widths_raw / (bone_len_raw + 1e-5)

        delta_bone_len = bone_len - bone_len_raw
        delta_part_widths_ratio = part_widths_ratio - part_widths_ratio_raw

        inp_per_view = torch.cat([part_widths_ratio, bone_len, delta_bone_len, delta_part_widths_ratio, pred_beta_raw], dim=-1) # batch_size, part_num, embedding_size+3
        pred_beta = self.beta_regressor(inp_per_view) + pred_beta_raw

        smpl_out_rest = self.smpl_layer.get_rest_pose(pred_beta, align=True)
        smpl_jts = smpl_out_rest.joints
        pred_vertices_struct_align = smpl_out_rest.vertices

        laplace = torch.einsum('ij,bjk->bik', self.laplacian_matrix, pred_vertices_struct_align)
        laplace_loss = torch.abs(laplace).mean() * 0.01 + torch.square(laplace - self.template_laplace).mean()

        out = {
            'pred_beta_raw': pred_beta_raw0,
            'vertices_lsq': beta_raw_out.vertices,
            'pred_betas': pred_beta,
            'pred_rest_jts': smpl_jts,
            'smpl_jts_full': smpl_jts,
            'pred_vertices_struct': pred_vertices_struct_align,
            'laplace': laplace,
            'laplace_loss': laplace_loss
        }

        return out
    
    def get_part_spine(self, pred_xyz, pred_weight=None):
        part_num = len(part_names)
        batch_size = pred_xyz.shape[0]
        if pred_weight is None:
            pred_weight = torch.ones_like(pred_xyz)

        part_spines_3d = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
        part_spine_weight = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
        for i, k in enumerate(part_names):
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
    
    def get_bone(self, pred_xyz, pred_weight=None):
        part_num = len(joints_name_24)
        batch_size = pred_xyz.shape[0]
        if pred_weight is None:
            pred_weight = torch.ones_like(pred_xyz)

        part_spines_3d = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
        part_spine_weight = torch.zeros((batch_size, part_num, 3), device=pred_xyz.device)
        for i, k in enumerate(joints_name_24):
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

