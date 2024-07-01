import torch
import torch.nn as nn
import math
from easydict import EasyDict as edict

from .layers.smpl.SMPL import SMPL_layer
from shapeboost.beta_decompose.beta_process2 import vertice2capsule_fast, part_names, spine_joints, part_names_ids
import torch.nn.functional as F
from .builder import LOSS


def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


def weighted_laplace_loss(input, target, sigma, weights, amp, size_average):
    loss = torch.log(sigma / amp) + torch.abs(input - target) / (math.sqrt(2) * sigma + 1e-9)
    loss = loss * weights

    if size_average and weights.sum() > 0:
        return loss.sum() / weights.sum()
    else:
        return loss.sum()



@LOSS.register_module
class L1LossDimSMPLCam(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.use_laplace = self.elements['USE_LAPLACE']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40
        self.amp = 1 / math.sqrt(2 * math.pi)

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        pred_xyz = pred_xyz.reshape(-1, 24, 3)
        target_xyz = target_xyz.reshape(-1, 24, 3)
        target_xyz_weight = target_xyz_weight.reshape(-1, 24, 3)
        pred_xyz = pred_xyz - pred_xyz[:, [0], :]
        target_xyz = target_xyz - target_xyz[:, [0], :]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]

        if self.use_laplace:
            sigma = output.sigma
            loss_uvd = weighted_laplace_loss(
                pred_uvd.reshape(batch_size, 29, 3),
                target_uvd.reshape(batch_size, 29, 3),
                sigma.reshape(batch_size, 29, 1),
                target_uvd_weight.reshape(batch_size, 29, -1),
                self.amp,
                self.size_average)
        else:
            loss_uvd = weighted_l1_loss(
                pred_uvd.reshape(batch_size, -1),
                target_uvd.reshape(batch_size, -1),
                target_uvd_weight.reshape(batch_size, -1), self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        if epoch_num > self.pretrain_epoch:
            loss += loss_xyz * self.xyz24_weight

        loss += loss_uvd * self.uvd24_weight

        target_xyz_weight = target_xyz_weight.reshape(-1, 72)
        smpl_weight = (target_xyz_weight.sum(axis=1) >= 6).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        # pred_trans = output.cam_trans * smpl_weight
        assert output.cam_scale.shape == smpl_weight.shape
        pred_scale = output.cam_scale * smpl_weight
        # target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        assert target_scale.shape == smpl_weight.shape
        # trans_loss = self.criterion_smpl(pred_trans, target_trans)
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        if epoch_num > self.pretrain_epoch:
            # loss += 0.1 * (scale_loss)
            loss += 1 * (scale_loss)
        else:
            loss += 1 * (scale_loss)

        return loss


@LOSS.register_module
class L1LossDimSMPLShapeRatio(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLShapeRatio, self).__init__()

        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.USE_WIDTH_SIGMA = self.elements.get('USE_WIDTH_SIGMA', False)

        self.criterion_smpl = nn.MSELoss()
        self.criterion_smpl_l1 = nn.L1Loss()
        self.size_average = size_average

        print('size_average', size_average)
        self.reduce = reduce

        self.amp = 1 / math.sqrt(2 * math.pi)

        self.smpl29_parents = [-1,  0,  0,  0,  1,  2,  3,
            4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11
        ]

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, num_epoch=0):
        if isinstance(output, dict):
            output = edict(output)

        smpl_weight = labels['target_smpl_weight']
        batch_size = smpl_weight.shape[0]
        width_changed = labels['width_changed'].reshape(batch_size, 1)
        width_not_changed = 1 - width_changed

        # SMPL params
        loss_beta = self.criterion_smpl_l1(
            output.pred_shape * smpl_weight * width_not_changed, 
            labels['target_beta'] * smpl_weight * width_not_changed
        )
        loss_beta_reg = torch.square(output.pred_shape) * 0.01
        loss_beta = loss_beta + loss_beta_reg.mean()

        pred_theta = output.pred_theta_mats.reshape(batch_size, 24, 9) * labels['target_theta_weight'].reshape(batch_size, 24, 4).mean(dim=-1, keepdim=True)
        target_theta = labels['target_theta'].reshape(batch_size, 24, 9) * labels['target_theta_weight'].reshape(batch_size, 24, 4).mean(dim=-1, keepdim=True)
        loss_theta = self.criterion_smpl(
            pred_theta.reshape(batch_size, -1) * smpl_weight * width_not_changed, 
            target_theta.reshape(batch_size, -1) * smpl_weight * width_not_changed)

        loss_twist = self.criterion_smpl(
            output.pred_phi * labels['target_twist_weight'] * width_not_changed.unsqueeze(-1), 
            labels['target_twist'] * labels['target_twist_weight'] * width_not_changed.unsqueeze(-1)
        )
        
        # widths regularization_loss
        pred_xyz = output.pred_xyz_jts_29.reshape(batch_size, -1)
        target_xyz = labels['target_xyz_29']
        target_xyz_weight = labels['target_xyz_weight_29']

        pred_xyz_refined = pred_xyz.reshape(batch_size, 29, 3)
        target_xyz = target_xyz.reshape(batch_size, 29, 3)
        target_xyz_weight = target_xyz_weight.reshape(batch_size, -1, 3)

        # pred_part_spine, _ = self.get_part_spine(pred_xyz_refined*2.2)
        # target_part_spine, _ = self.get_part_spine(target_xyz*2.2, target_xyz_weight)

        pred_bones, _ = self.get_bone_len(pred_xyz_refined*2.2)
        if 'smpl_jts_full' in output.smpl_out_rest:
            pred_bones_struct, _ = self.get_bone_len(output.smpl_out_rest.smpl_jts_full.reshape(batch_size, 29, 3))
            pred_rest_jts = output.smpl_out_rest.smpl_jts_full.reshape(batch_size, -1, 3)
        else:
            pred_bones_struct, _ = self.get_bone_len(output.smpl_out_rest.pred_rest_jts.reshape(batch_size, 29, 3))
            pred_rest_jts = output.smpl_out_rest.pred_rest_jts.reshape(batch_size, -1, 3)
        target_bones, target_bone_weight = self.get_bone_len(target_xyz*2.2, target_xyz_weight)
        
        gt_rest_jts = output.gt_smpl_out_rest['joints'].reshape(batch_size, -1, 3)
        gt_rest_jts = gt_rest_jts - gt_rest_jts[:, [0]]
        
        pred_rest_jts = pred_rest_jts - pred_rest_jts[:, [0]]

        bone_len_loss = torch.abs(pred_bones.norm(dim=-1).detach() - pred_bones_struct.norm(dim=-1))
        bone_len_loss2 = torch.abs(gt_rest_jts.reshape(batch_size, -1) - pred_rest_jts.reshape(batch_size, -1)) * width_not_changed * smpl_weight
        bone_len_loss = bone_len_loss.mean() + bone_len_loss2.mean()

        pred_uvd = output.pred_uvd_jts.reshape(batch_size,-1,3)[:, :29]
        target_uvd = labels['target_uvd_29'][:, :29*3]
        target_uvd_weight = labels['target_weight_29'][:, :29*3]

        width_mask_uvd = torch.ones_like(target_uvd).reshape(batch_size, -1, 3)
        width_mask_uvd[:, :, 2] = width_mask_uvd[:, :, 2] * width_not_changed
        loss_uvd = weighted_laplace_loss(
                pred_uvd.reshape(batch_size, 29, 3),
                target_uvd.reshape(batch_size, 29, 3),
                output.sigma.reshape(batch_size, 29, 1),
                target_uvd_weight.reshape(batch_size, 29, -1)*width_mask_uvd,
                self.amp,
                self.size_average)
        
        loss_width, width_ratio_l = self.get_widths_loss(
            pred_xyz0=pred_xyz_refined.reshape(batch_size, -1, 3)*2.2, 
            pred_xyz_struct=output.pred_xyz_jts_24_struct.reshape(batch_size, -1, 3)*2.2, 
            pred_beta=output.pred_shape, 
            smpl_out_rest=output.smpl_out_rest,
            gt_width=labels['part_widths'],
            gt_width_ratio=labels['width_ratio'], 
            gt_smpl_out_rest=output.gt_smpl_out_rest,
            smpl_weight=smpl_weight,
            widths_weight=width_not_changed,
            output=output
        )

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight + loss_twist * self.twist_weight
        loss += (bone_len_loss + loss_width)
        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.reshape(batch_size, -1).sum(dim=1) > 3) * 1.0 # batch
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_scale = output.cam_scale * smpl_weight
        target_trans = labels['camera_trans'] * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        
        trans_loss = 0.0
        if 'cam_trans' in output:
            pred_trans = output.cam_trans * smpl_weight
            trans_loss = self.criterion_smpl(pred_trans*width_not_changed, target_trans*width_not_changed)

        scale_loss = self.criterion_smpl(pred_scale*width_not_changed, target_scale*width_not_changed)

        cam_range = torch.maximum(labels['width_changed_ratio'], 1/(labels['width_changed_ratio']+1e-5)).reshape(batch_size, 1)
        target_scale_upper_bound = target_scale * cam_range
        target_scale_lower_bound = target_scale / cam_range
        scale_loss2 = F.relu(pred_scale - target_scale_upper_bound - 1e-3).mean() + F.relu(target_scale_lower_bound - pred_scale - 1e-3).mean()

        loss += (5 * scale_loss + scale_loss2*0.5 + 0.5 * trans_loss)
        
        loss_dict = {
            # 'uv_l': loss_uvd.detach().cpu().numpy(),
            'smpl_l': (loss_beta + loss_theta*0.01 + loss_twist).detach().cpu().numpy(),
            'w_l': loss_width.detach().cpu().numpy(),
            'w_r_l': width_ratio_l.mean().detach().cpu().numpy(),
            # 's_l': (5 * scale_loss + scale_loss2 + 0.5 * trans_loss).detach().cpu().numpy()
        }

        return loss, loss_dict
      
    def get_widths_loss(self, pred_xyz0, pred_xyz_struct, pred_beta, smpl_out_rest, 
                gt_width, gt_width_ratio, gt_smpl_out_rest, smpl_weight, widths_weight, output=None):
        # beta & pred part_widths compatible + beta & pred xyz_29 compatible
        # pred_widths ratio is compatible with gt_width_ratio
        # all unit=m
        if isinstance(gt_smpl_out_rest, dict):
            gt_smpl_out_rest = edict(gt_smpl_out_rest)
        if isinstance(smpl_out_rest, dict):
            smpl_out_rest = edict(smpl_out_rest)

        pred_xyz = pred_xyz0
        batch_size = pred_beta.shape[0]

        pred_vertices_full = smpl_out_rest.pred_vertices_full
        pred_vertices_struct = smpl_out_rest.pred_vertices_struct
        pred_vertices_struct_align = smpl_out_rest.pred_vertices_struct_align
        pred_vertices_full_align = smpl_out_rest.pred_vertices_full_align
        pred_rest_jts_full = smpl_out_rest.smpl_jts_full

        # beta process preserves widths
        part_widths_from_v = vertice2capsule_fast(pred_vertices_full_align, pred_rest_jts_full)
        part_widths_l = torch.square(part_widths_from_v - gt_width) * smpl_weight * 10 
        part_widths_l = part_widths_l.mean()

        # beta process preserves skeleton
        pred_xyz = pred_xyz - pred_xyz[:, [0]]
        pred_xyz_struct = pred_xyz_struct - pred_xyz_struct[:, [0]]
        skeleton_beta_l = torch.abs(pred_xyz[:, :24].detach() - pred_xyz_struct)

        # beta process estimates correct vertices
        vertices_loss = torch.abs(pred_vertices_full - pred_vertices_struct).reshape(batch_size, -1) + \
            torch.abs(gt_smpl_out_rest.vertices - pred_vertices_full).reshape(batch_size, -1) * widths_weight * smpl_weight

        # beta process estimates correct widths ratio
        part_widths_ratio = output.pred_width_ratio
        
        part_spine_full_rest, _ = self.get_part_spine(pred_rest_jts_full)
        part_widths_ratio_from_v = part_widths_from_v / (torch.norm(part_spine_full_rest, dim=-1) + 1e-5)
        
        if not self.USE_WIDTH_SIGMA:
            width_ratio_l = torch.square(part_widths_ratio - gt_width_ratio) * smpl_weight * 10.0
            width_ratio_l2 = torch.square(part_widths_ratio_from_v - part_widths_ratio.detach()) * 10.0
            width_ratio_l = width_ratio_l.mean() + width_ratio_l2.mean()
        else:
            with_ratio_sigma = output.width_ratio_sigma
            width_ratio_l = weighted_laplace_loss(
                part_widths_ratio, gt_width_ratio, with_ratio_sigma, 
                smpl_weight, self.amp, self.size_average)
            width_ratio_l2 = weighted_laplace_loss(
                part_widths_ratio_from_v, part_widths_ratio.detach(), with_ratio_sigma,
                torch.ones_like(part_widths_ratio), self.amp, self.size_average)

            width_ratio_l = width_ratio_l + width_ratio_l2
        
        laplacian_loss = smpl_out_rest.laplace_loss * 0.1

        loss = skeleton_beta_l.mean() + vertices_loss.mean() * 10.0 + \
                   part_widths_l + laplacian_loss + width_ratio_l
        
        return loss, part_widths_l+width_ratio_l

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
    
    def get_bone_len(self, pred_xyz, pred_weight=None):
        batch_size = pred_xyz.shape[0]
        if pred_weight is None:
            pred_weight = torch.ones_like(pred_xyz)

        part_spines_3d = torch.zeros((batch_size, 28, 3), device=pred_xyz.device)
        part_spine_weight = torch.zeros((batch_size, 28, 3), device=pred_xyz.device)
        for i, k in enumerate(self.smpl29_parents):
            if i == 0:
                continue

            base_joints_3d = pred_xyz[:, i], pred_xyz[:, k] # batch x 3
            weight_joints = pred_weight[:, i], pred_weight[:, k]

            part_spines_3d[:, i-1] = base_joints_3d[1] - base_joints_3d[0]
            part_spine_weight[:, i-1] = weight_joints[1] * weight_joints[0]
        
        return part_spines_3d, part_spine_weight
    
    def rendering_criterion(self, output, labels):
        if isinstance(output, dict):
            output = edict(output)

        batch_size = output.pred_shape.shape[0]

        # SMPL params
        loss_beta = self.criterion_smpl_l1(
            output.pred_shape, labels['a_betas']
        )

        loss_beta_reg = torch.square(output.pred_shape).mean() * 0.001
        loss_beta = loss_beta + loss_beta_reg

        pred_theta = output.pred_theta_mats.reshape(batch_size, 24, 9)
        target_theta = labels['a_theta'].reshape(batch_size, 24, 9)
        loss_theta = self.criterion_smpl(
            pred_theta.reshape(batch_size, -1), 
            target_theta.reshape(batch_size, -1)
        )

        loss_twist = self.criterion_smpl(
            output.pred_phi.reshape(batch_size, -1) * labels['a_phi_weight'].reshape(batch_size, -1), 
            labels['a_phi'].reshape(batch_size, -1) * labels['a_phi_weight'].reshape(batch_size, -1)
        )

        pred_uvd = output.pred_uvd_jts.reshape(batch_size,-1,3)[:, :29]
        target_uvd = labels['a_target_uvd'][:, :29*3]
        target_uvd_weight = torch.ones_like(target_uvd)

        loss_uvd = weighted_laplace_loss(
                pred_uvd.reshape(batch_size, 29, 3),
                target_uvd.reshape(batch_size, 29, 3),
                output.sigma.reshape(batch_size, 29, 1),
                target_uvd_weight.reshape(batch_size, 29, -1),
                self.amp,
                self.size_average)
        
        pred_xyz = output.pred_xyz_jts_29.reshape(batch_size, -1, 3)
        pred_xyz_struct = output.pred_xyz_jts_29_struct.reshape(batch_size, -1, 3)
        target_xyz = labels['a_gt_xyz_29'].reshape(batch_size, -1, 3)

        # beta process preserves widths
        smpl_out_rest = output.smpl_out_rest
        gt_smpl_out_rest = output.gt_smpl_out_rest

        pred_rest_jts_full = smpl_out_rest.smpl_jts_full
        pred_vertices_full_align = smpl_out_rest.pred_vertices_full_align
        part_widths_from_v = vertice2capsule_fast(pred_vertices_full_align, pred_rest_jts_full)
        gt_width = labels['a_gt_part_widths']
        gt_width_ratio = labels['a_part_widths_ratio']
        part_widths_l = torch.square(part_widths_from_v - gt_width).mean() * 10.0

        pred_vertices_full = smpl_out_rest.pred_vertices_full
        pred_vertices_struct = smpl_out_rest.pred_vertices_struct
        vertices_loss = torch.abs(gt_smpl_out_rest.vertices - pred_vertices_struct).reshape(batch_size, -1).mean() + \
            torch.abs(gt_smpl_out_rest.vertices - pred_vertices_full).reshape(batch_size, -1).mean()
        
        pred_xyz = pred_xyz - pred_xyz[:, [0]]
        pred_xyz_struct = pred_xyz_struct - pred_xyz_struct[:, [0]]
        skeleton_beta_l = torch.abs(pred_xyz_struct[:, :24] - target_xyz[:, :24]).mean()

        part_spine_full_rest, _ = self.get_part_spine(pred_rest_jts_full)
        part_widths_ratio = output.pred_width_ratio
        part_widths_ratio_from_v = part_widths_from_v / (torch.norm(part_spine_full_rest, dim=-1) + 1e-5)
        
        # width_ratio_l = torch.square(part_widths_ratio_from_v - gt_width_ratio) + torch.square(part_widths_ratio - gt_width_ratio)
        if not self.USE_WIDTH_SIGMA:
            width_ratio_l = torch.square(part_widths_ratio_from_v - gt_width_ratio) + \
                                torch.square(part_widths_ratio - gt_width_ratio)
            
            width_ratio_l = width_ratio_l.mean() * 10.0

        else:
            with_ratio_sigma = output.width_ratio_sigma
            width_ratio_l = torch.square(part_widths_ratio_from_v - gt_width_ratio).mean() * 10.0
            width_ratio_l2 = weighted_laplace_loss(
                part_widths_ratio, gt_width_ratio, with_ratio_sigma,
                torch.ones_like(part_widths_ratio), self.amp, self.size_average)
            
            width_ratio_l = width_ratio_l + width_ratio_l2

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight + loss_twist * self.twist_weight
        loss += loss_uvd * self.uvd24_weight

        pred_scale = output.cam_scale.reshape(-1)
        target_scale = labels['a_scale_trans'].reshape(-1, 3)[:, 0]

        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        laplacian_loss = smpl_out_rest.laplace_loss 
    
        loss += (5 * scale_loss + 10 * vertices_loss + skeleton_beta_l + part_widths_l + width_ratio_l + laplacian_loss)

        loss_dict = {
            'uv2': loss_uvd.detach().cpu().numpy() * 0.01,
            'w2': (part_widths_l+width_ratio_l).detach().cpu().numpy(),
            's2': loss_beta.detach().cpu().numpy(),
        }

        return loss, loss_dict


