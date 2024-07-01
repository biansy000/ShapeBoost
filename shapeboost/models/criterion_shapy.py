import torch
import torch.nn as nn
import numpy as np
import math
import pickle as pk
from easydict import EasyDict as edict

from .layers.smpl.SMPL import SMPL_layer
from shapeboost.beta_decompose.beta_process2 import vertice2capsule_fast, part_names, spine_joints, part_names_ids, used_part_seg, default_lbs_weights, part_root_joints
from shapeboost.utils.shapy_utils import BodyMeasurements, Polynomial, Beta2Meausrement, Converter
import torch.nn.functional as F
from .builder import LOSS


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


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


class ShapyLoss(nn.Module):
    def __init__(self,):
        super(ShapyLoss, self).__init__()

        meas_definition_path = 'data/pretrained_models/shapy_models/utility_files/measurements/measurement_defitions.yaml'
        meas_vertices_path = 'data/pretrained_models/shapy_models/utility_files/measurements/smpl_measurement_vertices.yaml'

        self.bm = BodyMeasurements(
            {'meas_definition_path': meas_definition_path,
                'meas_vertices_path': meas_vertices_path},
        )

        self.s2a_male = Polynomial()
        self.s2a_female = Polynomial()
        self.s2m = Beta2Meausrement()

        self.beta_converter = Converter()
    
    def _initialize(self):
        s2a_male_path = 'data/pretrained_models/shapy_models/caesar-male_smplx-neutral-10betas/last_simple.ckpt'
        s2a_female_path = 'data/pretrained_models/shapy_models/caesar-female_smplx-neutral-10betas/last_simple.ckpt'
        s2m_path = 'data/pretrained_models/shapy_models/width_bonelen2measurements_2layers_48.pth'
        
        self.s2a_male.load_state_dict(torch.load(s2a_male_path, map_location='cpu'))
        self.s2a_female.load_state_dict(torch.load(s2a_female_path, map_location='cpu'))
        self.s2m.load_state_dict(torch.load(s2m_path, map_location='cpu'))

        self.s2a_male.eval()
        self.s2a_female.eval()
        self.s2m.eval()

    def get_attr_pred(self, beta_smpl, triangles, part_widths, bone_len):
        x = self.beta_converter.SMPLb2SMPLXb_torch(beta_smpl)
        bm_out = self.bm(triangles)
        attr_male = self.s2a_male(x)
        attr_female = self.s2a_female(x)
        measure_pred = self.s2m(part_widths, bone_len)

        attr_pred = {
            'mass': bm_out['mass']['tensor'], 
            'height': bm_out['height']['tensor'], 
            'chest': measure_pred[:, 0], 
            'waist': measure_pred[:, 1], 
            'hips': measure_pred[:, 2],
            'attribute': torch.cat([attr_male, attr_female], dim=1)
        }

        return attr_pred

    def get_height(self, triangles):
        bm_out = self.bm(triangles)
        height = bm_out['height']['tensor'].reshape(-1, 1)

        return height
    
    def forward(self, beta_smpl, triangles, part_widths, bone_len, attr_gt, attr_w, v_shaped=None, return_val=False):
        loss_weight = torch.ones(35, device=part_widths.device)
        loss_weight[0] = 0.03
        loss_weight[1] = 10

        if beta_smpl is not None:
            batch_size = beta_smpl.shape[0]
            x = self.beta_converter.SMPLb2SMPLXb_torch(beta_smpl)
            x_inp = torch.clamp(x, min=-10, max=10)
        else:
            batch_size = v_shaped.shape[0]
            x = self.beta_converter.SMPLv2SMPLXb_torch(v_shaped)
            loss_weight[2:5] = 0

            x_inp = torch.clamp(x, min=-10, max=10)

        bm_out = self.bm(triangles)
        attr_male = self.s2a_male(x_inp)
        attr_female = self.s2a_female(x_inp)
        measure_pred = self.s2m(part_widths, bone_len)
        
        attr_male = torch.clamp(attr_male, min=0, max=5)
        attr_female = torch.clamp(attr_female, min=0, max=5)

        attr_pred = torch.cat([
            bm_out['mass']['tensor'].reshape(batch_size, 1),
            bm_out['height']['tensor'].reshape(batch_size, 1),
            measure_pred[:, [0]], 
            measure_pred[:, [1]], 
            measure_pred[:, [2]],
            attr_male, 
            attr_female
        ], dim=1)

        # print(bm_out['mass']['tensor'], attr_gt[:, 0])
        attr_w_2 = (attr_w * attr_gt)
        assert (attr_w_2 >= -1e-5).all()
        
        loss = torch.square(attr_pred - attr_gt) * loss_weight * attr_w
        
        reg_loss1 = torch.square(x) * 0.001 * ((attr_w > 0).any(dim=-1, keepdim=True) * 1.0)
        reg_loss2 = torch.square( F.relu(attr_pred[:, 1] - 3) ) + torch.square( F.relu(0.6 - attr_pred[:, 1]) )
        reg_loss3 = torch.square( F.relu(attr_pred[:, 0]*0.01 - 3) ) + torch.square( F.relu(0.3 - attr_pred[:, 0]*0.01) )
        reg_loss = reg_loss1.mean() + reg_loss2.mean() + reg_loss3.mean()
        if loss.sum() / attr_w.sum() > 100:
            for i, item in enumerate(loss):
                print(item, bm_out['mass']['tensor'][i], bm_out['height']['tensor'][i])

        
        if attr_w.sum() > 0:
            loss = loss.sum() / attr_w.sum() + reg_loss
        else:
            loss = loss.sum()
        
        if return_val:
            return loss, attr_pred
        else:
            return loss



@LOSS.register_module
class L1LossDimSMPLShapeRatioAnalytical(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLShapeRatioAnalytical, self).__init__()

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
        self.shapy_loss = ShapyLoss()
        self.shapy_loss._initialize()
        self.size_average = size_average

        print('size_average', size_average)
        self.reduce = reduce

        self.amp = 1 / math.sqrt(2 * math.pi)

        self.smpl29_parents = [-1,  0,  0,  0,  1,  2,  3,
            4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11
        ]

        model_path = 'model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        with open(model_path, 'rb') as smpl_file:
            smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))
        
        self.register_buffer('faces_tensor',
                to_tensor(to_np(smpl_data.f, dtype=np.int64), dtype=torch.long))
        
        self.register_buffer('lbs_weights', default_lbs_weights.clone())

    def forward(self, output, labels, num_epoch=0):
        if isinstance(output, dict):
            output = edict(output)

        smpl_weight = labels['target_smpl_weight']
        batch_size = smpl_weight.shape[0]
        width_changed = labels['width_changed'].reshape(batch_size, 1)
        width_not_changed = 1 - width_changed

        # Hybrik Loss, SMPL params
        loss_beta = self.criterion_smpl_l1(
            output.pred_shape * smpl_weight * width_not_changed, 
            labels['target_beta'] * smpl_weight * width_not_changed
        )
        loss_beta_reg = torch.square(output.pred_shape) * 1e-4
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

        pred_bones, _ = self.get_bone_len(pred_xyz_refined*2.2)
        # pred_bones_uvd, _ = self.get_bone_len( output.pred_uvd_jts.reshape(batch_size,-1,3)*2.2 )
        camDepth = 1000.0 / (256 * output.cam_scale + 1e-9).reshape(batch_size, 1, 1)
        pred_uvd = output.pred_uvd_jts.reshape(batch_size, 29, 3)
        pred_z_fromuvd = pred_uvd[:, :, 2:] * 2.2
        pred_xy_fromuvd = (pred_uvd[:, :, :2] * 256 / 1000.0) * (pred_z_fromuvd + camDepth)
        pred_xyz_fromuvd = torch.cat([pred_xy_fromuvd, pred_z_fromuvd], dim=-1)
        pred_bones_uvd, _ = self.get_bone_len(pred_xyz_fromuvd)
        target_bones, target_bone_weight = self.get_bone_len(target_xyz*2.2, target_xyz_weight)
        
        pred_depth_ratio = pred_bones_uvd[:, :, 2] / (torch.norm(pred_bones, dim=-1) + 1e-5)
        gt_depth_ratio = target_bones[:, :, 2] / (torch.norm(target_bones, dim=-1) + 1e-5)
        reg_z_loss = torch.square(pred_depth_ratio - gt_depth_ratio) * target_bone_weight[:, :, 2]
        reg_z_loss2 = torch.square(pred_bones_uvd[:, :, 2] - target_bones[:, :, 2]) * target_bone_weight[:, :, 2]
        bone_loss = torch.square(torch.norm(pred_bones_uvd, dim=-1) - torch.norm(target_bones, dim=-1)) * \
            ((target_bone_weight.sum(dim=-1) > 2) * 1.0) * smpl_weight
        reg_z_loss = reg_z_loss.mean() * 0.03 + reg_z_loss2.mean() * 0.1 + bone_loss.mean() * 0.01

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
        
        loss_width, width_ratio_l, shapy_loss = self.get_widths_loss(
            output=output, labels=labels, smpl_weight=smpl_weight,
            widths_weight=width_not_changed,
        )

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight + loss_twist * self.twist_weight
        loss += (reg_z_loss + loss_width)
        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.reshape(batch_size, -1).sum(dim=1) > 3) * 1.0 # batch
        smpl_weight = smpl_weight.unsqueeze(1)
        pred_scale = output.cam_scale * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight

        scale_loss = self.criterion_smpl(pred_scale*width_not_changed, target_scale*width_not_changed)

        cam_range = torch.maximum(labels['width_changed_ratio'], 1/(labels['width_changed_ratio']+1e-5)).reshape(batch_size, 1)
        target_scale_upper_bound = target_scale * cam_range
        target_scale_lower_bound = target_scale / cam_range
        scale_loss2 = F.relu(pred_scale - target_scale_upper_bound - 1e-3).mean() + F.relu(target_scale_lower_bound - pred_scale - 1e-3).mean()

        loss += (10 * scale_loss + scale_loss2)
        
        loss_dict = {
            'w_l': loss_width.detach().cpu().numpy(),
            'w_r_l': width_ratio_l.mean().detach().cpu().numpy(),
            'sp': shapy_loss.detach().cpu().numpy(),
            's': scale_loss.detach().cpu().numpy()
        }

        return loss, loss_dict
      
    def get_widths_loss(self, output, labels, smpl_weight, widths_weight):
        # beta & pred part_widths compatible + beta & pred xyz_29 compatible
        # pred_widths ratio is compatible with gt_width_ratio
        # all unit=m
        gt_smpl_out_rest = output.gt_smpl_out_rest
        smpl_out_rest = output.smpl_out_rest
        smpl_out_raw = output.smpl_out_raw
        # smpl_out_raw_detached = output.out_raw_detached
        if isinstance(gt_smpl_out_rest, dict):
            gt_smpl_out_rest = edict(gt_smpl_out_rest)
        if isinstance(smpl_out_rest, dict):
            smpl_out_rest = edict(smpl_out_rest)
        if isinstance(smpl_out_raw, dict):
            smpl_out_raw = edict(smpl_out_raw)
        # if isinstance(smpl_out_raw_detached, dict):
        #     smpl_out_raw_detached = edict(smpl_out_raw_detached)
        pred_xyz = output.pred_xyz_jts_29.reshape(-1, 29, 3) *2.2
        batch_size = pred_xyz.shape[0]

        width_changed = labels['width_changed'].reshape(batch_size, 1)
        width_not_changed = 1 - width_changed
        shapy_weight = (labels['attributes_w'].sum(dim=-1, keepdim=True) > 0) * 1.0

        beta_reg_loss = torch.square(smpl_out_rest.pred_beta_raw).mean() * 0.01

        # beta process preserves skeleton
        pred_xyz_struct = output.pred_xyz_jts_29_struct.reshape(batch_size, -1, 3) * 2.2
        pred_xyz = pred_xyz - pred_xyz[:, [0]]
        pred_xyz_struct = pred_xyz_struct - pred_xyz_struct[:, [0]]
        xyz_score = torch.clamp(1 - output.sigma.detach() * 3, min=0, max=1)
        skeleton_beta_l = torch.abs(pred_xyz[:, :24].detach() - pred_xyz_struct[:, :24]) * xyz_score[:, :24]
        skeleton_beta_l = skeleton_beta_l.mean()

        # beta process preserves widths
        pred_vertices_struct = smpl_out_rest.pred_vertices_struct
        pred_rest_jts_struct = smpl_out_rest.pred_rest_jts
        gt_widths = labels['part_widths']
        part_widths_from_v = vertice2capsule_fast(pred_vertices_struct, pred_rest_jts_struct, lbs_weights=self.lbs_weights)
        pred_width = output.pred_width
        part_widths_l = torch.square(part_widths_from_v - pred_width[:, part_names_ids].detach()) + torch.square(pred_width[:, part_names_ids] - gt_widths) * smpl_weight * 0.1
        part_widths_l = part_widths_l.mean()

        # beta process estimates correct vertices
        pred_vertices_raw = smpl_out_raw.v_shaped
        vertices_loss = torch.abs(gt_smpl_out_rest.vertices - pred_vertices_struct).reshape(batch_size, -1) * widths_weight * smpl_weight + \
            torch.abs(pred_vertices_struct - pred_vertices_raw.detach()).reshape(batch_size, -1) + \
            torch.square(smpl_out_rest.vertices_lsq - pred_vertices_raw).reshape(batch_size, -1) * 0.1 #+\ # reg

        # beta process estimates correct widths ratio
        gt_width_ratio = labels['width_ratio']
        part_widths_ratio = output.pred_width_ratio[:, part_names_ids]
        
        part_spine_full_rest, _ = self.get_part_spine(pred_rest_jts_struct)
        part_widths_ratio_from_v = part_widths_from_v / (torch.norm(part_spine_full_rest, dim=-1) + 1e-5)
        
        width_ratio_l = torch.square(part_widths_ratio - gt_width_ratio) * smpl_weight * 3.0
        width_ratio_l2 = torch.square(part_widths_ratio_from_v - part_widths_ratio.detach()) * 1.0
        width_ratio_l = width_ratio_l.mean() + width_ratio_l2.mean()

        shapy_loss, pred_attr = self.shapy_loss(
            output.pred_shape, pred_vertices_struct[:, self.faces_tensor], part_widths_from_v, torch.norm(part_spine_full_rest, dim=-1),
            labels['attributes'], labels['attributes_w'], return_val=True
        )

        pred_uv = output.pred_uvd_jts.reshape(batch_size,-1,3)[:, :29, :2].detach()
        target_uv = labels['target_uvd_29'].reshape(batch_size,-1,3)[:, :29, :2]
        target_uv_weight = labels['target_weight_29'].reshape(batch_size,-1,3)[:, :29, :2] * smpl_weight.unsqueeze(-1)
        gt_part_uv_vec, part_uv_weight = self.get_part_spine(target_uv, target_uv_weight, is_2d=True)
        pred_part_uv_vec, _ = self.get_part_spine(pred_uv, is_2d=True)
        part_uv_weight = (part_uv_weight > 0).all(dim=-1) * 1.0
        v2d_loss = 0
        for i, n in enumerate(part_names):
            part_seg = used_part_seg[n]
            root_joint = part_root_joints[n]
            part_lbs_weights = self.lbs_weights[part_seg, root_joint]
            pred_v2d_part = output.pred_vertices_2d[:, part_seg]
            gt_v2d_part = labels['pred_2dv'][:, part_seg]
            pred_widths2d = self.get_2d_widths(pred_v2d_part, pred_uv[:, root_joint], pred_part_uv_vec[:, i], part_lbs_weights)
            gt_widths2d = self.get_2d_widths(gt_v2d_part, target_uv[:, root_joint], gt_part_uv_vec[:, i], part_lbs_weights)

            l = torch.square(pred_widths2d - gt_widths2d) * part_uv_weight[:, i] * labels['pred_2dv_weight'].reshape(batch_size)
            # print(pred_widths2d[0], gt_widths2d[0])
            v2d_loss += l.mean() * 10.0

        # height_gt = self.shapy_loss.get_height(gt_smpl_out_rest.vertices[:, self.faces_tensor])
        # height_weight = labels['is_agora'].reshape(batch_size, 1) * width_not_changed
        # height_loss0 = torch.square(height_gt - output.pred_height) * height_weight
        # height_loss1 = (output.pred_height - labels['attributes'][:, [1]]) * labels['attributes_w'][:, [1]]
        # height_loss2 = torch.square(pred_attr[:, [1]] - output.pred_height.detach()) * 3.0
        # height_loss = output.height_loss * 0.1 + height_loss0.mean() * 3.0 + height_loss1.mean() * 3.0 + height_loss2.mean()
        
        loss = skeleton_beta_l + vertices_loss.mean() * 3.0 + \
                   part_widths_l + width_ratio_l + shapy_loss + v2d_loss + beta_reg_loss
        
        return loss, part_widths_l+width_ratio_l, v2d_loss+shapy_loss

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
        gt_smpl_out_rest = output.gt_smpl_out_rest
        smpl_out_rest = output.smpl_out_rest
        smpl_out_raw = output.smpl_out_raw

        pred_rest_jts = smpl_out_rest.pred_rest_jts
        pred_vertices_struct = smpl_out_rest.pred_vertices_struct
        pred_vertices_raw = smpl_out_raw.v_shaped
        part_widths_from_v = vertice2capsule_fast(pred_vertices_struct, pred_rest_jts)
        gt_width = labels['a_gt_part_widths']
        gt_width_ratio = labels['a_part_widths_ratio']
        pred_part_widths = output.pred_width[:, part_names_ids]
        part_widths_l = torch.square(part_widths_from_v - gt_width).mean() * 3.0 + \
            torch.square(pred_part_widths - gt_width).mean() * 3.0

        vertices_loss = torch.abs(gt_smpl_out_rest.vertices - pred_vertices_struct).reshape(batch_size, -1).mean() + \
            torch.abs(gt_smpl_out_rest.vertices - pred_vertices_raw).reshape(batch_size, -1).mean() + \
            torch.square(smpl_out_rest.vertices_lsq - pred_vertices_raw).reshape(batch_size, -1).mean() * 0.01 # reg
        
        pred_xyz = pred_xyz - pred_xyz[:, [0]]
        pred_xyz_struct = pred_xyz_struct - pred_xyz_struct[:, [0]]
        skeleton_beta_l = torch.abs(pred_xyz_struct[:, :24] - target_xyz[:, :24]).mean()

        part_spine_full_rest, _ = self.get_part_spine(pred_rest_jts)
        part_widths_ratio_from_v = part_widths_from_v / (torch.norm(part_spine_full_rest, dim=-1) + 1e-5)
        
        part_widths_ratio = output.pred_width_ratio[:, part_names_ids]
        width_ratio_l = torch.square(part_widths_ratio_from_v - gt_width_ratio) + torch.square(part_widths_ratio - gt_width_ratio)
        width_ratio_l = width_ratio_l.mean() * 3.0

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight + loss_twist * self.twist_weight
        loss += loss_uvd * self.uvd24_weight

        pred_scale = output.cam_scale.reshape(-1)
        target_scale = labels['a_scale_trans'].reshape(-1, 3)[:, 0]

        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        # height_pred = self.shapy_loss.get_height(pred_vertices_struct[:, self.faces_tensor])
        # height_gt = self.shapy_loss.get_height(gt_smpl_out_rest.vertices[:, self.faces_tensor])
        # height_loss = torch.square(height_pred - output.pred_height).mean() + torch.square(height_gt - output.pred_height).mean() * 3.0
        # height_loss = height_loss + output.height_loss
    
        loss += (10 * scale_loss + 3 * vertices_loss + skeleton_beta_l + part_widths_l + width_ratio_l)

        loss_dict = {
            'uv2': loss_uvd.detach().cpu().numpy() * 0.01,
            'w2': (part_widths_l+width_ratio_l).detach().cpu().numpy(),
            's2': loss_beta.detach().cpu().numpy(),
        }

        return loss, loss_dict

    def get_part_spine(self, pred_xyz, pred_weight=None, is_2d=False):
        part_num = len(part_names)
        batch_size = pred_xyz.shape[0]
        if pred_weight is None:
            pred_weight = torch.ones_like(pred_xyz)

        last_dim = 2 if is_2d else 3
        part_spines_3d = torch.zeros((batch_size, part_num, last_dim), device=pred_xyz.device)
        part_spine_weight = torch.zeros((batch_size, part_num, last_dim), device=pred_xyz.device)
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

    def get_2dv(self, pred_vertices, cam_params=None, transl=None):
        if cam_params is not None:
            v_abs_d = pred_vertices[:, :, [2]] + 1000 / (256.0 * cam_params[:, [0]].unsqueeze(-1))
            transl_xy = cam_params[:, 1:]
        else:
            v_abs_d = pred_vertices[:, :, [2]] + transl[:, [2]].unsqueeze(1)
            transl_xy = transl[:, :2]
        
        pred_2dv = (pred_vertices[:, :, :2] + transl_xy.reshape(-1, 1, 2)) / v_abs_d / (256.0 / 1000.0)
        return pred_2dv

    def get_2d_widths(self, v_2d, base_joint, part_spine, part_lbs_weights):
        # v_2d: batch x ? x 2 , uv1: batch x 2, uv2: batch x 2
        part_spine_normed = part_spine / (torch.norm(part_spine, dim=-1, keepdim=True) + 1e-5) # batch x 2
        part_spine_normed = part_spine_normed.unsqueeze(dim=1) # batch x 1 x 2

        vec = v_2d - base_joint.unsqueeze(1)
        # dis2spine = torch.cross(part_spine_normed.expand(-1, vec.shape[1], -1), vec, dim=-1)
        # dis2spine = torch.norm(dis2spine, dim=-1)
        dis2spine = part_spine_normed[:, :, 0] * vec[:, :, 1] - part_spine_normed[:, :, 1] * vec[:, :, 0]
        dis2spine = torch.abs(dis2spine)
        mean_dis2spine = (dis2spine * part_lbs_weights).sum(dim=-1) / part_lbs_weights.sum()

        return mean_dis2spine


