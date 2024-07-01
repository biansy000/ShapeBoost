"""Script for multi-gpu training."""
import os
import pickle as pk
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import clip_grad
from easydict import EasyDict as edict

from shapeboost.datasets import PW3D, SSP3DDataset, AGORA
from shapeboost.datasets.mix_dataset_shapy import MixDatasetShapy
from shapeboost.models import builder
from shapeboost.opt import cfg, logger, opt
from shapeboost.utils.env import init_dist
from shapeboost.utils.metrics import DataLogger, NullWriter, calc_coord_accuracy
from shapeboost.utils.transforms import flip, get_func_heatmap_to_coord
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shapeboost.beta_decompose.shape_utils import compute_pve_neutral_pose_scale_corrected, segment_iou
from shapeboost.utils.render_pytorch3d import render_mesh
import cv2
from shapeboost.models.layers.smpl.SMPL import SMPL_layer
from shapeboost.rendering.render_utils import augment_light, batch_add_rgb_background, inp_img_process
from shapeboost.rendering.render_configs import get_poseMF_shapeGaussian_cfg_defaults
from shapeboost.beta_decompose.beta_process2 import vertice2capsule_fast, part_names, spine_joints


# torch.set_num_threads(64)
num_gpu = torch.cuda.device_count()

rendering_cfgs = get_poseMF_shapeGaussian_cfg_defaults()


def _init_fn(worker_id):
    np.random.seed(opt.seed+worker_id)
    random.seed(opt.seed+worker_id)


def get_part_spine(pred_xyz, pred_weight=None):
    part_num = 20
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


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=64, epoch_num=0, dataset_name='', pass_eva=False):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
    kpt_pred = {}
    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    tot_len = len(gt_val_dataset) // batch_size // 4
    vis_iter_num = random.randint(0, tot_len-1)
    for it, (inps, labels, img_ids, bboxes) in enumerate(gt_val_loader):
    # for inps, labels, img_ids, bboxes in gt_val_loader:
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].float().cuda(opt.gpu)
            except AttributeError:
                assert k == 'type'

        output = m(inps, bboxes=bboxes, img_center=labels['img_center'])

        pred_xyz_jts_24 = output.pred_xyz_jts_29.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)
        pred_uvd_jts_29 = output.pred_uvd_jts.reshape(inps.shape[0], -1, 3)

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        pred_uvd_jts_29 = pred_uvd_jts_29.cpu().data.numpy()

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(
            pred_xyz_jts_17.shape[0], 17, 3)
        # pred_uvd_jts = pred_uvd_jts.reshape(
        #     pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(
            pred_xyz_jts_24.shape[0], 24, 3)
        pred_scores = output.maxvals.cpu().data[:, :29]

        for i in range(pred_xyz_jts_17.shape[0]):
            bbox = bboxes[i].tolist()
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
                'xyz_24': pred_xyz_jts_24[i],
                'uvd_jts': pred_uvd_jts_29[i, :24]
            }
        
        if it == vis_iter_num or (pass_eva and np.abs(it - vis_iter_num) < 2) or random.random() < 0.2:
            try_visualize(inps, output, labels, bboxes=bboxes, epoch_num=epoch_num, it=it, dataset_name=dataset_name, faces=m.module.smpl_layer.faces)

    if pass_eva:
        return None
    
    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = {}
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.update(kpt_pred)

        tot_err_17 = gt_val_dataset.evaluate_xyz_17(
            kpt_all_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))

        tot_err_24 = gt_val_dataset.evaluate_xyz_24(
            kpt_all_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))

        tot_err_uvd = gt_val_dataset.evaluate_uvd_24(
            kpt_all_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))

        return tot_err_17


def valid_ssp(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=1, epoch_num=0, dataset_name='', beta_model=None):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
    kpt_pred = {}
    m.eval()
    if beta_model is not None:
        beta_model.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    pve_neutral_errors = []

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    tot_len = len(gt_val_dataset) // batch_size // 4
    vis_iter_nums = [random.randint(0, tot_len-1) for _ in range(10)]
    for it, (inps, labels, img_ids, bboxes) in enumerate(gt_val_loader):
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            if isinstance(labels[k], torch.Tensor):
                labels[k] = labels[k].cuda(opt.gpu)

        use_gt_part_width = False
        part_widths_from_v_ratio = None
        if use_gt_part_width:
            gendered_smpl = m.module.smpl_male if labels['gender'][0] == 'm' else m.module.smpl_female
            outout = gendered_smpl.get_rest_pose(beta=labels['target_beta'], align=True)
            part_widths_from_v = vertice2capsule_fast(outout.vertices, outout.joints)

            part_spine, _ = get_part_spine(outout.joints)
            # part_spine2, _ = get_part_spine(output_backbone['pred_skeleton_new']*2.2)
            part_widths_from_v_ratio = part_widths_from_v / torch.norm(part_spine, dim=-1)

        output = m(inps, bboxes=bboxes, img_center=labels['img_center'], flip_test=True)#, part_widths_ratio=part_widths_from_v_ratio)
        test_betas = output.pred_shape
        if beta_model is not None:
            part_widths_from_v_ratio = output.pred_width_ratio
            beta_out = beta_model(part_widths_from_v_ratio, output.bone_len)
            test_betas = beta_out['pred_betas']

        pred_xyz_jts_24 = output.pred_xyz_jts_29.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()

        assert labels['gender'][0] == 'm' or labels['gender'][0] == 'f'
        gendered_smpl = m.module.smpl_male if labels['gender'][0] == 'm' else m.module.smpl_female
        pve_neutral_error = compute_pve_neutral_pose_scale_corrected(
            predicted_smpl_shape=test_betas, 
            target_smpl_shape=labels['target_beta'], 
            neutral_smpl=m.module.smpl_layer,
            gendered_smpl=gendered_smpl,
        )

        pve_neutral_errors.append(pve_neutral_error.mean())

        if it in vis_iter_nums or random.random() < 0.2:
            try_visualize(inps, output, labels, bboxes=bboxes, epoch_num=epoch_num, it=it, dataset_name=dataset_name, faces=m.module.smpl_layer.faces)

    with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(pve_neutral_errors, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = []
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred += kpt_pred

        m_pve_error = np.array(kpt_all_pred).mean()
        print(f'm_pve_error = {m_pve_error}, {np.array(pve_neutral_errors).mean()}')

        return m_pve_error


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    if opt.seed is not None:
        setup_seed(opt.seed)

    if opt.launcher == 'slurm':
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):
    if opt.seed is not None:
        setup_seed(opt.seed)

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    if not opt.log:
        logger.setLevel(50)
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    opt.nThreads = int(opt.nThreads / num_gpu)

    # Model Initialize
    m = preset_model(cfg)

    m.cuda(opt.gpu)
    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu])

    # beta_model = Beta_regression_bonelen_sratio()
    # beta_model.load_state_dict(torch.load('data/beta_regressor_model_23.pth'))
    # beta_model.cuda(opt.gpu)
    # beta_model = torch.nn.parallel.DistributedDataParallel(beta_model, device_ids=[opt.gpu])
    beta_model = None

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    print('nthreads', opt.nThreads)
    # gt val dataset
    gt_val_dataset_h36m = MixDatasetShapy(
        cfg=cfg,
        train=True)

    gt_val_dataset_3dpw = PW3D(
        cfg=cfg,
        ann_file='3DPW_test_new.json',
        train=False)

    gt_val_dataset_agora = AGORA(
        cfg=cfg,
        ann_file='data/AGORA/annotations/validation_all_SMPL_withjv_withkid_valid.pt',
        train=False)

    # gt_val_dataset_ssp3d = SSP3DDataset(
    #     cfg=cfg, ssp3d_dir_path='data/ssp_3d/ssp_3d')
    
    # gt_val_dataset_3dpw = PW3D(
    #     cfg=cfg,
    #     ann_file='3DPW_test_new.json',
    #     train=True)

    opt.trainIters = 0

    i = 0
    opt.epoch = i

    with torch.no_grad():
        gt_tot_err_h36m = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord, epoch_num=i, dataset_name='h36m')
        gt_tot_err_3dpw = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord, epoch_num=i, dataset_name='3dpw')
        gt_tot_err_agora = validate_gt(m, opt, cfg, gt_val_dataset_agora, heatmap_to_coord, epoch_num=i, dataset_name='agora')

        # m_pve_error = valid_ssp(m, opt, cfg, gt_val_dataset_ssp3d, heatmap_to_coord, epoch_num=i, dataset_name='ssp3d', beta_model=beta_model)
        # _ = validate_gt(m, opt, cfg, gt_val_dataset_3dpw_vis, heatmap_to_coord, epoch_num=i, dataset_name='3dpw_scale', pass_eva=True)


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL)
    # beta_net_pretrained = 'exp/mix_smpl_simulated/256x192_adam_lr1e-3-hrw48_reg_cam_2x_sratio_ddp.yaml-theta_aug_bp2_20_1view_rle/model_24.pth'

    # beta_net_pretrained
    
    cfg.MODEL.PRETRAINED = 'exp/mix_smpl_shapy/256x192_adam_lr1e-3-hrw48_cam_2x_sratio_render_ddp_finer_analytical.yaml-try_1e-4amass/model_22.pth'
    # beta_net_pretrained = cfg.MODEL.PRETRAINED
    if len(cfg.MODEL.PRETRAINED) > 0:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        save_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if type(save_dict) == dict:
            # print('yes')
            model_dict = save_dict['model']
            model.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(save_dict, strict=False)

    # save_dict = torch.load(beta_net_pretrained, map_location='cpu')
    # new_state_dict = {}
    # for k, v in save_dict.items():
    #     if 'part_encoder' in k or 'beta_regressor' in k:
    #         new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)
    # model.beta_regressor.load_state_dict(save_dict.beta_regressor, strict=True)

    return model



def try_visualize(inps, preds, labels, faces, bboxes, epoch_num, it=0, dataset_name=''):
    batch_size = inps.shape[0]
    pa_path = 'exp/visualize/shape_vis_ddp_val'
    faces = torch.from_numpy(faces.astype(np.int32))

    pred_vertices = preds.pred_vertices.detach()
    # pred_vertices = labels['target_vertices'].detach()
    transl = preds.transl.detach()
    # if bboxes is not None:
    #     bboxes = bboxes.detach().cpu().numpy().reshape(batch_size, 4)
    # else:
    #     bboxes = np.zeros((batch_size, 4))
    #     bboxes[:, ]

    f = 1000.0
    for bid in range(min(batch_size, 10)):

        saved_path = [f'{pa_path}/{epoch_num}_{dataset_name}_{it}_{bid:03d}_mesh.jpg']
        saved_path_uv = [f'{pa_path}/{epoch_num}_{dataset_name}_{it}_{bid:03d}_uv.jpg']

        new_focal = 1000.0

        ori_imgs = inps[bid].detach().cpu().numpy().transpose(1, 2, 0)
        ori_imgs = ori_imgs * np.array([0.225, 0.224, 0.229]) + np.array([0.406, 0.457, 0.480])
        ori_imgs = ori_imgs * 255.0

        focal = f
        bs = bid
        verts_batch = pred_vertices[[bs]]
        transl_batch = transl[[bs]]
        # transl_batch[:, :2] = 0

        color_batch = render_mesh(
            vertices=verts_batch, faces=faces,
            translation=transl_batch,
            focal_length=focal, height=256, width=256)

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()

        input_img = ori_imgs
        alpha = 0.9
        image_vis = alpha * color[:, :, :3] * valid_mask + (
            1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)



if __name__ == "__main__":
    main()
