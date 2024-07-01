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

from shapeboost.datasets.HBW_dataset import HBWDataset
from shapeboost.datasets import PW3D, SSP3DDataset,  AGORA, MixDatasetShapy
from shapeboost.models import builder
from shapeboost.opt import cfg, logger, opt
from shapeboost.utils.env import init_dist
from shapeboost.utils.metrics import DataLogger, NullWriter, calc_coord_accuracy
from shapeboost.utils.transforms import flip, get_func_heatmap_to_coord
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shapeboost.beta_decompose.shape_utils import compute_pve_neutral_pose_scale_corrected, segment_iou
from shapeboost.utils.shapy_utils import HBWErrorCalculator
from shapeboost.utils.render_pytorch3d import render_mesh
import cv2
from shapeboost.models.layers.smpl.SMPL import SMPL_layer

from shapeboost.rendering.renderer import TexturedIUVRenderer
from shapeboost.rendering.render_utils import augment_light, batch_add_rgb_background, inp_img_process
from shapeboost.rendering.render_configs import get_poseMF_shapeGaussian_cfg_defaults

# torch.set_num_threads(64)
num_gpu = torch.cuda.device_count()

rendering_cfgs = get_poseMF_shapeGaussian_cfg_defaults()


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def train(opt, train_loader, m, criterion, optimizer, writer, epoch_num, pytorch3d_renderer=None, optimizer2=None):
    loss_logger = DataLogger()
    acc_uvd_29_logger = DataLogger()
    acc_xyz_17_logger = DataLogger()
    m.train()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    depth_dim = cfg.MODEL.EXTRA.get('DEPTH_DIM')
    hm_shape = (hm_shape[1], hm_shape[0], depth_dim)
    root_idx_17 = train_loader.dataset.root_idx_17

    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    for j, (inps, labels, _, bboxes) in enumerate(train_loader):
        # m.train()
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu).requires_grad_() for inp in inps]
        else:
            inps = inps.cuda(opt.gpu).requires_grad_()

        for k, _ in labels.items():
            if not isinstance(labels[k], list):
                labels[k] = labels[k].float().cuda(opt.gpu)

        output = m(inps, bboxes=bboxes.cuda(), img_center=labels['img_center'], gt_beta=labels['target_beta'])

        loss, loss_dict = criterion(output, labels, num_epoch=epoch_num)

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_17 = output.pred_xyz_jts_17
        label_masks_29 = labels['target_weight_29']
        label_masks_17 = labels['target_weight_17']

        acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts.detach().cpu(), labels['target_uvd_29'].cpu(), label_masks_29.cpu(), hm_shape, num_joints=29)
        acc_xyz_17 = calc_coord_accuracy(pred_xyz_jts_17.detach().cpu(), labels['target_xyz_17'].cpu(), label_masks_17.cpu(), hm_shape, num_joints=17, root_idx=root_idx_17)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_uvd_29_logger.update(acc_uvd_29, batch_size)
        acc_xyz_17_logger.update(acc_xyz_17, batch_size)

        optimizer.zero_grad()
        loss.backward()
        for group in optimizer.param_groups:
            for param in group["params"]:
                clip_grad.clip_grad_norm_(param, 5)
        optimizer.step()

        # try_visualize(
        #     inps, output, labels, bboxes=bboxes, epoch_num=0, it=j, dataset_name='train', 
        #     faces=m.module.smpl_layer.faces)
        # if j % 2 == 1:
        if True:
            l2, loss_dict2 = train_synthetic(m, labels, criterion, optimizer, pytorch3d_renderer=pytorch3d_renderer, optimizer2=optimizer2)
            loss_dict.update(loss_dict2)
        else:
            l2 = 0

        opt.trainIters += 1
        if opt.log:
            summary_str = 'loss: {loss:.3f} | accuvd: {accuvd29:.3f} | acc17: {acc17:.3f}'.format(
                    loss=loss_logger.avg,
                    accuvd29=acc_uvd_29_logger.avg,
                    acc17=acc_xyz_17_logger.avg)
                
            for k, v in loss_dict.items():
                summary_str += f' | {k}: {int(v*1e2)}'
            
            summary_str += f' | l2: {int(l2*1e2)}'

            # TQDM
            train_loader.set_description(
                summary_str
            )

    if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_xyz_17_logger.avg


def train_synthetic(m, labels, criterion, optimizer, pytorch3d_renderer, epoch_num=0, optimizer2=None):
    batch_size = labels['a_pred_xyz_29'].shape[0]
    device = labels['a_pred_xyz_29'].device
    with torch.no_grad():
        lights_rgb_settings = augment_light(batch_size=1,
                                        device=device,
                                        rgb_augment_config=rendering_cfgs.TRAIN.SYNTH_DATA.AUGMENT.RGB)

        renderer_output = pytorch3d_renderer(vertices=labels['a_vertices'].reshape(batch_size, 6890, 3),
                                        textures=labels['a_texture'].reshape(batch_size, 1200, 800, 3),
                                        cam_t=labels['a_trans_l'].reshape(batch_size, 3),
                                        lights_rgb_settings=lights_rgb_settings,)
                                        # img_center=labels['a_img_center'])

        iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
        iuv_in[:, 1:, :, :] = iuv_in[:, 1:, :, :] * 255
        iuv_in = iuv_in.round()
        rgb_in = renderer_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)

        seg = iuv_in[:, 0, :, :]

        rgb_in = batch_add_rgb_background(backgrounds=labels['a_background'],
                                                    rgb=rgb_in,
                                                    seg=seg)

        rgb_in = inp_img_process(rgb_in).reshape(batch_size, 3, 256, 256)

    output = m(rgb_in, gt_beta=labels['a_betas'])

    loss, loss_dict = criterion.rendering_criterion(output, labels)

    if optimizer2 is None:
        optimizer.zero_grad()
        loss.backward()
        for group in optimizer.param_groups:
            for param in group["params"]:
                clip_grad.clip_grad_norm_(param, 5)
        optimizer.step()
    else:
        optimizer2.zero_grad()
        loss.backward()
        for group in optimizer2.param_groups:
            for param in group["params"]:
                clip_grad.clip_grad_norm_(param, 5)
        optimizer2.step()

    return loss.detach().cpu().numpy(), loss_dict


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=64, epoch_num=0, dataset_name='', pass_eva=False):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
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

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()

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
                'xyz_24': pred_xyz_jts_24[i]
            }
        
        if it == vis_iter_num or (pass_eva and np.abs(it - vis_iter_num) < 2):
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

        return tot_err_17


def valid_ssp(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=1, epoch_num=0, dataset_name=''):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
    kpt_pred = {}
    m.eval()

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

        output = m(inps, bboxes=bboxes, img_center=labels['img_center'])

        pred_xyz_jts_24 = output.pred_xyz_jts_29.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)

        test_betas = output.pred_shape

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

        if it in vis_iter_nums:
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



def valid_hbw(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=1, epoch_num=0, dataset_name=''):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, sampler=gt_val_sampler, pin_memory=True)
    kpt_pred = {}
    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    p2p_error_list = []

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)
    
    error_calculator = HBWErrorCalculator()

    tot_len = len(gt_val_dataset) // batch_size // 4
    vis_iter_nums = [random.randint(0, tot_len-1) for _ in range(10)]
    for it, (inps, labels, img_ids, img_paths, bboxes) in enumerate(gt_val_loader):
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            if isinstance(labels[k], torch.Tensor):
                labels[k] = labels[k].cuda(opt.gpu)

        output = m(inps, bboxes=bboxes, img_center=labels['img_center'], clothed=True)
        
        if 'smplx_out_rest' in output:
            test_betas = output.smplx_out_rest.pred_betas
            pred_vshaped = m.module.smplx_layer.get_rest_pose(test_betas)['vertices']
            pred_vshaped = pred_vshaped.cpu().numpy()[0]

            img_path = img_paths[0].split('/')
            img_path = '/'.join(img_path[3:])
            p2p_error = error_calculator.calc_error(img_path, pred_vshaped, smplx=True)

            p2p_error_list.append(p2p_error)
            
            theta = output.pred_theta_mats.reshape(-1, 24, 9)
            smplx_output = m.module.smplx_layer(
                betas=test_betas,
                global_orient=theta[:, [0]],
                body_pose=theta[:, 1:22]
            )

            preds = edict(
                pred_vertices=smplx_output.vertices,
                transl=output.transl,
                pred_uvd_jts=output.pred_uvd_jts
            )

            if it in vis_iter_nums:
                try_visualize(inps, preds, labels, bboxes=bboxes, epoch_num=epoch_num, it=it, dataset_name=dataset_name, faces=m.module.smplx_layer.faces)
        else:
            test_betas = output.pred_shape

            pred_vshaped = m.module.smpl_layer.get_rest_pose(test_betas)['vertices']
            pred_vshaped = pred_vshaped.cpu().numpy()[0]

            img_path = img_paths[0].split('/')
            img_path = '/'.join(img_path[3:])
            # print(img_path)
            p2p_error = error_calculator.calc_error(img_path, pred_vshaped)

            p2p_error_list.append(p2p_error)

            if it in vis_iter_nums:
                try_visualize(inps, output, labels, bboxes=bboxes, epoch_num=epoch_num, it=it, dataset_name=dataset_name, faces=m.module.smpl_layer.faces)

    with open(os.path.join(opt.work_dir, f'test_hbw_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(p2p_error_list, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = []
        for r in range(opt.world_size):
            with open(os.path.join(opt.work_dir, f'test_hbw_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join(opt.work_dir, f'test_hbw_kpt_rank_{r}.pkl'))

            kpt_all_pred += kpt_pred

        m_p2p_error = np.array(kpt_all_pred).mean()
        print(f'p2p_error = {m_p2p_error}, {np.array(p2p_error_list).mean()}')

        return m_p2p_error



def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
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

    criterion = builder.build_loss(cfg.LOSS).cuda(opt.gpu)
    optim_params = [{
        "params": m.parameters(),
        "lr": cfg.TRAIN.LR
    }]
    if 'reg' in opt.cfg or 'rle' in opt.cfg:
        print('use criterion params!!!')
        optim_params.append({
            "params": criterion.parameters(),
            "lr": cfg.TRAIN.LR
        })
        
    optimizer = torch.optim.Adam(optim_params, lr=cfg.TRAIN.LR)
    optimizer2 = None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=
        [item-cfg.TRAIN.BEGIN_EPOCH for item in cfg.TRAIN.LR_STEP], 
        gamma=cfg.TRAIN.LR_FACTOR)

    if opt.log:
        writer = SummaryWriter('.tensorboard/{}/{}-{}'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
    else:
        writer = None
    
    if cfg.DATASET.DATASET == 'mix_smpl_shapy':
        train_dataset = MixDatasetShapy(
            cfg=cfg,
            train=True)
    else:
        raise NotImplementedError

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    train_loader = torch.utils.data.DataLoader(
        # train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=opt.nThreads, 
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), num_workers=opt.nThreads, sampler=train_sampler, 
        worker_init_fn=_init_fn, drop_last=True, pin_memory=False)

    print('nthreads', opt.nThreads)
    # gt val dataset
    gt_val_dataset_h36m = MixDatasetShapy(
        cfg=cfg,
        train=False)

    gt_val_dataset_3dpw = PW3D(
        cfg=cfg,
        ann_file='3DPW_test_new.json',
        train=False)
    
    gt_val_dataset_agora = AGORA(
        cfg=cfg,
        ann_file='data/AGORA/annotations/validation_all_SMPL_withjv_withkid_valid.pt',
        train=False)

    gt_val_dataset_ssp3d = SSP3DDataset(
        cfg=cfg, ssp3d_dir_path='data/ssp_3d/ssp_3d')
    
    gt_val_dataset_hbw = HBWDataset(
        cfg=cfg,
        pkl_file='exp/HBW_results/bbox.pkl',
        train=False)

    opt.trainIters = 0
    best_err_h36m = 999
    best_err_3dpw = 999

    pytorch3d_renderer = TexturedIUVRenderer(device=torch.device('cuda', opt.gpu),
                                             batch_size=cfg.TRAIN.BATCH_SIZE,
                                             img_wh=256,
                                             projection_type='perspective',
                                             perspective_focal_length=1000.0,
                                             render_rgb=True,
                                             bin_size=32)

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        train_sampler.set_epoch(i)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, acc17 = train(opt, train_loader, m, criterion, optimizer, writer, i, pytorch3d_renderer=pytorch3d_renderer, optimizer2=optimizer2)
        logger.epochInfo('Train', opt.epoch, loss, acc17)

        lr_scheduler.step()

        if (i + 1) % (opt.snapshot//2) == 0:
            if opt.log:
                # Save checkpoint
                torch.save(m.module.state_dict(), './exp/{}/{}-{}/model_{}.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id, opt.epoch))
        
        if (i + 1) % (opt.snapshot//2) == 0:
            # Prediction Test
            with torch.no_grad():
                gt_tot_err_h36m = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord, epoch_num=i, dataset_name='h36m')
                gt_tot_err_3dpw = validate_gt(m, opt, cfg, gt_val_dataset_3dpw, heatmap_to_coord, epoch_num=i, dataset_name='3dpw')
                gt_tot_err_agora = validate_gt(m, opt, cfg, gt_val_dataset_agora, heatmap_to_coord, epoch_num=i, dataset_name='agora')

                m_pve_error = valid_ssp(m, opt, cfg, gt_val_dataset_ssp3d, heatmap_to_coord, epoch_num=i, dataset_name='ssp3d')
                p2p_error = valid_hbw(m, opt, cfg, gt_val_dataset_hbw, heatmap_to_coord, batch_size=1, epoch_num=i, dataset_name='hbw')

                if opt.log:
                    if gt_tot_err_h36m <= best_err_h36m:
                        best_err_h36m = gt_tot_err_h36m
                        torch.save(m.module.state_dict(), './exp/{}/{}-{}/best_h36m_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
                    
                    if gt_tot_err_3dpw <= best_err_3dpw:
                        best_err_3dpw = gt_tot_err_3dpw
                        torch.save(m.module.state_dict(), './exp/{}/{}-{}/best_3dpw_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))

                    logger.info(f'##### Epoch {opt.epoch} | h36m err: {gt_tot_err_h36m} / {best_err_h36m} | 3dpw err: {gt_tot_err_3dpw} / {best_err_3dpw} #####')
                    logger.info(f'ssp3d m_pve_error: {m_pve_error} | agora error: {gt_tot_err_agora} | hbw p2p error {p2p_error}#####')

        torch.distributed.barrier()


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL)

    if len(cfg.MODEL.PRETRAINED) > 0:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        save_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if type(save_dict) == dict:
            model_dict = save_dict['model']
            model.load_state_dict(model_dict, strict=True)
        else:
            model.load_state_dict(save_dict, strict=True)

    elif len(cfg.MODEL.TRY_LOAD) > 0:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD, map_location='cpu')
        if type(pretrained_state) == dict:
            pretrained_state = pretrained_state['model']

        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        for k, v in model.state_dict().items():
            if not k in pretrained_state:
                print(k)

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    elif len(cfg.MODEL.RESUME) > 0:
        logger.info(f'Resume model from {cfg.MODEL.RESUME}...')
        pretrained_state = torch.load(cfg.MODEL.RESUME, map_location='cpu')
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

    else:
        logger.info('Create new model')
        logger.info('=> Not init weights')
        # model._initialize()

    return model



def try_visualize(inps, preds, labels, faces, bboxes, epoch_num, it=0, dataset_name=''):
    batch_size = inps.shape[0]
    pa_path = 'exp/visualize/shape_vis_ddp'
    faces = torch.from_numpy(faces.astype(np.int32))

    pred_vertices = preds.pred_vertices.detach()

    transl = preds.transl.detach()

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

        cv2.imwrite(saved_path[0], image_vis)



if __name__ == "__main__":
    main()
