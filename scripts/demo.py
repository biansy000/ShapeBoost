"""Validation script."""
import argparse
import os
import pickle as pk

import torch
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapeboost.datasets.HBW_dataset import HBWDataset
from shapeboost.models import builder
from shapeboost.utils.config import update_config
from shapeboost.utils.env import init_dist
from shapeboost.utils.transforms import flip, get_func_heatmap_to_coord, get_one_box
import joblib
from shapeboost.utils.render_pytorch3d import render_mesh
import cv2
from shapeboost.utils.shapy_utils import BodyMeasurements
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import cv2
from torchvision import transforms as T


parser = argparse.ArgumentParser(description='PyTorch Pose Estimation Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--flip-shift',
                    default=False,
                    dest='flip_shift',
                    help='flip shift',
                    action='store_true')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed testing')
parser.add_argument('--dist-url', default='tcp://192.168.1.219:23456', type=str,
                    help='url used to set up distributed testing')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='job launcher')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed testing')
parser.add_argument('--folder_name', default='examples', type=str,
                    help='input image directory')
parser.add_argument('--work_dir', default='exp/examples_demo', type=str,
                    help='saved directory')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]

norm_method = cfg.LOSS.get('norm', 'softmax')

import os
def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    return(paths)

def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def run_detection(img_dir):
    det_transform = T.Compose([T.ToTensor()])

    det_model = fasterrcnn_resnet50_fpn(pretrained=True)
    det_model.cuda()
    det_model.eval()

    files = list_files(img_dir, 'png')
    bboxes = {}

    for file in tqdm(files):
            
        # process file name
        img_path = file

        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).cuda()
        det_output = det_model([det_input])
        # print(det_output)
        det_output = det_output[0]

        tight_bbox = get_one_box(det_output)

        bboxes[file] = tight_bbox
    
    return bboxes


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=4):
    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    v_shaped_list = []
    image_name_list = []
    beta_list = []

    meas_definition_path = 'extra_files/shapy_measurements/measurement_defitions.yaml'
    meas_vertices_path_smpl = 'extra_files/shapy_measurements/smpl_measurement_vertices.yaml'
    bm_smpl = BodyMeasurements(
            {'meas_definition_path': meas_definition_path,
                'meas_vertices_path': meas_vertices_path_smpl},
        ).cuda()
    
    meas_definition_path = 'extra_files/shapy_measurements/measurement_defitions.yaml'
    meas_vertices_path = 'extra_files/shapy_measurements/smplx_measurements.yaml'

    bm_smplx = BodyMeasurements(
        {'meas_definition_path': meas_definition_path,
            'meas_vertices_path': meas_vertices_path},
    ).cuda()

    if opt.log:
        gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for i, (inps, labels, img_ids, img_paths, bboxes) in enumerate(gt_val_loader):
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].cuda(opt.gpu)
            except AttributeError:
                pass

        bboxes = bboxes.cuda(opt.gpu)
        with torch.no_grad():
            output = m(inps, bboxes=bboxes,
                       img_center=labels['img_center'])

            pred_beta = output.pred_shape
            useflip = False
            if useflip:
                inps_flipped = flip(inps)
                bbox_flipped = labels['bbox_flipped']
                output_flipped = m(inps_flipped, bboxes=bbox_flipped, img_center=labels['img_center'])

                pred_beta = (output.pred_shape + output_flipped.pred_shape) / 2
            
            if 'smplx_out_rest' in output:
                test_betas = output.smplx_out_rest.pred_betas
                v_shaped = m.smplx_layer.get_rest_pose(test_betas)['vertices']

                height_fits = bm_smplx.compute_height(v_shaped[:, m.smplx_layer.faces_tensor])[0]
                height_pred = bm_smpl.compute_height(v_shaped[:, m.smpl_layer.faces_tensor])[0]
                
            else:
                v_shaped = m.smpl_layer.get_rest_pose(pred_beta)['vertices']
                height_fits = bm_smpl.compute_height(v_shaped[:, m.smpl_layer.faces_tensor])[0]
                height_pred = bm_smpl.compute_height(v_shaped[:, m.smpl_layer.faces_tensor])[0]

        v_shaped = v_shaped.cpu().numpy()
        pred_beta = pred_beta.cpu().numpy()

        vis = True
        if vis:
            try_visualize(inps, output, labels, bboxes=bboxes, epoch_num=0, it=i, dataset_name='HBW', faces=m.smpl_layer.faces, 
                          img_paths=img_paths, height_preds=height_pred, height_fits=height_fits)
        
        for k in range(pred_beta.shape[0]):
            img_path = img_paths[k].split('/')
            img_path = '/'.join(img_path[3:])
            image_name_list.append(img_path)
            v_shaped_list.append(v_shaped[k])

            beta_list.append(pred_beta[k])

    image_name_list = np.array(image_name_list)
    v_shaped_list = np.stack(v_shaped_list, axis=0)

    return image_name_list, v_shaped_list, beta_list


def main():
    main_worker(None, opt, cfg)


def main_worker(gpu, opt, cfg):

    opt.rank = 0
    opt.world_size = 1
    opt.ngpus_per_node = 1
    opt.gpu = 0
    
    hbw_root =  opt.folder_name
    opt.work_dir = opt.work_dir

    if not os.path.exists(opt.work_dir):
        os.makedirs(opt.work_dir)
    
    pkl_file = f'{opt.work_dir}/bbox.pkl'

    if opt.rank == 0:
        opt.log = True
    else:
        opt.log = False

    torch.backends.cudnn.benchmark = True

    m = builder.build_sppe(cfg.MODEL)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint, map_location='cpu'), strict=False)

    m.cuda()

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    if not os.path.exists(pkl_file):
        detected_bboxes = run_detection(hbw_root)
        with open(os.path.join(opt.work_dir, 'bbox.pkl'), 'wb') as handle:
            detected_bboxes = pk.dump(detected_bboxes, handle)
    
    gt_val_dataset_h36m = HBWDataset(
        cfg=cfg,
        pkl_file=pkl_file,
        train=False, )

    with torch.no_grad():
        image_name_list, v_shaped_list, beta_list = validate_gt(m, opt, cfg, gt_val_dataset_h36m, heatmap_to_coord)
    
    np.savez(
        os.path.join(opt.work_dir, 'hbw_prediction'), 
        image_name=image_name_list, 
        v_shaped=v_shaped_list
    )

    np.savez(
        os.path.join(opt.work_dir, 'hbw_prediction_beta'), 
        image_name=image_name_list, 
        beta=beta_list
    )

    

def try_visualize(inps, preds, labels, faces, bboxes, epoch_num, it=0, dataset_name='', img_paths=None, height_preds=None, height_fits=None):
    batch_size = inps.shape[0]
    pa_path = f'{opt.work_dir}/shape_{dataset_name}_test_new'
    print(pa_path)
    if not os.path.exists(pa_path):
        os.makedirs(pa_path)

    faces = torch.from_numpy(faces.astype(np.int32))

    pred_vertices = preds.pred_vertices.detach()
    transl = preds.transl.detach()
    named_clothes = False

    f = 1000.0
    for bid in range(batch_size):
        if img_paths is None:
            saved_path = [f'{pa_path}/{epoch_num}_{dataset_name}_{it}_{bid:03d}_mesh.jpg']
            saved_path_uv = [f'{pa_path}/{epoch_num}_{dataset_name}_{it}_{bid:03d}_uv.jpg']
        elif not named_clothes:
            img_path = img_paths[bid].split('/')
            saved_name = f's_{img_path[-2]}_{img_path[-1]}'
                
            saved_path = [f'{pa_path}/s_{saved_name}_mesh_{float(height_preds[bid]):.2f}_{height_fits[bid]:.2f}.jpg']
            saved_path_uv = [f'{pa_path}/s_{saved_name}_uv_{float(height_preds[bid]):.2f}_{height_fits[bid]:.2f}.jpg']

            print(saved_name)
        else:
            names = ['thin', 'normal', 'thick']
            pred_clothes_label = preds.pred_clothes_label[bid].reshape(2, 3)
            upper_argmax = torch.argmax(pred_clothes_label[0])
            lower_argmax = torch.argmax(pred_clothes_label[1])

            upper_name = names[upper_argmax]
            lower_name = names[lower_argmax]

            img_path = img_paths[bid].split('/')
            saved_name = f's_{img_path[-2]}_{img_path[-1]}'
                
            saved_path = [f'{pa_path}/s_{saved_name}_mesh_{float(height_preds[bid]):.2f}_{height_fits[bid]:.2f}.jpg']
            saved_path_uv = [f'{pa_path}/s_{saved_name}_uv_{float(height_preds[bid]):.2f}_{height_fits[bid]:.2f}.jpg']

            print(saved_name)

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


