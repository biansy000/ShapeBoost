"""3DPW dataset."""
import copy
import json
import os
from PIL import Image

import numpy as np
import torch
import scipy.misc
import torch.utils.data as data
from pycocotools.coco import COCO
from shapeboost.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy, xyxy_to_center_scale
from shapeboost.utils.pose_utils import pixel2cam, reconstruction_error
from shapeboost.utils.presets import SimpleTransform3DSMPLCam
from shapeboost.utils.transforms import batch_rodrigues, batch_rodrigues_numpy
import random
import joblib
import cv2

def getPersonMaskPath_low(maskBaseFolder, imgName, pnum,):
    scene_type='_'.join(imgName.split('_')[0:5])
    maskFolder = os.path.join(maskBaseFolder, scene_type)
    imgBase = '_'.join(imgName.split('_')[0:-2])
    ending = '_1280x720.png'

    i = int(imgName.replace('.png','').split('_')[-2])
    format_pnum = format(pnum+1, '05d')
    format_i = format(i, '06d')
    if 'archviz' in imgName:
        cam = imgBase.split('_')[-1]
        imgBase = '_'.join(imgName.split('_')[0:-3])
        maskPerson = os.path.join(maskFolder, imgBase+'_mask_'+cam+'_'+format_i+'_'+format_pnum+ending)
    else:
        maskPerson = os.path.join(maskFolder, imgBase+'_mask_'+format_i+'_'+format_pnum+ending)

    return maskPerson

class AGORA(data.Dataset):
    """ 3DPW dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/pw3d'
        Path to the PW3D dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']

    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 17
    bbox_3d_shape = (2.2, 2.2, 2.2)
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
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
        'left_thumb', 'right_thumb'             # 23
    )
    joints_name_14 = (
        'R_Ankle', 'R_Knee', 'R_Hip',           # 2
        'L_Hip', 'L_Knee', 'L_Ankle',           # 5
        'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
        'Neck', 'Head'
    )
    skeleton = (
        (1, 0), (2, 1), (3, 2),  # 2
        (4, 0), (5, 4), (6, 5),  # 5
        (7, 0), (8, 7),  # 7
        (9, 8), (10, 9),  # 9
        (11, 7), (12, 11), (13, 12),  # 12
        (14, 7), (15, 14), (16, 15),  # 15
    )

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/AGORA',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False,
                 return_img_path=False, 
                 change_widths=True):
        self._cfg = cfg

        self._ann_file = ann_file
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg
        self._check_centers = return_img_path

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION
        self.use_kid = cfg.DATASET.get('USE_KID', False)
        self.use_mask = (cfg.DATASET.get('USE_MASK', False) and self._train)
        print('self.use_kid', self.use_kid, 'self.use_mask', self.use_mask)

        # if self.use_mask:
        #     assert 'valid' in ann_file and 'train' in ann_file

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = -1

        self.augment = cfg.MODEL.EXTRA.AUGMENT

        self._loss_type = cfg.LOSS['TYPE']

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)
        self.classfier = cfg.MODEL.EXTRA.get('WITHCLASSFIER', False)

        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.lshoulder_idx_17 = self.joints_name_17.index('L_Shoulder')
        self.rshoulder_idx_17 = self.joints_name_17.index('R_Shoulder')
        self.root_idx_smpl = self.joints_name_24.index('pelvis')
        self.lshoulder_idx_24 = self.joints_name_24.index('left_shoulder')
        self.rshoulder_idx_24 = self.joints_name_24.index('right_shoulder')

        self.preset_amb = cfg.MODEL.EXTRA.get('AMBIGUOUS', False)

        self.db = joblib.load(ann_file, None)
        self._items, self._labels = self._lazy_load_pt(self.db)
        self.focal_length = 1000.0

        self._use_finer_part = cfg.MODEL.EXTRA.get('USE_FINER_PART', False)

        # self.transformation = NonsenseTransform(\
        self.transformation = SimpleTransform3DSMPLCam( 
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=False,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg,
            loss_type=self._loss_type,
            focal_length=self.focal_length,
            scale_mult=1.0, change_widths=change_widths,
            use_finer=self._use_finer_part)

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        if self._check_centers: 
            # temorarily reuse an unused var
            return img, target, img_id, img_path, bbox
        
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_pt(self, db):
        """Load all image paths and labels from json annotation files into buffer."""

        items = []
        labels = []

        local_random = random.Random(0)

        db_len = len(db['ann_path'])

        for k, v in db.items():
            assert len(v) == db_len, k
            # print(k)
        
        img_cnt = 0
        prev_img_base_name = ''
        mask_base_path = 'data/AGORA/masks/train_masks_1280x720/train'
        person_id = 0

        for idx in range(db_len):
            img_name = db['img_path'][idx]
            ann_path = db['ann_path'][idx]
            ann_file = ann_path.split('/')[-1]

            ann_file = ann_file.split('_')
            if 'train' in img_name:
                img_parent_path = os.path.join(self._root, 'images', f'{ann_file[0]}_{ann_file[1]}')
            else:
                img_parent_path = os.path.join(self._root, 'images', 'validation')

            img_path = os.path.join(img_parent_path, img_name)

            beta = np.array(db['shape'][idx]).reshape(10)
            theta = np.array(db['pose'][idx]).reshape(24, 3)

            joint_rel_17 = db['xyz_17'][idx].reshape(17, 3)
            joint_vis_17 = np.ones((17, 3))

            joint_rel_17 = joint_rel_17 - joint_rel_17[0, :]

            joint_cam_24 = db['gt_joints_3d'][idx].reshape(-1, 3)[:24]
            joint_rel_29 = db['xyz_29'][idx].reshape(29, 3)
            joint_rel_29 = joint_rel_29 - joint_rel_29[0]

            joint_2d_full = db['uv_24'][idx].reshape(-1, 2)
            
            if self._train:
                # print('yes')
                joint_2d_29 = db['uv_29'][idx].reshape(-1, 2)
                joint_2d = joint_2d_29[:24]
                joint_img_29 = np.zeros_like(joint_rel_29)
                joint_img_29[:, 2] = joint_rel_29[:, 2]
                joint_img_29[:, :2] = joint_2d_29

                joint_vis_24 = np.ones((24, 3))
                joint_vis_29 = np.ones((29, 3))
            else:
                joint_2d = joint_2d_full[:24]
                joint_img_29 = np.zeros_like(joint_rel_29)
                joint_img_29[:, 2] = joint_rel_29[:, 2]
                joint_img_29[:24, :2] = joint_2d

                joint_vis_24 = np.ones((24, 3))
                joint_vis_29 = np.zeros((29, 3))
                joint_vis_29[:24, :] = joint_vis_24

            root_cam = joint_cam_24[0]

            if 'twist_angle' in db:
                angle = db['twist_angle'][idx].reshape(-1)
                cos = np.cos(angle)
                sin = np.sin(angle)

                phi = np.stack((cos, sin), axis=1)
                phi_weight = (angle > -10) * 1.0
                phi_weight = np.stack([phi_weight, phi_weight], axis=1)
            else:
                phi = np.zeros((23, 2))
                phi_weight = np.zeros_like(phi)
            
            # generate bbox from kpt2d
            left, right, upper, lower = \
                joint_2d_full[:, 0].min(), joint_2d_full[:, 0].max(), joint_2d_full[:, 1].min(), joint_2d_full[:, 1].max()

            center = np.array([(left+right)*0.5, (upper+lower)*0.5], dtype=np.float32)
            scale = [right-left, lower-upper]

            scale = float(max(scale))
            
            # if self.preset_amb and self._train:
            #     rand_shift = 0.15 * scale * rand_norm
            #     rand_scale_shift = 0.2 * scale * rand_scale_norm
            # else:
            if not self._train:
                rand_norm = np.array([local_random.gauss(mu=0, sigma=1), local_random.gauss(mu=0, sigma=1)])
                rand_scale_norm = local_random.gauss(mu=0, sigma=1)

                rand_shift = 0.05 * scale * rand_norm
                rand_scale_shift = 0.1 * scale * rand_scale_norm
            
                center = center + rand_shift
                scale = scale + rand_scale_shift
            
            scale = scale * 1.3

            xmin, ymin, xmax, ymax = center[0] - scale*0.5, center[1] - scale*0.5, center[0] + scale*0.5, center[1] + scale*0.5

            if not(xmin < 1280-5 and ymin < 720-5 and xmax > 5 and ymax > 5):
                continue

            # get rough uv17 by orthogonal projection
            uv_xy_scale = joint_2d.std(axis=0) / joint_cam_24[:, :2].std(axis=0)
            uv_17 = uv_xy_scale * joint_rel_17[:, :2] + joint_2d[0]
            joint_img_17 = np.zeros_like(joint_rel_17)
            joint_img_17[:, :2] = uv_17
            joint_img_17[:, 2] = joint_rel_17[:, 2]

            is_valid = db['is_valid'][idx]

            if not is_valid:
                continue

            if self.use_kid:
                is_kid = db['is_kid'][idx]
                beta_kid = np.array(db['shape_kid'][idx])
            else:
                beta_kid = np.zeros(1)
                if 'is_kid' in db and db['is_kid'][idx]:
                    # delete kid annotations
                    continue
            
            mask_img_name = ''
            if self.use_mask:
                img_base_name = '_'.join(img_name.split('_')[0:-2])
                if img_base_name != prev_img_base_name:
                    person_id = 0
                
                mask_img_name = getPersonMaskPath_low(mask_base_path, img_name, person_id)
                person_id += 1
                # cam_param = db['cam_param'][idx]
                # print(cam_param)

            items.append(img_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': img_cnt,
                'img_path': img_path,
                'img_name': img_name,
                'joint_img_17': joint_img_17.copy(), # TODO: NOT ACCURATE AND INFLUENCE HALF-BODY TRANSFORM!
                'joint_vis_17': joint_vis_17.copy(),
                'joint_cam_17': joint_rel_17.copy(), # change to relative, no bug
                'joint_relative_17': joint_rel_17.copy(),
                'joint_img_29': joint_img_29.copy(),
                'joint_vis_29': joint_vis_29.copy(),
                'joint_cam_29': joint_rel_29.copy(), # change to relative, no bug
                'twist_phi': phi,
                'twist_weight': phi_weight,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': [-1, -1], # TODO: WRONG, BUT NOT ACTUALLY USED
                'c': [-1, -1],
                'beta_kid': beta_kid,
                'mask_img_name': mask_img_name
            })

            if self._check_centers:
                labels[img_cnt]['is_valid'] = is_valid
            
            img_cnt += 1

        print('datalen', db_len, len(items))
        return items, labels

    @property
    def joint_pairs_17(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    @property
    def joint_pairs_24(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    @property
    def joint_pairs_29(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 1), (2, 3), (4, 5), (7, 8), (9, 10), (11, 12))

    def evaluate_uvd_24(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            f = gt['f']
            c = gt['c']
            bbox_xyxy = gt['bbox'] # xyxy
            bbox = xyxy_to_center_scale(bbox_xyxy)
            gt_2d_kpt = gt['joint_img_29'][:24, :].copy()

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd_jts'][:24, :].copy()
            gt_2d_kpt[:, 0] = (gt_2d_kpt[:, 0] - bbox[0]) / bbox[2]
            gt_2d_kpt[:, 1] = (gt_2d_kpt[:, 1] - bbox[1]) / bbox[3]
            
            # error calculate
            error[n] = np.sqrt(np.sum((pred_2d_kpt - gt_2d_kpt)**2, 1))
            error_x[n] = np.abs(pred_2d_kpt[:, 0] - gt_2d_kpt[:, 0])
            error_y[n] = np.abs(pred_2d_kpt[:, 1] - gt_2d_kpt[:, 1])
            error_z[n] = np.abs(pred_2d_kpt[:, 2] - gt_2d_kpt[:, 2])
            img_name = gt['img_path']

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_kp = np.mean(error, axis=0) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000
        metric = 'MPJPE'

        eval_summary = f'UVD_24 error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)
        # print(f'UVD_24 error per joint: {tot_err_kp}')

        return tot_err

    def evaluate_xyz_24(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_align = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam'].copy()
            gt_3d_kpt = gt['joint_cam_29'][:24, :].copy()

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_24'].copy() * 2.2

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # rigid alignment for PA MPJPE
            pred_3d_kpt_align = reconstruction_error(
                pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['img_path']

            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_align = np.mean(error_align) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000

        eval_summary = f'XYZ_24 >> tot: {tot_err:2f}, tot_pa: {tot_err_align:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_17(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)

        pred_save = []
        error = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_pa = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_x = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_y = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_z = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        # error for each sequence
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_relative_17']

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_17'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]

            # select eval 14 joints
            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = gt['img_path']

            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox, 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_pa = np.mean(error_pa) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000

        eval_summary = f'PA MPJPE >> tot: {tot_err_pa:2f}; MPJPE >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err_pa

