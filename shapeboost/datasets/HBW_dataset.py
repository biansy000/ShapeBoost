"""3DPW dataset."""
import copy
import json
import os
import joblib
import math
import pickle as pk
import glob

import numpy as np
import torch
import cv2
import torch.utils.data as data
from shapeboost.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from shapeboost.utils.pose_utils import pixel2cam, reconstruction_error
from shapeboost.utils.presets import TrivialTransform3D


class HBWDataset(data.Dataset):
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
                 img_root='data/HBW/images/val',
                 pkl_file='',
                 train=True,
                 use_detected_bbox=False
                 ):
        self._cfg = cfg

        self._img_root = img_root
        self._pkl_file = pkl_file
        self._train = train
        self.use_detected_bbox = use_detected_bbox

        self._scale_factor = 0.3
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

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

        self.img_cnt = 0

        self._items, self._labels = self._lazy_load_json()

        self.focal_length = cfg.DATASET.get('FOCAL_LENGTH', 1000)

        print('data length', len(self._items))

        self.transformation = TrivialTransform3D( 
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=self._occlusion,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=False,
            loss_type=self._loss_type,
            focal_length=self.focal_length,
            scale_mult=1.2)

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

        # target['vid_name'] = self.vid_name
        # target['img_name'] = self._labels[idx]['img_name']
        # target['frame_id'] = idx

        return img, target, img_id, img_path, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):

        if os.path.exists(self._pkl_file) and self._pkl_file.split('.')[-1] == 'pkl':
            with open(self._pkl_file, 'rb') as f:
                pkl_labels = pk.load(f)
        else:
            pkl_labels = None
            raise NotImplementedError

        items = []
        labels = []
        len_data = len(pkl_labels)

        bboxes_list = []
        i = 0
        for img_path, bbox_det in pkl_labels.items():
            
            if bbox_det is None:
                bbox = (-1, -1, -1, -1)
            else:
                bbox = bbox_det

            joint_img_17 = np.zeros((17, 3))
            joint_vis_17 = np.zeros((17, 3))
            joint_cam_17 = np.zeros((17, 3))
            joint_relative_17 = np.zeros((17, 3))
            joint_img_29 = np.zeros((29, 3))
            joint_vis_29 = np.zeros((29, 3))
            joint_cam_29 = np.zeros((29, 3))
            root_cam = np.zeros(3)

            focal = np.array([1000.0, 1000.0])
            center = np.array([0.0, 0.0])

            img_name = img_path.split('/')[-1]

            items.append(img_path)
            labels.append({
                'bbox': bbox, # xmin, ymin, xmax, ymax
                'img_id': i,
                'img_path': img_path,
                'img_name': img_name,

                'joint_img_17': joint_img_17,
                'joint_vis_17': joint_vis_17,
                'joint_cam_17': joint_cam_17,
                'joint_relative_17': joint_relative_17,
                'joint_img_29': joint_img_29,
                'joint_vis_29': joint_vis_29,
                'joint_cam_29': joint_cam_29,
                'root_cam': root_cam,
                'f': focal,
                'c': center
            })

            bboxes_list.append(bbox)
            i += 1
        
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

    def keyps_to_bbox(self, keypoints):
        body_keypoints = np.array(keypoints['pose_keypoints_2d']).reshape(25, 3)
        kpt_xmin, kpt_ymin, kpt_xmax, kpt_ymax = 4096, 4096, -1, -1
        cnt_num = 0
        for j, single_kpt in enumerate(body_keypoints):
            if single_kpt[2] < 0.1:
                continue

            kpt_xmin, kpt_ymin = min(kpt_xmin, single_kpt[0]), min(kpt_ymin, single_kpt[1])
            kpt_xmax, kpt_ymax = max(kpt_xmax, single_kpt[0]), max(kpt_ymax, single_kpt[1])
            cnt_num += 1
        
        for j, single_kpt in enumerate(body_keypoints):
            if single_kpt[2] < 0.1:
                continue

            kpt_xmin, kpt_ymin = min(kpt_xmin, single_kpt[0]), min(kpt_ymin, single_kpt[1])
            kpt_xmax, kpt_ymax = max(kpt_xmax, single_kpt[0]), max(kpt_ymax, single_kpt[1])
            cnt_num += 1
        
        hand_keypoints = np.array(keypoints['hand_right_keypoints_2d'] + keypoints['hand_left_keypoints_2d']).reshape(-1, 3)
        for j, single_kpt in enumerate(hand_keypoints):
            if single_kpt[2] < 0.2:
                continue

            kpt_xmin, kpt_ymin = min(kpt_xmin, single_kpt[0]), min(kpt_ymin, single_kpt[1])
            kpt_xmax, kpt_ymax = max(kpt_xmax, single_kpt[0]), max(kpt_ymax, single_kpt[1])

        if cnt_num < 6:
            return -1, -1, -1, -1
        
        else:
            return kpt_xmin, kpt_ymin, kpt_xmax, kpt_ymax