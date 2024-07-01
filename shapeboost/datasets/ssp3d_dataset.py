import scipy.misc
import copy

import cv2
import numpy as np
import os
import torch
import joblib

from torch.utils.data import Dataset

from shapeboost.beta_decompose.shape_utils import convert_bbox_centre_hw_to_corners

from shapeboost.models.layers.smpl.SMPL import SMPL_layer
from shapeboost.utils.presets.trivial_transform import TrivialTransform3D
from shapeboost.utils.transforms import get_affine_transform
import cv2


class SSP3DDataset(Dataset):
    def __init__(self, cfg, ssp3d_dir_path, silh_from='pointrend'):
        super(SSP3DDataset, self).__init__()

        # Paths
        self.images_dir = os.path.join(ssp3d_dir_path, 'images')
        self.silhouettes_dir = os.path.join(ssp3d_dir_path, 'silhouettes')
        self.pred_silhouettes_dir = os.path.join(ssp3d_dir_path, 'silhouettes')

        # Data
        data = np.load(os.path.join(ssp3d_dir_path, 'labels.npz'))

        self.image_fnames = data['fnames']
        self.body_shapes = data['shapes']
        self.body_poses = data['poses']
        self.cam_trans = data['cam_trans']
        self.joints2D = data['joints2D']
        self.bbox_centres = data['bbox_centres']  # Tight bounding box centre
        self.bbox_whs = data['bbox_whs']  # Tight bounding box width/height
        self.genders = data['genders']

        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE
        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM
        self.bbox_3d_shape = (2.2, 2.2, 2.2)
        self._loss_type = cfg.LOSS['TYPE']
        self.focal_length = cfg.DATASET.get('FOCAL_LENGTH', 5000)
        self._train = False

        self._items, self._labels = self._lazy_load_json()

        self.transformation = TrivialTransform3D( 
            self, scale_factor=0.3,
            color_factor=self._color_factor,
            occlusion=False,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=False,
            loss_type=self._loss_type,
            focal_length=self.focal_length,
            scale_mult=1.25)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        img_id = int(self._labels[idx]['img_id'])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        target['vid_name'] = 'ssp_3d'
        target['img_name'] = self._labels[idx]['img_name']
        target['frame_id'] = idx
        target['target_beta'] = torch.from_numpy(self._labels[idx]['shape']).float()
        target['gender'] = self._labels[idx]['gender']
        # print(target['target_beta'].shape)

        return img, target, img_id, bbox
    
    def _lazy_load_json(self):
        items = []
        labels = []
        for index in range(len(self.image_fnames)):

            fname = self.image_fnames[index]
            img_path = os.path.join(self.images_dir, fname)
            # image = cv2.imread(os.path.join(self.images_dir, fname))[:,:,::-1]
            # silhouette = cv2.imread(os.path.join(self.silhouettes_dir, fname), 0)
            joints2D = self.joints2D[index]
            shape = self.body_shapes[index]
            pose = self.body_poses[index]
            cam_trans = self.cam_trans[index]
            gender = self.genders[index]

            # Crop images to bounding box if needed
            bbox_centre = self.bbox_centres[index]
            bbox_wh = self.bbox_whs[index]
            bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, bbox_wh, bbox_wh)
            bbox_corners[bbox_corners < 0] = 0
            # top_left = bbox_corners[:2].astype(np.int16)
            # bottom_right = bbox_corners[2:].astype(np.int16)
            # top_left[top_left < 0] = 0

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

            items.append(img_path)
            labels.append({
                'bbox': bbox_corners, # xmin, ymin, xmax, ymax
                'img_id': index,
                'img_path': img_path,
                'img_name': fname,

                'joint_img_17': joint_img_17,
                'joint_vis_17': joint_vis_17,
                'joint_cam_17': joint_cam_17,
                'joint_relative_17': joint_relative_17,
                'joint_img_29': joint_img_29,
                'joint_vis_29': joint_vis_29,
                'joint_cam_29': joint_cam_29,
                'root_cam': root_cam,
                'f': focal,
                'c': center,

                'shape': shape,
                'pose': pose,
                'cam_trans': cam_trans,
                'joints2D': joints2D,
                'gender': gender
            })
    
        return items, labels

