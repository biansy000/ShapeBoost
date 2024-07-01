"""3DPW dataset."""
import copy
import json
import os
import joblib
import math
import pickle as pk
import glob
import bisect
import random

import numpy as np
import torch
import cv2
import torch.utils.data as data
from shapeboost.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from shapeboost.utils.pose_utils import pixel2cam, reconstruction_error
from shapeboost.utils.presets.simple_transform_cam_ma import SimpleTransformCamMA

class ModelAgencyDataset(data.Dataset):
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
                 img_root='data/ModelAgencyData/images',
                 ann_file='data/ModelAgencyData/keypoints_new.json',
                 train=True,
                 filtered=False,
                 use_openpose=False,
                 ):
        self._cfg = cfg
        self.filtered = filtered

        self._img_root = img_root
        self._ann_file = ann_file
        self._train = train

        self._scale_factor = 0.3
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM
        self.use_mask = cfg.DATASET.get('USE_MASK', False)

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

        partition = np.array(self.ratio_list)

        self.partition = partition / partition.sum()
        self.cumulative_sizes = self.cumsum(self.partition)

        # remap gender and image number
        def gender_img_remap(len_image, gender):
            if len_image < 10:
                added_ratio = (10 / len_image)
                ratio = added_ratio
            elif len_image > 30:
                ratio = (30 / len_image)
            else:
                ratio = 1
            
            if gender == 'male':
                ratio = ratio * 2
            
            return ratio

        partition2 = [gender_img_remap(item0, item1) for item0, item1 in zip(self.len_images_list, self.gender_list)]
        partition2 = np.array(partition2)

        self.partition2 = partition2 / partition2.sum()
        self.cumulative_sizes2 = self.cumsum(self.partition2)
        self._use_finer_part = cfg.MODEL.EXTRA.get('USE_FINER_PART', False)

        self.transformation = SimpleTransformCamMA( 
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
            scale_mult=1.2, use_mask=self.use_mask,
            use_finer=self._use_finer_part)

    def __getitem__(self, idx):
        p = random.random()
        idx = bisect.bisect_right(self.cumulative_sizes, p)

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

        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):

        if self.filtered:
            from shapeboost.datasets.model_indices import male_indices, female_indices

        with open(self._ann_file, 'r') as f:
            kpt_gt_annot = json.load(f)
        
        with open('data/ModelAgencyData/refined_betas/attributes_summary.json', 'r') as f:
            attri_annot = json.load(f)
        
        with open('data/ModelAgencyData/wrong_images.json', 'r') as f:
            wrong_images = json.load(f)
        
        with open('data/ModelAgencyData/mask2former_bbox.pkl', 'rb') as f:
            image_bboxes = pk.load(f)
        
        items = []
        labels = []
        
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

        bmi_list = []
        kpt_group = {i: i for i in range(15)}
        for i in [15, 16, 17, 18]:
            kpt_group[i] = 0
        
        for i in [22, 23, 24]:
            kpt_group[i] = 11
        
        for i in [19, 20, 21]:
            kpt_group[i] = 14

        imgid = 0
        male_num = 0
        female_num = 0
        male_num_img, female_num_img = 0, 0
        len_images_list = []
        len_images_list_male = []
        len_images_list_female = []
        gender_list = []
        height_list = []
        height_list_male = []
        height_list_female = []
        height_list_male_human = []
        height_list_female_human = []
        weight_list = []
        image_belong_to_male = []
        image_belong_to_female = []
        male_index = -1
        female_index = -1
        for web_name, web_val in kpt_gt_annot.items():

            if web_name not in attri_annot:
                continue

            for model_name, model_v in web_val.items():

                if model_name not in attri_annot[web_name]:
                    continue

                attribute = attri_annot[web_name][model_name]
                images = model_v['images']

                gender = attribute['gender']
                if gender == 'male':
                    male_num += 1
                    male_num_img += len(images)
                    height_list_male_human.append(attribute['height'])
                    len_images_list_male.append(len(images))
                    male_index += 1

                    if self.filtered and male_index not in male_indices:
                        continue

                elif gender == 'female':
                    female_num += 1
                    female_num_img += len(images)
                    height_list_female_human.append(attribute['height'])
                    len_images_list_female.append(len(images))
                    female_index += 1
                    # if isinstance(attribute['attributes'], list):
                    #     print(attribute['height'], attribute['attributes'][-2], attribute['attributes'][-1])

                    if self.filtered and female_index not in female_indices:
                        continue
                else:
                    print(gender)

                for i, image_name in enumerate(images):
                    img_path = os.path.join(self._img_root, web_name, model_name, image_name)
                    
                    if img_path in wrong_images:
                        continue

                    body_keypoints = model_v['body_keypoints'][i]

                    kpt_exist = np.zeros(15)
                    kpt_xmin, kpt_ymin, kpt_xmax, kpt_ymax = 4096, 4096, -1, -1
                    cnt_num = 0
                    for j, single_kpt in enumerate(body_keypoints):
                        if single_kpt[2] < 0.1:
                            continue

                        kpt_xmin, kpt_ymin = min(kpt_xmin, single_kpt[0]), min(kpt_ymin, single_kpt[1])
                        kpt_xmax, kpt_ymax = max(kpt_xmax, single_kpt[0]), max(kpt_ymax, single_kpt[1])
                        cnt_num += 1

                        kpt_idx = kpt_group[j]
                        kpt_exist[kpt_idx] += 1

                    center_raw = np.array([(kpt_xmin+kpt_xmax)*0.5, (kpt_ymin+kpt_ymax)*0.5], dtype=np.float32)
                    scale_raw = [kpt_xmax-kpt_xmin, kpt_ymax-kpt_ymin]
                    scale_raw = float(max(scale_raw))

                    xmin0, ymin0, xmax0, ymax0 = center_raw[0] - scale_raw*0.5, center_raw[1] - scale_raw*0.5, \
                        center_raw[0] + scale_raw*0.5, center_raw[1] + scale_raw*0.5

                    xmin1, ymin1, xmax1, ymax1 = image_bboxes[img_path][:-1]

                    xmin, ymin, xmax, ymax = min(xmin0, xmin1), min(ymin0, ymin1), max(xmax0, xmax1), max(ymax0, ymax1)
                    
                    if xmin > xmax -20 or ymin > ymax -20:
                        xmin, ymin, xmax, ymax = -1, -1, -1, -1

                    img_name = img_path.split('/')[-1]
                    mask_path = img_path.replace('/images/', '/masks2/')

                    items.append(img_path)
                    labels.append({
                        'bbox': [xmin, ymin, xmax, ymax], # xmin, ymin, xmax, ymax
                        'img_id': imgid,
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
                        'c': center,
                        'attribute': attribute,
                        'mask_path': mask_path,
                        'body_keypoints': body_keypoints
                    })

                    if attribute['mass_w'] > 0.5 and attribute['height_w'] > 0.5:
                        bmi = attribute['mass'] / (attribute['height']**2)
                        weight_list.append(attribute['mass'])
                    else:
                        bmi = 20
                        weight_list.append(-1)
                    
                    assert attribute['height_w'] > 0.5
                    height_list.append(attribute['height'])
                    if gender == 'male': 
                        height_list_male.append(attribute['height'])
                        image_belong_to_male.append(male_index)
                    else: 
                        height_list_female.append(attribute['height'])
                        image_belong_to_female.append(female_index)

                    bmi_list.append(bmi)
                    len_images_list.append(len(images))
                    gender_list.append(gender)
                    imgid += 1
                    

        # import matplotlib.pyplot as plt  
        height_bins = np.array([1.5 + 0.05 * i for i in range(10)])
        
        # height_indices_male_human = np.digitize(height_list_male_human, height_bins)
        # height_indices_female_human = np.digitize(height_list_female_human, height_bins)

        # choosed_male_idx = []
        # choosed_female_idx = []
        # img_num=0
        # for i in range(11):
        #     male_human_in_height = (height_indices_male_human == i)
        #     female_human_in_height = (height_indices_female_human == i)

        #     print(male_human_in_height.sum(), female_human_in_height.sum())

        #     inheight_male_imgnum = np.array(len_images_list_male).copy()
        #     inheight_female_imgnum = np.array(len_images_list_female).copy()

        #     inheight_male_imgnum[~male_human_in_height] = 0
        #     inheight_female_imgnum[~female_human_in_height] = 0

        #     imgnum_index_male = np.argsort(-inheight_male_imgnum)
        #     imgnum_index_female = np.argsort(-inheight_female_imgnum)

        #     for k in range(300):

        #         if k > 100 and i > 4:
        #             continue

        #         choosed = imgnum_index_female[k]
        #         if inheight_female_imgnum[choosed] > 0:
        #             choosed_female_idx.append(choosed)
        #             img_num += inheight_female_imgnum[choosed]

        #         if k > 80:
        #             continue

        #         choosed = imgnum_index_male[k]
        #         if inheight_male_imgnum[choosed] > 0:
        #             choosed_male_idx.append(choosed)
        #             img_num += inheight_male_imgnum[choosed]
            
        #     # print(inheight_male_imgnum, inheight_female_imgnum)
        
        # print(choosed_male_idx, choosed_female_idx, img_num)

        self.gender_list = gender_list
        self.len_images_list = len_images_list

        height_indices_male = np.digitize(height_list_male, height_bins)
        height_indices_female = np.digitize(height_list_female, height_bins)

        ratio_male = [(height_indices_male == i).sum() for i in range(11)]
        ratio_female = [(height_indices_female == i).sum() for i in range(11)]

        print('ratio_male, ratio_female', ratio_male, ratio_female)
        ratio_male = np.clip(ratio_male, a_min=200, a_max=5000)
        ratio_female = np.clip(ratio_female, a_min=200, a_max=5000)

        male_cnt = 0
        female_cnt = 0
        ratio_male_cnt = [0 for i in range(11)]
        ratio_female_cnt = [0 for i in range(11)]
        ratio_list = []
        bmi_list_new, height_list_new, weight_list_new = [], [], []
        ratios = []
        for i, bmi in enumerate(bmi_list):
            gender = gender_list[i]
            added_ratio = 1
            if bmi > 23:
                added_ratio = (bmi - 23)**2
            
            ratio = added_ratio +1

            if gender == 'male':
                ind = height_indices_male[male_cnt]
                # ind2 = np.digitize(height_list[i], height_bins)
                # assert ind2 == ind
                male_cnt += 1
                assert ratio_male[ind] > 1, (height_list[i], ind)
                ratio = ratio * 10000.0 / (ratio_male[ind] + 1e-5)
                ratio_male_cnt[ind] += 1 / ratio_male[ind]
            else:
                ind = height_indices_female[female_cnt]
                # ind2 = np.digitize(height_list[i], height_bins)
                # assert ind2 == ind
                female_cnt += 1
                assert ratio_female[ind] > 1, (height_list[i], ind)
                ratio = ratio * 10000.0 / (ratio_female[ind] + 1e-5)
                ratio_female_cnt[ind] += 1 / ratio_female[ind]

            ratio_append = np.clip(ratio * 0.001, a_min=0.01, a_max=1)
            ratio_list.append(ratio_append)
            ratio = int(ratio_append*1000 + 0.5)
            bmi_list_new += [bmi] * ratio
            height_list_new += [height_list[i]] * ratio
            weight_list_new += [weight_list[i]] * ratio
            ratios.append(ratio * 0.001)

        # print('ratio range', np.array(ratios).max(), np.array(ratios).min(), np.array(ratios).mean())
        self.ratio_list = ratio_list
        # print('ratio_male_cnt, ratio_female_cnt', ratio_male_cnt, ratio_female_cnt)
        # plt.hist(bmi_list_new)
        # plt.savefig('tmp_ma_bmi_new.png')
        # plt.clf()
        # plt.hist(weight_list_new)
        # plt.savefig('tmp_ma_weight_new.png')
        # plt.clf()
        # plt.hist(height_list_new)
        # plt.savefig('tmp_ma_height_new.png')
        # plt.clf()
        # plt.hist(height_list_male)
        # plt.savefig('tmp_ma_height_male.png')
        # plt.clf()
        # plt.hist(height_list_female)
        # plt.savefig('tmp_ma_height_female.png')

        # # self.len_images_gender_list = len_images_gender_list
        # print('male_num, female_num, male_num_img, female_num_img', male_num, female_num, male_num_img, female_num_img)
        
        # len_images_list_new = []
        # for len_image in len_images_list:
        #     added_ratio = 1
        #     if len_image < 10:
        #         added_ratio = (10 / len_image)
        #         ratio = added_ratio
        #     elif len_image > 30:
        #         ratio = (30 / len_image)
        #     else:
        #         ratio = 1
            
        #     ratio = int(ratio * 20)
        #     len_images_list_new += [len_image] * ratio

        # plt.hist(len_images_list_new)
        # plt.savefig('tmp_len_images_new.png')
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
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 1), (2, 3), (4, 5), (7, 8), (9, 10), (11, 12))
    
    def cumsum(self, sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r
