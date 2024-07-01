import numpy as np
import torch


COCO_JOINTS = {
    'Right Ankle': 16, 'Right Knee': 14, 'Right Hip': 12,
    'Left Hip': 11, 'Left Knee': 13, 'Left Ankle': 15,
    'Right Wrist': 10, 'Right Elbow': 8, 'Right Shoulder': 6,
    'Left Shoulder': 5, 'Left Elbow': 7, 'Left Wrist': 9,
    'Right Ear': 4, 'Left Ear': 3, 'Right Eye': 2, 'Left Eye': 1,
    'Nose': 0
}

# The SMPL model (im smpl_official.py) returns a large superset of joints.
# Different subsets are used during training - e.g. H36M 3D joints convention and COCO 2D joints convention.
# Joint label conversions from SMPL to H36M/COCO/LSP
ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # Using OP Hips
ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

# Joint label and body part seg label matching
# 24 part seg: COCO Joints
TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP = {19: 7,
                                          21: 7,
                                          20: 8,
                                          22: 8,
                                          4: 9,
                                          3: 10,
                                          12: 13,
                                          14: 13,
                                          11: 14,
                                          13: 14,
                                          5: 15,
                                          6: 16}
                                          
def convert_2Djoints_to_gaussian_heatmaps_torch(joints2D,
                                                img_wh,
                                                std=4):
    """
    :param joints2D: (B, N, 2) tensor - batch of 2D joints.
    :param img_wh: int, dimensions of square heatmaps
    :param std: standard deviation of gaussian blobs
    :return heatmaps: (B, N, img_wh, img_wh) - batch of 2D joint heatmaps (channels first).
    """
    device = joints2D.device

    xx, yy = torch.meshgrid(torch.arange(img_wh, device=device),
                            torch.arange(img_wh, device=device))
    xx = xx[None, None, :, :].float()
    yy = yy[None, None, :, :].float()

    j2d_u = joints2D[:, :, 0, None, None]  # Horizontal coord (columns)
    j2d_v = joints2D[:, :, 1, None, None]  # Vertical coord (rows)
    heatmap = torch.exp(-(((xx - j2d_v) / std) ** 2) / 2 - (((yy - j2d_u) / std) ** 2) / 2)
    return heatmap

