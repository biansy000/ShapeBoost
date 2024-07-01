import numpy as np
import torch
from hybrik.models.layers.smpl.SMPL import SMPL_layer

def compute_pve_neutral_pose_scale_corrected(predicted_smpl_shape, target_smpl_shape, gendered_smpl, 
        neutral_smpl=None, pred_smpl_neutral_pose_vertices=None, return_vertices=False):
    """
    Given predicted and target SMPL shape parameters, computes neutral-pose per-vertex error
    after scale-correction (to account for scale vs camera depth ambiguity).
    :param predicted_smpl_parameters: predicted SMPL shape parameters tensor with shape (1, 10)
    :param target_smpl_parameters: target SMPL shape parameters tensor with shape (1, 10)
    :param gender: gender of target
    """
    # Get neutral pose vertices
    if pred_smpl_neutral_pose_vertices is None:
        pred_smpl_neutral_pose_output = neutral_smpl(
            betas=predicted_smpl_shape,
            pose_axis_angle=torch.zeros((1, 24, 3), device=predicted_smpl_shape.device),
            global_orient=None
        )
        pred_smpl_neutral_pose_vertices = pred_smpl_neutral_pose_output.vertices.cpu().numpy()

    # print(target_smpl_shape.shape, 'target_smpl_shape')
    target_smpl_neutral_pose_output = gendered_smpl(
        betas=target_smpl_shape,
        pose_axis_angle=torch.zeros((1, 24, 3), device=predicted_smpl_shape.device),
        global_orient=None
    )
    
    target_smpl_neutral_pose_vertices = target_smpl_neutral_pose_output.vertices.cpu().numpy()

    # Rescale such that RMSD of predicted vertex mesh is the same as RMSD of target mesh.
    # This is done to combat scale vs camera depth ambiguity.
    pred_smpl_neutral_pose_vertices_rescale = scale_and_translation_transform_batch(pred_smpl_neutral_pose_vertices,
                                                                                    target_smpl_neutral_pose_vertices)

    # Compute PVE-T-SC
    pve_neutral_pose_scale_corrected = np.linalg.norm(pred_smpl_neutral_pose_vertices_rescale
                                                      - target_smpl_neutral_pose_vertices,
                                                      axis=-1)  # (1, 6890)

    if return_vertices:
        return pve_neutral_pose_scale_corrected, pred_smpl_neutral_pose_vertices, target_smpl_neutral_pose_vertices
    return pve_neutral_pose_scale_corrected


def scale_and_translation_transform_batch(P, T):
    """
    First normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2], dtype=np.int16)


def segment_iou(seg1, seg2):
    intersect = np.ones_like(seg1)
    union = np.zeros_like(seg1)

    intersect[seg1 <= 0.5] = 0.0
    intersect[seg2 <= 0.5] = 0.0

    union[seg1 > 0.5] = 1.0
    union[seg2 > 0.5] = 1.0

    return intersect.sum() * 1.0 / (1e-5 + union.sum())