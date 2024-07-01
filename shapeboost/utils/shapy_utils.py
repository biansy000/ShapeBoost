import os.path as osp
import yaml
from typing import NewType, Tuple, Optional, Union, Dict, Callable, IO
import pickle as pk

from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
import torch
import torch.nn as nn
import smplx 

from shapeboost.beta_decompose.beta_process2 import mean_part_width_ratio, part_names, mean_bone_len

Tensor = NewType('Tensor', torch.Tensor)


class BodyMeasurements(nn.Module):

    # The density of the human body is 1050 kg / m^3
    DENSITY = 985

    def __init__(self, cfg, **kwargs):
        ''' Loss that penalizes deviations in weight and height
        '''
        super(BodyMeasurements, self).__init__()

        meas_definition_path = cfg.get('meas_definition_path', '')
        meas_definition_path = osp.expanduser(
            osp.expandvars(meas_definition_path))
        meas_vertices_path = cfg.get('meas_vertices_path', '')
        meas_vertices_path = osp.expanduser(
            osp.expandvars(meas_vertices_path))

        with open(meas_vertices_path, 'r') as f:
            meas_vertices = yaml.safe_load(f)

        head_top = meas_vertices['HeadTop']
        left_heel = meas_vertices['HeelLeft']

        left_heel_bc = left_heel['bc']
        self.left_heel_face_idx = left_heel['face_idx']

        left_heel_bc = torch.tensor(left_heel['bc'], dtype=torch.float32)
        self.register_buffer('left_heel_bc', left_heel_bc)

        head_top_bc = torch.tensor(head_top['bc'], dtype=torch.float32)
        self.register_buffer('head_top_bc', head_top_bc)

        self.head_top_face_idx = head_top['face_idx']

    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Human Body Density: {self.DENSITY}')
        return '\n'.join(msg)

    def _get_plane_at_heights(self, height: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        device = height.device
        batch_size = height.shape[0]

        verts = torch.tensor(
            [[-1., 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
            device=device).unsqueeze(dim=0).expand(batch_size, -1, -1).clone()
        verts[:, :, 1] = height.reshape(batch_size, -1)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device,
                             dtype=torch.long)

        return verts, faces, verts[:, faces]

    def compute_height(self, shaped_triangles: Tensor) -> Tuple[Tensor, Tensor]:
        ''' Compute the height using the heel and the top of the head
        '''
        head_top_tri = shaped_triangles[:, self.head_top_face_idx]
        head_top = (head_top_tri[:, 0, :] * self.head_top_bc[0] +
                    head_top_tri[:, 1, :] * self.head_top_bc[1] +
                    head_top_tri[:, 2, :] * self.head_top_bc[2])
        head_top = (
            head_top_tri * self.head_top_bc.reshape(1, 3, 1)
        ).sum(dim=1)
        left_heel_tri = shaped_triangles[:, self.left_heel_face_idx]
        left_heel = (
            left_heel_tri * self.left_heel_bc.reshape(1, 3, 1)
        ).sum(dim=1)

        return (torch.abs(head_top[:, 1] - left_heel[:, 1]),
                torch.stack([head_top, left_heel], axis=0)
                )

    def compute_mass(self, tris: Tensor) -> Tensor:
        ''' Computes the mass from volume and average body density
        '''
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def forward(
        self,
        triangles: Tensor,
        compute_mass: bool = True,
        compute_height: bool = True,
    ):
        measurements = {}
        if compute_mass:
            measurements['mass'] = {}
            mesh_mass = self.compute_mass(triangles)
            measurements['mass']['tensor'] = mesh_mass

        if compute_height:
            measurements['height'] = {}
            mesh_height, points = self.compute_height(triangles)
            measurements['height']['tensor'] = mesh_height
            measurements['height']['points'] = points

        return measurements



class Polynomial(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 15,
        degree: int = 2,
        alpha: float = 0.0,
    ) -> None:
        super(Polynomial, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha

        x = torch.rand([input_dim])
        y = torch.vander(x, degree)

        combinations = list(
            self._combinations(input_dim, degree, False, False))

        num_input = len(combinations)
        self.coeff_size = len(combinations)
        self.linear = nn.Linear(num_input, output_dim)
        for ii in range(degree):
            indices = []
            for c in combinations:
                if len(c) == ii + 1:
                    indices.append(c)
            indices = torch.tensor(indices, dtype=torch.long)
            # logger.info(f'{ii + 1} : {indices}')
            self.register_buffer(f'indices_{ii:03d}', indices)

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        start = int(not include_bias)
        return chain.from_iterable(combinations_w_r(range(n_features), i)
                                   for i in range(start, degree + 1))

    def build_polynomial_coeffs(self, X, degree=2):
        A = []
        for ii in range(self.degree):
            indices = getattr(self, f'indices_{ii:03d}')
            values = X[:, indices]
            A.append(torch.prod(values, dim=-1))

        A = torch.cat(A, dim=-1)
        return A

    # @staticmethod
    # def load_checkpoint(
    #     checkpoint_path: Union[str, IO],
    #     map_location: Optional[
    #         Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    #     strict: bool = True
    # ):
    #     ckpt_dict = torch.load(checkpoint_path)
    #     # hparams = ckpt_dict['hparams']

    #     # obj = Polynomial(**hparams)

    #     # obj.load_state_dict(ckpt_dict['model'])

    #     return obj

    def predict(self, X):
        to_numpy = False
        if not torch.is_tensor(X):
            to_numpy = True
            X = torch.from_numpy(X).to(dtype=torch.float32)
        output = self(X)
        # output = self.sk_polynomial.predict(X)
        if to_numpy:
            # return output
            return output.detach().cpu().numpy()
        else:
            # return torch.from_numpy(output).to(dtype=torch.float32)
            return output

    def forward(self, x):
        A = self.build_polynomial_coeffs(x, degree=self.degree)
        output = self.linear(A)
        return output



body_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'spine2', 'spine3']
body_index = [part_names.index(item) for item in body_names]

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = (2.**torch.linspace(0., max_freq, steps=N_freqs)) * np.pi
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs) * np.pi
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim

        # print(len(self.embed_fns), freq_bands)
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=4, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 10,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class Beta2Meausrement(nn.Module):
    def __init__(self, inp_dim=12*9, out_dim=3):
        super(Beta2Meausrement, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(inp_dim, 512),
            nn.Tanh(),
            nn.Linear(512, out_dim),
        )

        self.embedder = get_embedder()
        body_width_ratio = torch.tensor(mean_part_width_ratio).float()[body_index]
        body_bone_len = torch.tensor(mean_bone_len).float()[body_index]
        self.register_buffer('body_normed_ratio', torch.cat([body_width_ratio, body_bone_len], dim=0) )

    def forward(self, part_widths, bone_len):
        # x: part_width, bone_len
        # return: chest, waist, hips
        body_normed_ratio = self.body_normed_ratio.clone()
        body_normed_ratio[:6] = body_normed_ratio[:6] * body_normed_ratio[6:]

        x = torch.cat(
            [part_widths[:, body_index], bone_len[:, body_index]], dim=1
        )
        x_normed = x / body_normed_ratio / 3

        x_normed_embedding = self.embedder[0](x_normed)
        return self.layer(x_normed_embedding)


DEFAULT_BODY_MODEL_FOLDER = 'data/pretrained_models/shapy_models/body_models'
DEFAULT_POINT_REG_SMPLX = 'data/pretrained_models/shapy_models/utility_files/evaluation/eval_point_set/HD_SMPLX_from_SMPL.pkl'
DEFAULT_POINT_REG_SMPL = 'data/pretrained_models/shapy_models/utility_files/evaluation/eval_point_set/HD_SMPL_sparse.pkl'
class Converter(nn.Module):
    def __init__(self):
        super(Converter, self).__init__()
        self.smpl = smplx.create(
            model_path=DEFAULT_BODY_MODEL_FOLDER,
            model_type='smpl'
        )

        self.smplx = smplx.create(
            model_path=DEFAULT_BODY_MODEL_FOLDER,
            model_type='smplx'
        )

        point_reg_gt = DEFAULT_POINT_REG_SMPLX
        point_reg_fit = DEFAULT_POINT_REG_SMPL
        with open(point_reg_gt, 'rb') as f:
            self.point_regressor_smplx = pk.load(f)

        with open(point_reg_fit, 'rb') as f:
            self.point_regressor_smpl = pk.load(f)

        # print(type(self.point_regressor_smpl))
        self.smpl_pr_x_sd = self.point_regressor_smpl.dot(self.smpl.shapedirs.cpu().numpy().reshape(6890, 30)).reshape(20000, 3, 10)
        self.smplx_pr_x_sd = self.point_regressor_smplx.dot(self.smplx.shapedirs.cpu().numpy().reshape(-1, 30)).reshape(20000, 3, 10)
        self.register_buffer('point_regressor_smpl_tensor', torch.from_numpy(self.point_regressor_smpl.toarray()).float() )
        print('point_regressor_smpl_tensor', self.point_regressor_smpl_tensor.shape)

        smpl_pr_x_t = self.point_regressor_smpl.dot(self.smpl.v_template.cpu().numpy().reshape(-1, 3)) # 20000 x 3
        smplx_pr_x_t = self.point_regressor_smplx.dot(self.smplx.v_template.cpu().numpy().reshape(-1, 3))

        self.smpl_pr_x_t = smpl_pr_x_t
        self.smplx_pr_x_t = smplx_pr_x_t
        self.pr_x_t_diff = smplx_pr_x_t - smpl_pr_x_t

        self.num_v_smplx = len(self.smplx.shapedirs)
        self.num_v_smpl = len(self.smpl.shapedirs)

        self.register_buffer('smpl_pr_x_sd_tensor', torch.from_numpy(self.smpl_pr_x_sd.reshape(20000*3, 10)).float() )
        self.register_buffer('smplx_pr_x_sd_tensor', torch.from_numpy(self.smplx_pr_x_sd.reshape(20000*3, 10)).float() )
        # self.register_buffer('smpl_pr_x_t_tensor', torch.from_numpy(smpl_pr_x_t).float())
        self.register_buffer('smplx_pr_x_t_tensor', torch.from_numpy(smplx_pr_x_t).float())
        self.register_buffer('pr_x_t_diff_tensor', torch.from_numpy(self.pr_x_t_diff.reshape(20000*3)).float() )

        smpl2smplx_A = np.zeros((20000*3, 13))
        smpl2smplx_A[:, :10] = self.smplx_pr_x_sd.reshape(20000*3, 10)

        smpl2smplx_A[::3, 10] = 1
        smpl2smplx_A[1::3, 11] = 1
        smpl2smplx_A[2::3, 12] = 1

        smpl2smplx_A_s = np.matmul(smpl2smplx_A.T, smpl2smplx_A)

        smpl2smplx_A_s_inv = np.linalg.inv(smpl2smplx_A_s)

        self.register_buffer('smpl2smplx_A_s_inv', torch.from_numpy(smpl2smplx_A_s_inv).float())
        self.register_buffer('smpl2smplx_A_T', torch.from_numpy(smpl2smplx_A.T).float())
    
    def SMPLb2SMPLXb_np(self, smpl_beta):
        # shapedir: N x 3 x 10, point_regressor: n x N
        A = np.zeros((20000*3, 13))
        A[:, :10] = self.smplx_pr_x_sd.reshape(20000*3, 10)

        A[::3, 10] = 1
        A[1::3, 11] = 1
        A[2::3, 12] = 1

        b = np.matmul(self.smpl_pr_x_sd.reshape(20000*3, 10), smpl_beta) - self.pr_x_t_diff.reshape(20000*3)

        A_s = np.dot(A.T, A)
        b_s = np.dot(A.T, b)

        solu = np.linalg.solve(A_s, b_s)

        smplx_beta = solu[:10]

        return smplx_beta
    
    def SMPLXb2SMPLb_np(self, smplx_beta):
        # shapedir: N x 3 x 10, point_regressor: n x N
        A = np.zeros((20000*3, 13))
        A[:, :10] = self.smpl_pr_x_sd.reshape(20000*3, 10)

        A[::3, 10] = 1
        A[1::3, 11] = 1
        A[2::3, 12] = 1

        b = np.matmul(self.smplx_pr_x_sd.reshape(20000*3, 10), smplx_beta) + self.pr_x_t_diff.reshape(20000*3)

        A_s = np.dot(A.T, A)
        b_s = np.dot(A.T, b)

        solu = np.linalg.solve(A_s, b_s)

        smpl_beta = solu[:10]

        return smpl_beta
    
    def SMPLb2SMPLXb_torch(self, smpl_beta):
        b = torch.einsum('nl,bl->bn', self.smpl_pr_x_sd_tensor, smpl_beta) - self.pr_x_t_diff_tensor # b x (20000*3)
        b_s = torch.einsum('ln,bn->bl', self.smpl2smplx_A_T, b) # b x 13

        solution = torch.einsum('ij,bj->bi', self.smpl2smplx_A_s_inv, b_s)
        smplx_beta = solution[:, :10]
        return smplx_beta
    
    def SMPLv2SMPLXb_torch(self, smpl_vshaped):
        b = torch.einsum('nl,blk->bnk', self.point_regressor_smpl_tensor, smpl_vshaped) - self.smplx_pr_x_t_tensor # b x (20000*3)
        b_s = torch.einsum('ln,bn->bl', self.smpl2smplx_A_T, b.reshape(-1, 20000*3)) # b x 13

        solution = torch.einsum('ij,bj->bi', self.smpl2smplx_A_s_inv, b_s)
        smplx_beta = solution[:, :10]
        return smplx_beta



def point_error(x, y, align=True):
    t = 0.0
    if align:
        t = x.mean(0, keepdims=True) - y.mean(0, keepdims=True)

    x_hat = x - t
    error = np.sqrt(np.power(x_hat - y, 2).sum(axis=-1))

    return error.mean().item()


class HBWErrorCalculator:
    def __init__(self):
        self.hbw_folder = 'data/HBW'
        point_reg_gt = 'data/pretrained_models/shapy_models/utility_files/evaluation/eval_point_set/HD_SMPLX_from_SMPL.pkl'
        point_reg_fit = 'data/pretrained_models/shapy_models/utility_files/evaluation/eval_point_set/HD_SMPL_sparse.pkl'
        with open(point_reg_gt, 'rb') as f:
            self.point_regressor_gt = pk.load(f)

        with open(point_reg_fit, 'rb') as f:
            self.point_regressor_fit = pk.load(f)
        
        return
    
    def calc_error(self, label, pred_vshaped, smplx=False):
        split, subject, _, img_fn = label.split('/')[-4:]
        subject_id_npy = subject.split('_')[0] + '.npy'
        v_shaped_gt_path = osp.join(self.hbw_folder, 'smplx', split, subject_id_npy)
        v_shaped_gt = np.load(v_shaped_gt_path)

        # cast v-shaped
        v_shaped_gt = v_shaped_gt.astype(np.float32)
        v_shaped_fit = pred_vshaped.astype(np.float32)

        # compute P2P-20k error
        if not smplx:
            points_fit = self.point_regressor_fit.dot(v_shaped_fit)
        else:
            points_fit = self.point_regressor_gt.dot(v_shaped_fit)
        
        points_gt = self.point_regressor_gt.dot(v_shaped_gt)

        p2p_error = point_error(points_gt, points_fit, align=True)
        return p2p_error
